from types import SimpleNamespace
from typing import Dict

import torch
import transformers
from bitsandbytes.optim import AdamW as BnbAdamW
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed

from GenAIQuant.logger import logger

from .datautils_e2e import make_data_module
from .quantizer.int_linear_fake import QuantLinear

DEFAULT_PAD_TOKEN = "[PAD]"
SEED = 0


def e2e_qp(model, tokenizer, config):
    """Performs e2e Quantization after blockwise quantization
    args:
        model: model object of type Model Preparer
        tokenizer: huggingface tokenizer
        config: Config object passed by user
    """
    qat_config = config.quantization.qat
    e2e_config = SimpleNamespace(**qat_config.e2e)

    target_dtype = qat_config.target_dtype
    if not isinstance(target_dtype, torch.dtype):
        raise RuntimeError("Target dtype could not be resolved for Efficient QAT E2E.")
    e2e_config.bf16 = target_dtype == torch.bfloat16
    logger.info("Using target dtype %s for E2E", target_dtype)

    if config.device == "cpu":
        model.cpu()
    else:
        model.gpu()

    model.model.to(dtype=target_dtype)

    tokenizer.model_max_length = e2e_config.pt_context_len

    setattr(model.model, "model_parallel", True)
    setattr(model.model, "is_parallelizable", True)

    model.model.config.torch_dtype = target_dtype

    model.model.train()

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model.model,
        )

    special_tokens = {}
    eos_token = _resolve_token_string(
        tokenizer=tokenizer, token_id=model.model.config.eos_token_id
    )
    if eos_token is not None:
        special_tokens["eos_token"] = eos_token

    bos_token = _resolve_token_string(
        tokenizer=tokenizer, token_id=model.model.config.bos_token_id
    )
    if bos_token is not None:
        special_tokens["bos_token"] = bos_token

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)

    for _, param in model.model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if e2e_config.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )
        model.model.gradient_checkpointing_enable()

    if e2e_config.bf16:
        for name, module in model.model.named_modules():
            if "norm" in name or "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight") and module.weight.dtype != target_dtype:
                    module.to(target_dtype)

    model.model_config.use_cache = False  # disable caching for training
    logger.info("Blockwise Quantized model loaded")
    seed = getattr(e2e_config, "seed", SEED)
    set_seed(seed)
    data_module = make_data_module(tokenizer=tokenizer, args=e2e_config)

    scale_params: list[torch.nn.Parameter] = []
    for name, module in model.model.named_modules():
        if isinstance(module, QuantLinear) and "head" not in name:
            scale_param = getattr(module.weight_quantizer, "scale", None)
            if isinstance(scale_param, torch.nn.Parameter):
                scale_param.requires_grad = True
                scale_params.append(scale_param)
    if not scale_params:
        logger.warning("No QuantLinear scale parameters found for E2E optimization.")
    optimizer_grouped_parameters = [
        {
            "params": scale_params,
            "weight_decay": 0.0,
            "lr": e2e_config.learning_rate,
        }
    ]
    optimizer = BnbAdamW(optimizer_grouped_parameters)

    use_cuda = torch.cuda.is_available() and config.device != "cpu"
    enable_fp16 = use_cuda and target_dtype == torch.float16
    enable_bf16 = use_cuda and target_dtype == torch.bfloat16

    per_device_train_batch_size = getattr(e2e_config, "per_device_train_batch_size", 1)
    grad_acc_steps = getattr(e2e_config, "gradient_accumulation_steps", 16)
    max_grad_norm = getattr(e2e_config, "max_grad_norm", 0.3)
    lr_scheduler_type = getattr(e2e_config, "lr_scheduler_type", "cosine")
    warmup_ratio = getattr(e2e_config, "warmup_ratio", 0.03)
    logging_steps = getattr(e2e_config, "logging_steps", 10)
    predict_with_generate = getattr(e2e_config, "predict_with_generate", False)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=e2e_config.learning_rate,
        weight_decay=0.0,
        per_device_train_batch_size=per_device_train_batch_size,
        predict_with_generate=predict_with_generate,
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        gradient_accumulation_steps=grad_acc_steps,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        fp16=enable_fp16,
        bf16=enable_bf16,
    )
    # Step 8: Initialize Hugging Face trainer
    trainer = Seq2SeqTrainer(
        model=model.model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(optimizer, None),
        **{
            k: v for k, v in data_module.items() if k != "predict_dataset"
        },  # exclude test dataset
    )

    print_trainable_parameters(e2e_config, model.model)

    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # final clean-up and save
    torch.cuda.empty_cache()
    if qat_config.save_quant_dir:
        logger.info("Saving model after E2E")
        model_path = config.output_dir / f"{model.model_id}_e2e"
        model.model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    model.model.to(target_dtype)
    return model


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def _resolve_token_string(tokenizer: transformers.PreTrainedTokenizer, token_id):
    """Return a valid token string for tokenizer.add_special_tokens."""
    if token_id is None:
        return None
    if isinstance(token_id, (list, tuple)):
        if len(token_id) == 0:
            return None
        token_id = token_id[0]
    token = tokenizer.convert_ids_to_tokens(token_id)
    if isinstance(token, list):
        return token[0] if token else None
    return token


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print("*" * 80)
    if args.wbits == 4:
        trainable_params /= 2
    logger.info(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )
