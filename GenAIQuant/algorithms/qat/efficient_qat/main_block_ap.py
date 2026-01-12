import os
import re
import time
from pathlib import Path

import torch

from GenAIQuant.logger import logger
# from GenAIQuant.utils.dataset_utils_vl import get_vl_dataset

from .block_ap import block_ap
# from .datautils_block import get_dataset
from GenAIQuant.algorithms.datasets.dataset import DatasetUtils
from .utils import find_layers

# from .quantizer.int_linear_real import load_quantized_model


def _sanitize_cache_component(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value)


def main_block(config, interface, tokenizer, processor=None):
    model = interface.model
    qat_config = config.quantization.qat
    model_id = model.model_id

    is_multimodal = bool(qat_config.multimodal)

    target_dtype = qat_config.target_dtype
    if not isinstance(target_dtype, torch.dtype):
        raise RuntimeError("Target dtype could not be resolved for Efficient QAT.")
    logger.info("Using target dtype %s for Efficient QAT", target_dtype)

    if is_multimodal:
        if processor is None:
            raise ValueError("Processor is required for multimodal calibration.")
        tokenizer_ref = getattr(processor, "tokenizer", None)
    else:
        tokenizer_ref = tokenizer

    pad_token_id = getattr(tokenizer_ref, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer_ref, "pad_token_type_id", 0)
    pad_token_id = pad_token_id if pad_token_id is not None else 0

    layers_name_prefix = model.meta.layers
    bit_config = {}

    if interface.bit_allocation is not None:
        logger.info(
            "Found Bit Allocation Config in `interface`. Using"
            " `interface.bit_allocation` for Efficient QAT"
        )
        bit_config = interface.bit_allocation
        bit_config["layers_name_prefix"] = layers_name_prefix
    else:
        logger.info(
            "No Bit Allocation Config found in `interface` "
            f"using bit_width={config.quantization.qat.wbits} for Efficient QAT"
        )

        bit_config["layers_name_prefix"] = layers_name_prefix
        for i in range(len(model.layers)):
            bit = config.quantization.qat.wbits
            layer_name = f"{layers_name_prefix}.{i}"
            layer = model.layers[i]
            full = find_layers(layer)
            sequential = [list(full.keys())]
            for names in sequential:
                subset = {n: full[n] for n in names}
                for name in subset:
                    full_name = layer_name + "." + name
                    bit_config[full_name] = bit

    # init logger and output dirs
    if config.output_dir:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    if qat_config.cache_dir:
        Path(qat_config.cache_dir).mkdir(parents=True, exist_ok=True)
    if qat_config.save_quant_dir:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    if qat_config.net is None:
        qat_config.net = model_id.split("/")[-1]
        logger.info(f"net is None, setting as {qat_config.net}")

    # Freeze all model parameters (no gradient updates)
    for param in model.model.parameters():
        param.requires_grad = False

        # quantization

    logger.info("=== start quantization ===")
    logger.info(
        "EQAT: using calibration dataset '%s' (train_size=%d, val_size=%d)",
        qat_config.calib_dataset,
        qat_config.train_size,
        qat_config.val_size,
    )
    tick = time.time()
    # load calibration dataset
    dataset_tag = _sanitize_cache_component(str(qat_config.calib_dataset))
    cache_prefix = (
        f"{qat_config.cache_dir}/"
        f"dataloader_{qat_config.net}_{dataset_tag}_{qat_config.train_size}_{qat_config.val_size}_{qat_config.training_seqlen}"
    )
    cache_trainloader = f"{cache_prefix}_train.cache"
    cache_valloader = f"{cache_prefix}_val.cache"

    # Load cached dataloaders if available
    if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
        trainloader = torch.load(cache_trainloader)
        logger.info(f"load trainloader from {cache_trainloader}")
        valloader = torch.load(cache_valloader)
        logger.info(f"load valloader from {cache_valloader}")
    else:
        if is_multimodal:
            trainloader, valloader = DatasetUtils.get_vl_dataset(
                processor,
                qat_config.calib_dataset,
                qat_config.train_size,
                qat_config.val_size,
                seed=getattr(qat_config, "seed", 0),
                max_length=qat_config.training_seqlen,
            )
        else:
            trainloader, valloader = DatasetUtils.get_dataset(
                tokenizer,
                qat_config.calib_dataset,
                qat_config.train_size,
                qat_config.val_size,
                getattr(qat_config, "seed", 0),
                qat_config.training_seqlen,
                test_only=False,
            )
        torch.save(trainloader, cache_trainloader)
        torch.save(valloader, cache_valloader)

    block_ap(
        model=model,
        config=qat_config,
        pad_token_id=pad_token_id,
        wbits_config=bit_config,
        trainloader=trainloader,
        valloader=valloader,
        device=config.device,
        logger=logger,
    )

    logger.info(f"Time taken for QAT: {(time.time() - tick):.2f}s")

    # final clean-up and save
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if qat_config.save_quant_dir:
        logger.info("Saving model after Block_AP")
        model_path = config.output_dir / f"{model.model_id}_block_ap"
        model.model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    return model
