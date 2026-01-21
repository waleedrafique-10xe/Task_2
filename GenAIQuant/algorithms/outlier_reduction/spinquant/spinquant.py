from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import datasets
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from GenAIQuant.algorithms.outlier_reduction.quarot.had_utils import get_hadK
from GenAIQuant.algorithms.outlier_reduction.quarot.quarot import handle_qwen_patch_embed
from GenAIQuant.algorithms.outlier_reduction.quarot.qutils import get_rotation_matrix
from GenAIQuant.interfaces import RotationMatrices
from GenAIQuant.model_preparer.utils import ModelType
from GenAIQuant.utils import get_nested_attr
from GenAIQuant.algorithms.datasets.dataset import DatasetUtils
# from GenAIQuant.utils.dataset_utils import DatasetProvider

from ..base import OutlierReduction
from .optimizer import SGDG
from .quant_utils import (
    ActQuantWrapper,
    add_actquant,
    add_hooks,
    find_qlayers,
    rtn_fwrd,
)
from .utils import (
    CustomJsonDataset,
    DownProjHook,
    OfflineRotation,
    OnlineRotation,
    RotateModule,
    CustomDataset,
)

if TYPE_CHECKING:
    from ....model_preparer import Model


def llama_down_proj_groupsize(model, groupsize):
    assert groupsize > 1, "groupsize should be greater than 1!"

    if model.config.intermediate_size % groupsize == 0:
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert groupsize * group_num == model.config.hidden_size, (
        "Invalid groupsize for llama!"
    )

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert down_proj_groupsize * group_num == model.config.intermediate_size, (
        "Invalid groupsize for down_proj!"
    )
    return down_proj_groupsize


class SpinQuant(OutlierReduction):
    def __init__(self, config: "Config"):
        super().__init__(config)
        self.rotate_mode = self.config.rotate_mode
        self.online = self.config.online
        self.device = config.device
        self.seqlen = 2048

        self._counter = 0

    def register_online_hooks(
        self,
        name: str,
        module: torch.nn.Module,
        model: Model,
        hidden_size: int,
        num_attention_heads: int,
    ):
        split = ".".join(name.rsplit(".", 2)[-2:])
        rotate_v_layers = model.meta.online_rotations["rotate_v"]
        rotate_output_layers = model.meta.online_rotations["rotate_output"]

        if split in rotate_v_layers:
            OnlineRotation.rotate_v(
                module.weight,
                model.model_config.hidden_size
                // model.model_config.num_attention_heads,
                Qs[".".join(name.split(".")[:3]) + ".self_attn.R2"],
            )
        elif split in rotate_output_layers:
            parent_module_name = ".".join(name.split(".")[:-1])
            curr_module_name = name.split(".")[-1]
            parent_module = get_nested_attr(model.model, parent_module_name)

            if self._counter % 2 == 0:
                OnlineRotation.rotate_o(
                    module.weight,
                    model.model_config.hidden_size
                    // model.model_config.num_attention_heads,
                    Qs[".".join(name.split(".")[:3]) + ".self_attn.R2"],
                )
            else:
                OnlineRotation.rotate_d(module.weight)
                setattr(
                    parent_module,
                    curr_module_name,
                    DownProjHook(
                        module,
                        model.model_config.text_config.intermediate_size,  # TODO: Figure this out
                    ),
                )
            self._counter += 1

    def apply_wrappers(self, model, meta):
        model.eval()
        args = self.config

        vision_model_name = getattr(meta, "vision_model_name", None)
        for name, module in model.named_modules():
            if (
                "down" in name
                and "module" in name
                and not (
                    isinstance(vision_model_name, str) and vision_model_name in name
                )
            ):
                OnlineRotation.rotate_d(module.weight)

        add_actquant(model)

        for name, module in model.named_modules():
            if "down" in name and not (
                isinstance(vision_model_name, str) and vision_model_name in name
            ):
                had, K = get_hadK(model.config.intermediate_size)
                module.online_full_had = True
                module.had_K = had
                module.K = K
                module.fp32_had = args.fp32_had

        rtn_fwrd(model, meta, self.device, args)

        qlayers = find_qlayers(model, layers=[ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = llama_down_proj_groupsize(model, args.a_groupsize)

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            head_dim = model.config.hidden_size // model.config.num_attention_heads

            if "v_proj" in name and args.v_bits < 16:
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name:
                layer_groupsize = head_dim

            if "lm_head" in name:
                layer_input_bits = 16

            if "down_proj" in name:
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    def optimize_matrices(self, model):
        training_arguments = TrainingArguments(
            "./output",
            gradient_checkpointing=True,
            learning_rate=1.5,
            logging_steps=1.0,
            lr_scheduler_type="cosine",
            max_steps=self.config.max_steps,
            save_safetensors=False,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            torch_empty_cache_steps=1,
        )
        original_model = model.model
        original_model.seqlen = self.seqlen
        # TODO: pick this from interface -- don't need to reload
        tokenizer = AutoTokenizer.from_pretrained(model.model_id)
        add_hooks(original_model, model.meta)

        layers = None
        if model.meta.model_type == ModelType.VLM:
            layers = original_model.language_model.layers
        elif model.meta.model_type == ModelType.LLM:
            layers = original_model.model.layers

        for idx, layer in enumerate(layers):
            for sub_name, sub_module in layer.named_modules():
                sub_module.layer_idx = idx

        self.apply_wrappers(original_model, model.meta)
        for param in original_model.parameters():
            param.requires_grad = False

        Q = get_rotation_matrix(
            model.model_config.hidden_size, self.rotate_mode, self.device
        )
        original_model.R1 = RotateModule(Q, self.device)
        for i in range(model.model_config.num_hidden_layers):
            Q2 = get_rotation_matrix(
                model.model_config.hidden_size
                // model.model_config.num_attention_heads,
                self.rotate_mode,
                self.device,
            )
            layers[i].self_attn.R2 = RotateModule(Q2, self.device)

        original_model.config.use_cache = False
        # calibration_datasets = datasets.load_dataset(
        #     "Salesforce/wikitext", "wikitext-2-raw-v1"
        # )
        # train_data = CustomJsonDataset(
        #     calibration_datasets["train"],
        #     tokenizer,
        #     block_size=self.seqlen,
        # )

        train_data = DatasetUtils.get_lm_dataset(
                tokenizer=tokenizer,
                dataset_name='wikitext',
                num_samples=self.seqlen * 3, # This can be increased, thus needs discussion what should be the sample size
                split="train",
                seqlen=self.seqlen,
                return_only_inputs=True
            )

        train_data_cls = CustomDataset(train_data)
        trainable_parameters = [original_model.R1.weight] + [
            layers[i].self_attn.R2.weight
            for i in range(model.model_config.num_hidden_layers)
        ]

        optimizer = SGDG(trainable_parameters, lr=1.5, stiefel=True)
        trainer = Trainer(
            model=original_model,
            tokenizer=tokenizer,
            train_dataset=train_data_cls,
            args=training_arguments,
            data_collator=default_data_collator,
            optimizers=(optimizer, None),
        )
        trainer.train()

        Qs = {
            key.replace(".weight", ""): value
            for key, value in trainer.model.state_dict().items()
            if "R1.weight" in key or "self_attn.R2" in key
        }
        model.model = original_model
        return Qs

    def forward(
        self, interface: "ModuleInterface", verbose: bool = True, **kwargs
    ) -> "ModuleInterface":
        model: "Model" = interface.model

        # assume layer norms are folded at this point
        assert model.layernorm_fused, (
            "Require that layer norms need to be folded for rotations"
        )

        assert model.model.device.type == "cpu"

        new_model = deepcopy(model)
        new_model.gpu()

        Qs = self.optimize_matrices(new_model)

        del new_model
        # model.gpu()

        if interface.rotation_matrices is None:
            interface.rotation_matrices = RotationMatrices()

        vision_hidden_size = None
        vision_num_heads = None

        if model.meta.model_type == ModelType.VLM:
            text_hidden_size = model.model_config.text_config.hidden_size
            vision_hidden_size = model.model_config.vision_config.hidden_size

            # Modelled after Qwen 2.5 VL -- might fail for other VLMs because
            # hugging face does not guarantee a uniform scheme for configs across
            # models -- TODO: Figure out how to handle other models in this case
            text_num_heads = model.model_config.text_config.num_attention_heads
            vision_num_heads = model.model_config.num_attention_heads

        else:
            text_hidden_size = model.model_config.hidden_size
            text_num_heads = model.model_config.num_attention_heads

        # TODO: should make Q rotations part of interface or model

        # text_Q = get_rotation_matrix(text_hidden_size, self.rotate_mode, "cpu")
        text_Q = Qs["R1"].to("cpu")
        vision_Q = (
            get_rotation_matrix(vision_hidden_size, self.rotate_mode, "cpu")
            if vision_hidden_size is not None
            else None
        )

        interface.rotation_matrices.text_R1 = text_Q
        interface.rotation_matrices.text_R2 = Qs
        interface.rotation_matrices.vision_R1 = vision_Q

        rotate_forward_layers = model.meta.offline_rotations["rotate_forward"]
        rotate_inverse_layers = model.meta.offline_rotations["rotate_inverse"]

        self._counter = 0  # make sure the counter is zero

        if model.model_config.model_type == "qwen2_5_vl":
            assert vision_Q is not None
            assert vision_hidden_size is not None
            handle_qwen_patch_embed(model.model, vision_Q, vision_hidden_size, "cpu")

        for name, module in tqdm(list(model.model.named_modules())):
            Q = text_Q
            hidden_size = text_hidden_size
            num_attention_heads = text_num_heads

            if name == "model.visual.patch_embed":
                continue

            # TODO: Need to handle special case for Qwen 2.5 VL merger submodules
            # where merger.mlp.0 is a different size than the rotation matrix.
            #
            # To handle this we can divide the weights of blocks and rotate each block
            # separetly

            if (
                model.meta.vision_model_name is not None
                and model.meta.vision_model_name in name
            ):
                Q = vision_Q
                assert vision_hidden_size is not None
                assert vision_num_heads is not None
                hidden_size = vision_hidden_size
                num_attention_heads = vision_num_heads

            # all nn.Embedding layers should be forward rotated
            if isinstance(module, torch.nn.Embedding):
                # logger.info(f"Forward Rotation on `{name}`")
                OfflineRotation.rotate_forward(module.weight, Q)  # type: ignore
                continue

            split = ".".join(name.rsplit(".", 2)[-2:])

            # we only rotate nn.Linear and nn.Embeddings layers
            if not isinstance(module, torch.nn.Linear):
                continue

            if split in rotate_forward_layers:
                # logger.info(f"Forward Rotation on `{name}`")
                # print(name, module.weight.shape, Q.shape)
                OfflineRotation.rotate_forward(module.weight, Q)  # type: ignore

            elif split in rotate_inverse_layers:
                # logger.info(f"Inverse Rotation on `{name}`")
                if "merger" in name:
                    print(f"{text_Q.shape = }")
                    print(f"{module.weight.shape = }")
                    OfflineRotation.rotate_inverse(module.weight, text_Q)  # type: ignore

                else:
                    OfflineRotation.rotate_inverse(module.weight, Q)  # type: ignore

                if module.bias is not None:
                    # logger.info(f"Rotating bias of `{name}`")
                    dtype = module.bias.dtype
                    if "merger" in name:
                        module.bias.data = (module.bias.double() @ text_Q).to(
                            dtype=dtype
                        )
                    else:
                        module.bias.data = (module.bias.double() @ Q).to(dtype=dtype)

            vision_model_name = getattr(model.meta, "vision_model_name", None)
            if isinstance(vision_model_name, str) and vision_model_name in name:
                continue

            split = ".".join(name.rsplit(".", 2)[-2:])
            rotate_v_layers = model.meta.online_rotations["rotate_v"]
            rotate_output_layers = model.meta.online_rotations["rotate_output"]

            if split in rotate_v_layers:
                OnlineRotation.rotate_v(
                    module.weight,
                    model.model_config.hidden_size
                    // model.model_config.num_attention_heads,
                    Qs[".".join(name.split(".")[:-2]) + ".self_attn.R2"],
                )
                if module.bias is not None:
                    dtype = module.bias.dtype
                    t_shape = module.bias.shape
                    had_dim = (
                        model.model_config.hidden_size
                        // model.model_config.num_attention_heads
                    )
                    module.bias.data = (
                        (
                            module.bias.reshape(
                                -1, t_shape[-1] // had_dim, had_dim
                            ).double()
                            @ Qs[".".join(name.split(".")[:-2]) + ".self_attn.R2"].to(
                                module.bias.device
                            )
                        )
                        .reshape(t_shape)
                        .to(dtype=dtype)
                    )
            elif split in rotate_output_layers:
                if self._counter % 2 == 0:
                    OnlineRotation.rotate_o(
                        module.weight,
                        model.model_config.hidden_size
                        // model.model_config.num_attention_heads,
                        Qs[".".join(name.split(".")[:-2]) + ".self_attn.R2"],
                    )
                self._counter += 1

            # if self.online:
            # self.register_online_hooks(
            #     name, module, model, hidden_size, num_attention_heads
            # )

        model.cpu()

        interface.model = model
        return interface
