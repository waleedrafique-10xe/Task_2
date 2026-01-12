from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from tqdm import tqdm

from GenAIQuant.interfaces import RotationMatrices
from GenAIQuant.logger import logger
from GenAIQuant.model_preparer.utils import ModelType
from GenAIQuant.utils import get_nested_attr

from ..base import OutlierReduction
from ..rotaters import (
    DownProjHook,
    OfflineRotation,
    OnlineRotation,
    OutProjHook,
)
from .qutils import get_rotation_matrix

if TYPE_CHECKING:
    from torch import nn

    from GenAIQuant.config import Config
    from GenAIQuant.interfaces import ModuleInterface
    from GenAIQuant.model_preparer import Model


def handle_qwen_patch_embed(
    model: nn.Module,
    vision_Q: torch.Tensor,
    visual_hidden_size: int,
    device: str = "cpu",
):
    logger.info("Registering forward hook on `model.visual.patch_embed`")
    old_forward = model.visual.patch_embed.forward
    rotation = nn.Linear(
        visual_hidden_size,
        visual_hidden_size,
        False,
        device=device,
        dtype=torch.bfloat16,
    )

    rotation.load_state_dict({"weight": vision_Q.T.bfloat16()})
    model.visual.patch_embed.register_module("rotation", rotation)

    def new_forward(input_tensor):
        output = old_forward(input_tensor)
        return model.visual.patch_embed.rotation(output)

    model.visual.patch_embed.forward = new_forward


class QuaRot(OutlierReduction):
    def __init__(self, config: "Config"):
        super().__init__(config)

        self.online = self.config.online
        self.rotate_mode = self.config.rotate_mode

        self.device = config.device

        self._counter = 0

    def register_online_hooks(
        self,
        name: str,
        module: nn.Module,
        model: Model,
        hidden_size: int,
        num_attention_heads: int,
    ):
        split = ".".join(name.rsplit(".", 2)[-2:])
        rotate_v_layers = model.meta.online_rotations["rotate_v"]
        rotate_output_layers = model.meta.online_rotations["rotate_output"]

        if split in rotate_v_layers:
            OnlineRotation.rotate_v(
                module.weight,  # type: ignore
                model.model_config.text_config.hidden_size
                // model.model_config.text_config.num_attention_heads,
            )
        elif split in rotate_output_layers:
            OnlineRotation.rotate_output(module.weight)  # type: ignore

            parent_module_name = ".".join(name.split(".")[:-1])
            curr_module_name = name.split(".")[-1]
            parent_module = get_nested_attr(model.model, parent_module_name)

            if self._counter % 2 == 0:
                setattr(
                    parent_module,
                    curr_module_name,
                    OutProjHook(module, hidden_size, num_attention_heads),
                )
            else:
                setattr(
                    parent_module,
                    curr_module_name,
                    DownProjHook(
                        module,
                        model.model_config.text_config.intermediate_size,  # TODO: Figure this out
                    ),
                )
            self._counter += 1

    def forward(
        self, interface: "ModuleInterface", verbose: bool = True, **kwargs
    ) -> "ModuleInterface":
        model: "Model" = interface.model
        model_config = model.model_config

        if interface.rotation_matrices is None:
            interface.rotation_matrices = RotationMatrices()

        assert model.layernorm_fused, (
            "Requires that layer norms need to be folded for rotations"
        )

        vision_hidden_size = None
        vision_num_heads = None

        if model.meta.model_type == ModelType.VLM:
            text_hidden_size = model_config.text_config.hidden_size
            vision_hidden_size = model_config.vision_config.hidden_size

            # Modelled after Qwen 2.5 VL -- might fail for other VLMs because
            # hugging face does not guarantee a uniform scheme for configs across
            # models -- TODO: Figure out how to handle other models in this case
            text_num_heads = model_config.text_config.num_attention_heads
            vision_num_heads = model_config.num_attention_heads

        else:
            text_hidden_size = model_config.hidden_size
            text_num_heads = model_config.num_attention_heads

        text_Q = get_rotation_matrix(text_hidden_size, self.rotate_mode, "cpu")
        vision_Q = (
            get_rotation_matrix(vision_hidden_size, self.rotate_mode, "cpu")
            if vision_hidden_size is not None
            else None
        )

        interface.rotation_matrices.text_R1 = text_Q
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
            #     handle_qwen_patch_embed()

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
                logger.info(f"Forward Rotation on `{name}`")
                OfflineRotation.rotate_forward(module.weight, Q)  # type: ignore
                continue

            split = ".".join(name.rsplit(".", 2)[-2:])

            # we only rotate nn.Linear and nn.Embeddings layers
            if not isinstance(module, torch.nn.Linear):
                continue

            if split in rotate_forward_layers:
                logger.info(f"Forward Rotation on `{name}`")
                OfflineRotation.rotate_forward(module.weight, Q)  # type: ignore

            elif split in rotate_inverse_layers:
                logger.info(f"Inverse Rotation on `{name}`")
                if "merger" in name:
                    print(f"{text_Q.shape = }")
                    print(f"{module.weight.shape = }")
                    OfflineRotation.rotate_inverse(module.weight, text_Q)  # type: ignore

                else:
                    OfflineRotation.rotate_inverse(module.weight, Q)  # type: ignore

                if module.bias is not None:
                    logger.info(f"Rotating bias of `{name}`")
                    dtype = module.bias.dtype
                    if "merger" in name:
                        module.bias.data = (module.bias.double() @ text_Q).to(
                            dtype=dtype
                        )
                    else:
                        module.bias.data = (module.bias.double() @ Q).to(dtype=dtype)

            if self.online:
                if (
                    model.meta.vision_model_name is not None
                    and model.meta.vision_model_name in name
                ):
                    continue
                self.register_online_hooks(
                    name, module, model, hidden_size, num_attention_heads
                )

        model.cpu()

        interface.model = model
        return interface
