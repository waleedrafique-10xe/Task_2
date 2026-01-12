from transformers import Qwen2_5_VLForConditionalGeneration

from ..utils import ModelType
from .meta import ArchitectureMeta

qwen2_5_vl_meta = ArchitectureMeta(
    architecture_class=Qwen2_5_VLForConditionalGeneration,
    model_type=ModelType.VLM,
    base_modules=["model.embed_tokens, model.norm"],  # TODO: fill these
    layers="model.language_model.layers",
    embedding="model.language_model.embed_tokens",
    lm_head="lm_head",
    rope="model.language_model.rotary_emb",
    model_norm="model.language_model.norm",
    layers_modules=[
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        [
            "mlp.gate_proj",
            "mlp.up_proj",
        ],
        ["mlp.down_proj"],
    ],
    offline_rotations={
        "rotate_forward": [
            "model.language_model.embed_tokens",
            "model.visual.patch_embed",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "lm_head",
            "attn.qkv",
            "mlp.0",
        ],
        "rotate_inverse": [
            "self_attn.o_proj",
            "mlp.down_proj",
            "attn.proj",
            "mlp.2",
        ],
    },
    online_rotations={
        "rotate_v": ["self_attn.v_proj"],
        "rotate_output": ["self_attn.o_proj", "mlp.down_proj"],
    },
    layernorm_fuses={
        "model.language_model.layers.#": {
            "post_attention_layernorm": ["mlp.up_proj", "mlp.gate_proj"],
            "input_layernorm": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ],
        },
        "model.visual.blocks.#": {
            "norm1": ["attn.qkv"],
            "norm2": ["mlp.gate_proj", "mlp.up_proj"],
        },
        "model.visual.merger.ln_q": ["model.visual.merger.mlp.0"],
        "model.language_model.norm": ["lm_head"],
    },
    vision_layers="model.visual.blocks",
    vision_layers_modules=[
        ["attn.qkv"],
        ["attn.proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ],
    language_model_name="language_model",
    vision_model_name="visual",
)
