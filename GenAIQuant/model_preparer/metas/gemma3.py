from transformers import Gemma3ForConditionalGeneration

from ..utils import ModelType
from .meta import ArchitectureMeta

gemma3_meta = ArchitectureMeta(
    architecture_class=Gemma3ForConditionalGeneration,
    model_type=ModelType.VLM,
    base_modules=["model.embed_tokens, model.norm"],
    layers="model.language_model.layers",
    embedding="model.language_model.embed_tokens",
    lm_head="lm_head",
    rope="model.language_model.rotary_emb",
    model_norm="model.language_model.norm",
    layers_modules=[
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
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
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "lm_head",
        ],
        "rotate_inverse": ["self_attn.o_proj", "mlp.down_proj"],
    },
    online_rotations={
        "rotate_v": ["self_attn.v_proj"],
        "rotate_output": ["self_attn.o_proj", "mlp.down_proj"],
    },
    layernorm_fuses={
        "model.language_model.layers.#": {
            "pre_feedforward_layernorm": ["mlp.up_proj", "mlp.gate_proj"],
            "post_feedforward_layernorm": ["mlp.down_proj"],
            "input_layernorm": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ],
            "post_attention_layernorm": ["self_attn.o_proj"],
        },
        "model.vision_tower.vision_model.encoder.layers.#": {
            "layer_norm2": ["mlp.fc1"],
            "layer_norm1": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ],
        },
        "model.language_model.norm": ["lm_head"],
    },
    vision_layers="model.vision_tower.vision_model.encoder.layers",
    vision_layers_modules=[
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.out_proj"],
        [
            "mlp.fc2",
        ],
        ["mlp.fc2"],
    ],
    vision_model_name="vision_tower.vision_model",
)
