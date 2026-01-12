from transformers import LlamaForCausalLM

from ..utils import ModelType
from .meta import ArchitectureMeta

llama_meta = ArchitectureMeta(
    architecture_class=LlamaForCausalLM,
    model_type=ModelType.LLM,
    base_modules=["model.embed_tokens, model.norm"],
    layers="model.layers",
    embedding="model.embed_tokens",
    lm_head="lm_head",
    rope="model.rotary_emb",
    model_norm="model.norm",
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
            "model.embed_tokens",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
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
        "model.layers.#": {
            "post_attention_layernorm": ["mlp.up_proj", "mlp.gate_proj"],
            "input_layernorm": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ],
        },
        "model.norm": ["lm_head"],
    },
)
