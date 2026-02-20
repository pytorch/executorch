from typing import Dict

import torch

from safetensors.torch import load_file
from torchtune.models.convert_weights import get_mapped_key

_UNSLOTH_TO_META = {
    "base_model.model.model.layers.{}.mlp.down_proj.lora_A.weight": "layers.{}.feed_forward.w2.lora_a.weight",
    "base_model.model.model.layers.{}.mlp.down_proj.lora_B.weight": "layers.{}.feed_forward.w2.lora_b.weight",
    "base_model.model.model.layers.{}.mlp.gate_proj.lora_A.weight": "layers.{}.feed_forward.w1.lora_a.weight",
    "base_model.model.model.layers.{}.mlp.gate_proj.lora_B.weight": "layers.{}.feed_forward.w1.lora_b.weight",
    "base_model.model.model.layers.{}.mlp.up_proj.lora_A.weight": "layers.{}.feed_forward.w3.lora_a.weight",
    "base_model.model.model.layers.{}.mlp.up_proj.lora_B.weight": "layers.{}.feed_forward.w3.lora_b.weight",
    "base_model.model.model.layers.{}.self_attn.k_proj.lora_A.weight": "layers.{}.attention.wk.lora_a.weight",
    "base_model.model.model.layers.{}.self_attn.k_proj.lora_B.weight": "layers.{}.attention.wk.lora_b.weight",
    "base_model.model.model.layers.{}.self_attn.o_proj.lora_A.weight": "layers.{}.attention.wo.lora_a.weight",
    "base_model.model.model.layers.{}.self_attn.o_proj.lora_B.weight": "layers.{}.attention.wo.lora_b.weight",
    "base_model.model.model.layers.{}.self_attn.q_proj.lora_A.weight": "layers.{}.attention.wq.lora_a.weight",
    "base_model.model.model.layers.{}.self_attn.q_proj.lora_B.weight": "layers.{}.attention.wq.lora_b.weight",
    "base_model.model.model.layers.{}.self_attn.v_proj.lora_A.weight": "layers.{}.attention.wv.lora_a.weight",
    "base_model.model.model.layers.{}.self_attn.v_proj.lora_B.weight": "layers.{}.attention.wv.lora_b.weight",
}


def unsloth_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from unsloth format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in unsloth format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}

    for key, value in state_dict.items():
        try:
            new_key = get_mapped_key(key, _UNSLOTH_TO_META)
        except Exception as e:
            raise ValueError(f"Key {key} not found in mapping") from e

        converted_state_dict[new_key] = value
    return converted_state_dict


def load_and_convert_unsloth_to_meta(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a checkpoint file and convert it to Meta's format.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    state_dict = load_file(checkpoint_path)
    return unsloth_to_meta(state_dict)
