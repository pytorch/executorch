import argparse

import json
import os
from typing import Dict

import torch
from safetensors.torch import load_file

from torchtune.models.convert_weights import get_mapped_key

_LFM_2_TO_META = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.embedding_norm.weight": "norm.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.out_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.k_layernorm.weight": "layers.{}.attention.k_norm_fn.weight",
    "model.layers.{}.self_attn.q_layernorm.weight": "layers.{}.attention.q_norm_fn.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.layers.{}.operator_norm.weight": "layers.{}.attention_norm.weight",
}


def lfm_2_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from LFM2 HF format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in LFM2 HF format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}

    for key, value in state_dict.items():
        try:
            new_key = get_mapped_key(key, _LFM_2_TO_META)
        except:
            new_key = key.removeprefix("model.")

        # split in_proj
        if new_key.endswith(".conv.in_proj.weight"):
            for name, split_value in zip(
                ["B_proj", "C_proj", "x_proj"], torch.chunk(value, 3, dim=0)
            ):
                converted_state_dict[new_key.replace("in_proj", name)] = split_value
        else:
            converted_state_dict[new_key] = value

    # If lm_head.weight is not present in state dict, assume tied embeddings
    if "lm_head.weight" not in state_dict:
        converted_state_dict["output.weight"] = converted_state_dict[
            "tok_embeddings.weight"
        ]

    return converted_state_dict


def load_checkpoint(input_dir: str) -> Dict:
    print("Loading checkpoint from safetensors directory")
    state_dict = load_file(os.path.join(input_dir, "model.safetensors"))
    return state_dict


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading checkpoint...")
    sd = load_checkpoint(input_dir)
    print("Converting checkpoint...")
    sd = lfm_2_to_meta(sd)
    print("Saving checkpoint...")
    torch.save(sd, output_file)
    print("Done.")
