import argparse

import json
import os
from typing import Dict

import torch
from safetensors.torch import load_file

from torchtune.models.convert_weights import get_mapped_key

# Standard _FROM_META weight mapping of Meta weights to TorchTune + additional bias weight mappings.
_QWEN_3_FROM_META = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.k_norm_fn.weight": "model.layers.{}.self_attn.k_norm.weight",
    "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.q_norm_fn.weight": "model.layers.{}.self_attn.q_norm.weight",
    "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
    "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    # Note: gate_proj and up_proj are reversed, usually w1 is the up_proj,
    # w2 is the gate_proj, and activation is applied on the up_proj, but since
    # Qwen3 applies activation on the gate_proj, we just swap the gate_proj
    # and up_proj in the checkpoint itself as a hack.
    "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
    "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
    "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
}


def qwen_3_tune_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from torchtune's format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in torchtune's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _QWEN_3_FROM_META.items()}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    # If lm_head.weight is not present in state dict, assume tied embeddings (e.g., 0.6b and 4b models)
    if "lm_head.weight" not in state_dict:
        converted_state_dict["output.weight"] = converted_state_dict[
            "tok_embeddings.weight"
        ]

    return converted_state_dict


def load_checkpoint_from_safetensors(input_dir: str) -> Dict:
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        # Sharded checkpoint.
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        checkpoint_shards = sorted(set(weight_map.values()))

        # Load all the shards into memory
        shard_to_weights = {}
        for shard in checkpoint_shards:
            shard_to_weights[shard] = load_file(os.path.join(input_dir, shard))

        # Merge tensors into consolidated state dict.
        merged_state_dict = {}
        for weight_name, shard in weight_map.items():
            tensor = shard_to_weights[shard][weight_name]
            merged_state_dict[weight_name] = tensor
        return merged_state_dict
    else:
        # Single checkpoint.
        state_dict = load_file(os.path.join(input_dir, "model.safetensors"))
        return state_dict


def load_checkpoint(input_dir: str) -> Dict:
    pytorch_path = os.path.join(input_dir, "pytorch_model.bin")
    if os.path.exists(pytorch_path):
        print("Loading checkpoint from PyTorch .bin file")
        return torch.load(pytorch_path, map_location="cpu", weights_only=True)
    print("Loading checkpoint from safetensors directory")
    return load_checkpoint_from_safetensors(input_dir)


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading checkpoint...")
    sd = load_checkpoint(input_dir)
    print("Converting checkpoint...")
    sd = qwen_3_tune_to_meta(sd)
    print("Saving checkpoint...")
    torch.save(sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3 weights to Meta format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing safetensor checkpoint files, or PyTorch checkpoint file.",
    )
    parser.add_argument("output", type=str, help="Path to the output checkpoint")

    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
