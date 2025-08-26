import argparse

import json
import os
from typing import Dict

import torch
from safetensors.torch import load_file

from torchtune.models.convert_weights import get_mapped_key


# Weight mappings from Gemma's checkpoint to ExecuTorch's transformer parameters.
_GEMMA_TO_EXECUTORCH = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
}


def gemma_to_executorch(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convert the state dict so that it matches what ExecuTorch's transformer definition expects.
    """
    converted_state_dict = {}
    for key, value in state_dict.items():
        new_key = get_mapped_key(key, _GEMMA_TO_EXECUTORCH)
        converted_state_dict[new_key] = value
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
    sd = gemma_to_executorch(sd)
    print("Saving checkpoint...")
    torch.save(sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemma weights to ExecuTorch transformer format."
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
