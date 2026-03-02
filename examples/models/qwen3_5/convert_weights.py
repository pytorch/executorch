import argparse
import json
import os
import re
from typing import Dict

import torch
from executorch.examples.models.checkpoint import (
    get_mapped_key,
    load_checkpoint_from_pytorch_model,
)

_QWEN_3_5_TO_META = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    # Full-attention layers.
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm_fn.weight",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm_fn.weight",
    # Linear-attention layers.
    "model.layers.{}.linear_attn.in_proj_qkv.weight": "layers.{}.attention.in_proj_qkv.weight",
    "model.layers.{}.linear_attn.in_proj_z.weight": "layers.{}.attention.in_proj_z.weight",
    "model.layers.{}.linear_attn.in_proj_b.weight": "layers.{}.attention.in_proj_b.weight",
    "model.layers.{}.linear_attn.in_proj_a.weight": "layers.{}.attention.in_proj_a.weight",
    "model.layers.{}.linear_attn.conv1d.weight": "layers.{}.attention.conv1d.weight",
    "model.layers.{}.linear_attn.conv1d.bias": "layers.{}.attention.conv1d.bias",
    "model.layers.{}.linear_attn.dt_bias": "layers.{}.attention.dt_bias",
    "model.layers.{}.linear_attn.A_log": "layers.{}.attention.A_log",
    "model.layers.{}.linear_attn.norm.weight": "layers.{}.attention.norm.weight",
    "model.layers.{}.linear_attn.out_proj.weight": "layers.{}.attention.out_proj.weight",
}


def _load_checkpoint_from_safetensors(input_dir: str) -> Dict:
    from safetensors.torch import load_file

    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        checkpoint_shards = sorted(set(weight_map.values()))

        shard_to_weights = {}
        for shard in checkpoint_shards:
            shard_to_weights[shard] = load_file(os.path.join(input_dir, shard))

        merged_state_dict = {}
        for weight_name, shard in weight_map.items():
            merged_state_dict[weight_name] = shard_to_weights[shard][weight_name]
        return merged_state_dict

    model_path = os.path.join(input_dir, "model.safetensors")
    if os.path.exists(model_path):
        return load_file(model_path)

    raise FileNotFoundError(f"Could not find safetensors checkpoint in {input_dir}")


def load_checkpoint(input_dir: str) -> Dict:
    try:
        print("Loading checkpoint from pytorch_model directory")
        return load_checkpoint_from_pytorch_model(input_dir)
    except FileNotFoundError:
        print(
            "Could not find pytorch_model checkpoints in directory, trying safetensors"
        )

    try:
        print("Loading checkpoint from safetensors directory")
        return _load_checkpoint_from_safetensors(input_dir)
    except FileNotFoundError:
        pass

    raise FileNotFoundError(f"Could not find checkpoint in {input_dir}")


def qwen_3_5_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted_state_dict = {}
    pending_qkvz = {}
    pending_ba = {}

    for key, value in state_dict.items():
        normalized_key = key
        # HF multimodal Qwen3.5 checkpoints store text weights under
        # `model.language_model.*`. Normalize to `model.*`.
        if normalized_key.startswith("model.language_model."):
            normalized_key = normalized_key.replace(
                "model.language_model.", "model.", 1
            )

        # Legacy packed tensors (older checkpoints):
        #   in_proj_qkvz -> split into in_proj_qkv and in_proj_z
        #   in_proj_ba   -> split into in_proj_b and in_proj_a
        if normalized_key.endswith(".linear_attn.in_proj_qkvz.weight"):
            pending_qkvz[normalized_key] = value
            continue
        if normalized_key.endswith(".linear_attn.in_proj_ba.weight"):
            pending_ba[normalized_key] = value
            continue

        try:
            new_key = get_mapped_key(normalized_key, _QWEN_3_5_TO_META)
        except Exception:
            # Ignore non-text weights and training-only extras (e.g., MTP).
            if (
                key.startswith("mtp.")
                or key.startswith("model.visual.")
                or ".vision_" in key
                or key.startswith("visual.")
            ):
                continue
            # Ignore unsupported keys that are not required by the export model.
            continue
        converted_state_dict[new_key] = value

    for key, value in pending_qkvz.items():
        layer_match = re.search(r"model\.layers\.(\d+)\.", key)
        if layer_match is None:
            raise ValueError(f"Failed to parse layer id from key: {key}")
        layer_id = layer_match.group(1)
        out_proj_key = f"layers.{layer_id}.attention.out_proj.weight"
        if out_proj_key not in converted_state_dict:
            raise ValueError(
                f"Cannot split {key}: missing {out_proj_key} to infer value dimension."
            )

        value_dim = converted_state_dict[out_proj_key].shape[1]
        total_dim = value.shape[0]
        conv_dim = total_dim - value_dim
        if conv_dim <= 0 or (conv_dim - value_dim) % 2 != 0:
            raise ValueError(
                f"Invalid packed in_proj_qkvz shape for {key}: {tuple(value.shape)}"
            )
        key_dim = (conv_dim - value_dim) // 2

        qkv, z = torch.split(value, [conv_dim, value_dim], dim=0)
        converted_state_dict[f"layers.{layer_id}.attention.in_proj_qkv.weight"] = qkv
        converted_state_dict[f"layers.{layer_id}.attention.in_proj_z.weight"] = z
        print(f"Split legacy packed key {key} -> in_proj_qkv + in_proj_z")

    for key, value in pending_ba.items():
        layer_match = re.search(r"model\.layers\.(\d+)\.", key)
        if layer_match is None:
            raise ValueError(f"Failed to parse layer id from key: {key}")
        layer_id = layer_match.group(1)
        if value.shape[0] % 2 != 0:
            raise ValueError(
                f"Invalid packed in_proj_ba shape for {key}: {tuple(value.shape)}"
            )
        half = value.shape[0] // 2
        b, a = torch.split(value, [half, half], dim=0)
        converted_state_dict[f"layers.{layer_id}.attention.in_proj_b.weight"] = b
        converted_state_dict[f"layers.{layer_id}.attention.in_proj_a.weight"] = a
        print(f"Split legacy packed key {key} -> in_proj_b + in_proj_a")

    # Handle tied embeddings.
    if "lm_head.weight" not in state_dict:
        converted_state_dict["output.weight"] = converted_state_dict[
            "tok_embeddings.weight"
        ]

    return converted_state_dict


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading checkpoint...")
    state_dict = load_checkpoint(input_dir)
    print("Converting checkpoint...")
    state_dict = qwen_3_5_to_meta(state_dict)
    print("Saving checkpoint...")
    torch.save(state_dict, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.5 weights to ExecuTorch meta format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing safetensor or PyTorch checkpoint files.",
    )
    parser.add_argument("output", type=str, help="Path to the output checkpoint")

    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
