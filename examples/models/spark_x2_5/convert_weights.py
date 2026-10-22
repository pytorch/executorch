import argparse
import json
import os
from typing import Dict

import torch

from executorch.examples.models.checkpoint import get_mapped_key
from safetensors.torch import load_file

_SPARK_X2_5_TO_META = {
    "model.embedding.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "model.layers.{}.self_attn.out_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.g_proj.weight": "layers.{}.attention.og.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
}


def spark_x2_5_to_meta(
    state_dict: Dict[str, torch.Tensor],
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> Dict[str, torch.Tensor]:
    """Convert Spark-X2.5 HF state dict to Meta format.

    Splits the fused q_k_v_proj into separate wq/wk/wv projections and handles
    tied word embeddings (no lm_head → output.weight = tok_embeddings.weight).
    """
    converted: Dict[str, torch.Tensor] = {}
    q_size = n_heads * head_dim
    kv_size = n_kv_heads * head_dim

    for key, value in state_dict.items():
        # Split fused QKV projection.
        if key.endswith(".self_attn.q_k_v_proj.weight"):
            layer_idx = key.split(".")[2]
            q, k, v = torch.split(value, [q_size, kv_size, kv_size], dim=0)
            converted[f"layers.{layer_idx}.attention.wq.weight"] = q
            converted[f"layers.{layer_idx}.attention.wk.weight"] = k
            converted[f"layers.{layer_idx}.attention.wv.weight"] = v
            continue

        try:
            new_key = get_mapped_key(key, _SPARK_X2_5_TO_META)
        except Exception:
            new_key = key.removeprefix("model.")

        converted[new_key] = value

    # Tied embeddings: no lm_head in Spark-X2.5.
    if "lm_head.weight" not in state_dict:
        converted["output.weight"] = converted["tok_embeddings.weight"]

    return converted


def load_checkpoint(input_dir: str) -> Dict[str, torch.Tensor]:
    """Load a safetensors checkpoint, supporting both single-file and sharded formats."""
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        # Sharded checkpoint.
        print("Loading checkpoint from sharded safetensors")
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        checkpoint_shards = sorted(set(weight_map.values()))

        shard_to_keys: Dict[str, list] = {}
        for weight_name, shard in weight_map.items():
            shard_to_keys.setdefault(shard, []).append(weight_name)

        merged: Dict[str, torch.Tensor] = {}
        for shard in checkpoint_shards:
            shard_data = load_file(os.path.join(input_dir, shard))
            for weight_name in shard_to_keys[shard]:
                merged[weight_name] = shard_data[weight_name]
            del shard_data
        return merged

    # Single checkpoint.
    model_path = os.path.join(input_dir, "model.safetensors")
    if os.path.exists(model_path):
        print("Loading checkpoint from safetensors directory")
        return load_file(model_path)

    raise FileNotFoundError(f"Could not find safetensors checkpoint in {input_dir}")


def convert_weights(input_dir: str, output_file: str) -> None:
    """Convert Spark-X2.5 HF weights to Meta format.

    Reads model hyperparameters from config.json in input_dir so the function
    signature stays compatible with download_and_convert_hf_checkpoint.
    """
    config_path = os.path.join(input_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        n_heads = config.get("num_attention_heads", 8)
        n_kv_heads = config.get("num_key_value_heads", 2)
        head_dim = config.get("head_dim", 256)
    else:
        n_heads, n_kv_heads, head_dim = 8, 2, 256

    print("Loading checkpoint...")
    sd = load_checkpoint(input_dir)
    print("Converting checkpoint...")
    sd = spark_x2_5_to_meta(sd, n_heads, n_kv_heads, head_dim)
    print("Saving checkpoint...")
    torch.save(sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Spark-X2.5 weights to Meta format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing safetensor checkpoint files.",
    )
    parser.add_argument("output", type=str, help="Path to the output checkpoint")

    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
