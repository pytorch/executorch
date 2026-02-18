import argparse
import json
import os
from typing import Dict

import torch
from executorch.examples.models.checkpoint import get_mapped_key
from safetensors.torch import load_file

_HF_PHI_4_FROM_META = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
    "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
    "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
    "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
    "output.weight": "lm_head.weight",
}


def phi_4_hf_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _HF_PHI_4_FROM_META.items()}

    for key, value in state_dict.items():
        if key.endswith("mlp.gate_up_proj.weight"):
            # Split the fused gate_up_proj into gate_proj and up_proj.
            hidden_dim = value.shape[0] // 2
            assert 2 * hidden_dim == value.shape[0]
            gate = value[0:hidden_dim, :]
            up = value[hidden_dim:, :]
            for new_key, new_value in [("gate_proj", gate), ("up_proj", up)]:
                new_key = key.replace("gate_up_proj", new_key)
                new_key = get_mapped_key(new_key, inverted_mapping_dict)
                converted_state_dict[new_key] = new_value
        elif key.endswith("self_attn.qkv_proj.weight"):
            # Split the fused qkv_proj into q_proj, k_proj, and v_proj.
            q_dim = value.shape[1]
            kv_dim = (value.shape[0] - q_dim) // 2
            assert 2 * kv_dim + q_dim == value.shape[0]
            q = value[0:q_dim, :]
            k = value[q_dim : (q_dim + kv_dim), :]
            v = value[(q_dim + kv_dim) :, :]
            for new_key, new_value in [("q_proj", q), ("k_proj", k), ("v_proj", v)]:
                new_key = key.replace("qkv_proj", new_key)
                new_key = get_mapped_key(new_key, inverted_mapping_dict)
                converted_state_dict[new_key] = new_value
        else:
            new_key = get_mapped_key(key, inverted_mapping_dict)
            converted_state_dict[new_key] = value

    if "lm_head.weight" not in state_dict:
        converted_state_dict["output.weight"] = converted_state_dict[
            "tok_embeddings.weight"
        ]

    return converted_state_dict


def load_checkpoint_from_safetensors(input_dir: str) -> Dict:
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        checkpoint_shards = sorted(set(weight_map.values()))

        shard_to_keys = {}
        for weight_name, shard in weight_map.items():
            shard_to_keys.setdefault(shard, []).append(weight_name)

        merged_state_dict = {}
        for shard in checkpoint_shards:
            shard_data = load_file(os.path.join(input_dir, shard))
            for weight_name in shard_to_keys[shard]:
                merged_state_dict[weight_name] = shard_data[weight_name]
            del shard_data
        return merged_state_dict

    model_path = os.path.join(input_dir, "model.safetensors")
    if os.path.exists(model_path):
        return load_file(model_path)

    raise FileNotFoundError(f"Could not find safetensors checkpoint in {input_dir}")


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading checkpoint...")
    sd = load_checkpoint_from_safetensors(input_dir)
    print("Converting checkpoint...")
    sd = phi_4_hf_to_meta(sd)
    print("Saving checkpoint...")
    torch.save(sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Phi-4-mini weights to Meta format."
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
