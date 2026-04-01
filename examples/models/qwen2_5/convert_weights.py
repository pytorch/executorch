import argparse
import json
import os
from typing import Dict

import torch
from safetensors.torch import load_file
from torchtune.models.convert_weights import get_mapped_key

# Weight mapping from Meta's llama_transformer format to HuggingFace Qwen2 format.
_QWEN_2_5_FROM_META = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.wq.bias": "model.layers.{}.self_attn.q_proj.bias",
    "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.wk.bias": "model.layers.{}.self_attn.k_proj.bias",
    "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wv.bias": "model.layers.{}.self_attn.v_proj.bias",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
    "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
    "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
    "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
}


def qwen_2_5_hf_to_meta(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _QWEN_2_5_FROM_META.items()}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    # Models with tied embeddings (e.g. 0.5B, 1.5B) don't have a separate lm_head.weight.
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

        # Group weights by shard so we load each shard file only once.
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

    # Single checkpoint.
    model_path = os.path.join(input_dir, "model.safetensors")
    if os.path.exists(model_path):
        return load_file(model_path)

    raise FileNotFoundError(f"Could not find safetensors checkpoint in {input_dir}")


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading checkpoint...")
    sd = load_checkpoint_from_safetensors(input_dir)
    print("Converting checkpoint...")
    sd = qwen_2_5_hf_to_meta(sd)
    print("Saving checkpoint...")
    torch.save(sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen2.5 weights to Meta format."
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
