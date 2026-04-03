import argparse
import json
import os
from typing import Dict

import torch
from executorch.examples.models.checkpoint import (
    get_mapped_key,
    load_checkpoint_from_pytorch_model,
)


_GEMMA4_TO_EXECUTORCH = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.embed_tokens_per_layer.weight": "embed_tokens_per_layer.weight",
    "model.per_layer_model_projection.weight": "per_layer_model_projection.weight",
    "model.per_layer_projection_norm.weight": "per_layer_projection_norm.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm_fn.weight",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm_fn.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_norm.weight",
    "model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.post_ffn_norm.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.layer_scalar": "layers.{}.layer_scalar",
    "model.layers.{}.per_layer_input_gate.weight": "layers.{}.per_layer_input_gate.weight",
    "model.layers.{}.per_layer_projection.weight": "layers.{}.per_layer_projection.weight",
    "model.layers.{}.post_per_layer_input_norm.weight": "layers.{}.post_per_layer_input_norm.weight",
}


_IGNORED_UNMAPPED_SUFFIXES = (
    "rotary_emb.inv_freq",
    "self_attn.v_norm.weight",
)


def _load_checkpoint_from_safetensors(input_dir: str) -> Dict:
    from safetensors.torch import load_file

    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        checkpoint_shards = sorted(set(weight_map.values()))

        merged_state_dict = {}
        shard_to_weight_names = {}
        for weight_name, shard in weight_map.items():
            shard_to_weight_names.setdefault(shard, []).append(weight_name)

        for shard in checkpoint_shards:
            shard_weights = load_file(os.path.join(input_dir, shard))
            for weight_name in shard_to_weight_names[shard]:
                merged_state_dict[weight_name] = shard_weights[weight_name]
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


def gemma4_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted_state_dict = {}

    for key, value in state_dict.items():
        normalized_key = key
        if normalized_key.startswith("model.language_model."):
            normalized_key = normalized_key.replace("model.language_model.", "model.", 1)

        if not normalized_key.startswith(
            (
                "model.layers.",
                "model.embed_tokens.",
                "model.embed_tokens_per_layer.",
                "model.per_layer_model_projection.",
                "model.per_layer_projection_norm.",
                "model.norm.",
                "lm_head.",
            )
        ):
            continue

        try:
            new_key = get_mapped_key(normalized_key, _GEMMA4_TO_EXECUTORCH)
        except Exception as err:
            if normalized_key.endswith(_IGNORED_UNMAPPED_SUFFIXES):
                continue
            raise ValueError(
                f"Unexpected checkpoint key not mapped for Gemma4 export: {key}"
            ) from err
        converted_state_dict[new_key] = value

    if "output.weight" not in converted_state_dict:
        converted_state_dict["output.weight"] = converted_state_dict[
            "tok_embeddings.weight"
        ]

    return converted_state_dict


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading checkpoint...")
    state_dict = load_checkpoint(input_dir)
    print("Converting checkpoint...")
    state_dict = gemma4_to_meta(state_dict)
    print("Saving checkpoint...")
    torch.save(state_dict, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemma4 weights to ExecuTorch meta format."
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
