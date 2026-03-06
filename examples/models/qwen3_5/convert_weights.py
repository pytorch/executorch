import argparse
import json
import os
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
    "model.layers.{}.linear_attn.dt_bias": "layers.{}.attention.dt_bias",
    "model.layers.{}.linear_attn.A_log": "layers.{}.attention.A_log",
    "model.layers.{}.linear_attn.norm.weight": "layers.{}.attention.norm.weight",
    "model.layers.{}.linear_attn.out_proj.weight": "layers.{}.attention.out_proj.weight",
}


_IGNORED_UNMAPPED_SUFFIXES = (
    "rotary_emb.inv_freq",
    "linear_attn.conv1d.bias",
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

        # Load each shard once and copy only the tensor names mapped to that shard.
        # This avoids holding all shard tensors in memory at the same time.
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


def qwen_3_5_to_meta(  # noqa: C901
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    converted_state_dict = {}

    for key, value in state_dict.items():
        normalized_key = key
        # HF multimodal Qwen3.5 checkpoints store text weights under
        # `model.language_model.*`. Normalize to `model.*`.
        if normalized_key.startswith("model.language_model."):
            normalized_key = normalized_key.replace(
                "model.language_model.", "model.", 1
            )

        # Ignore non-text-model keys up front.
        if not normalized_key.startswith(
            (
                "model.layers.",
                "model.embed_tokens.",
                "model.norm.",
                "lm_head.",
            )
        ):
            continue

        try:
            new_key = get_mapped_key(normalized_key, _QWEN_3_5_TO_META)
        except Exception as err:
            if normalized_key.endswith(_IGNORED_UNMAPPED_SUFFIXES):
                continue
            raise ValueError(
                f"Unexpected checkpoint key not mapped for Qwen3.5 export: {key}"
            ) from err
        converted_state_dict[new_key] = value

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
