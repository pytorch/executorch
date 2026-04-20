import argparse
import json
import os
from typing import Dict

import torch

from executorch.examples.models.checkpoint import get_mapped_key
from safetensors.torch import load_file


_GEMMA4_TO_EXECUTORCH = {
    # Embeddings
    "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
    "model.language_model.norm.weight": "norm.weight",
    # Per-layer input (PLI) global weights
    "model.language_model.embed_tokens_per_layer.weight": "pli_embeddings.weight",
    "model.language_model.per_layer_model_projection.weight": "pli_projection.weight",
    "model.language_model.per_layer_projection_norm.weight": "pli_norm.weight",
    # Attention
    "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.language_model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm_fn.weight",
    "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.language_model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm_fn.weight",
    "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    # NOTE: Gemma4 v_norm uses with_scale=False (no learnable weight) — implemented
    # inline in attention.py forward(); no mapping entry needed here.
    # Layer norms (4 per layer: pre/post attention + pre/post FFN)
    "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_norm.weight",
    "model.language_model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.language_model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.post_ffn_norm.weight",
    # MLP
    "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    # Per-layer input (PLI) per-layer weights
    "model.language_model.layers.{}.per_layer_input_gate.weight": "layers.{}.per_layer_input_gate.weight",
    "model.language_model.layers.{}.per_layer_projection.weight": "layers.{}.per_layer_projection.weight",
    "model.language_model.layers.{}.post_per_layer_input_norm.weight": "layers.{}.post_per_layer_input_norm.weight",
    # Layer scalar
    "model.language_model.layers.{}.layer_scalar": "layers.{}.layer_scalar",
}


def gemma4_to_executorch(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    converted_state_dict = {}
    skipped = 0
    for key, value in state_dict.items():
        # Skip non-language-model weights (vision_tower, audio_tower,
        # multi_modal_projector, etc.) — those are handled separately
        # for multimodal export.
        if not key.startswith("model.language_model."):
            skipped += 1
            continue
        new_key = get_mapped_key(key, _GEMMA4_TO_EXECUTORCH)
        converted_state_dict[new_key] = value
    if skipped:
        print(f"  Skipped {skipped} non-language-model tensors (vision/audio)")
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
        shard_to_weights = {}
        for shard in checkpoint_shards:
            shard_to_weights[shard] = load_file(os.path.join(input_dir, shard))
        merged_state_dict = {}
        for weight_name, shard in weight_map.items():
            tensor = shard_to_weights[shard][weight_name]
            merged_state_dict[weight_name] = tensor
        return merged_state_dict
    else:
        state_dict = load_file(os.path.join(input_dir, "model.safetensors"))
        return state_dict


def load_checkpoint(input_dir: str) -> Dict:
    pytorch_path = os.path.join(input_dir, "pytorch_model.bin")
    if os.path.exists(pytorch_path):
        return torch.load(pytorch_path, map_location="cpu", weights_only=True)
    return load_checkpoint_from_safetensors(input_dir)


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading checkpoint...")
    sd = load_checkpoint(input_dir)
    print("Converting checkpoint...")
    sd = gemma4_to_executorch(sd)
    print("Saving checkpoint...")
    torch.save(sd, output_file)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemma4 weights to ExecuTorch transformer format."
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
