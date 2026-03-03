# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert LFM2.5-VL text decoder weights from HuggingFace safetensors format
to the Meta/ET format expected by construct_transformer.

Usage:
    python examples/models/lfm2_5_vl/convert_weights.py \
        /path/to/LFM2-VL-1.6B \
        lfm2_5_vl_1_6b.pt

The input directory must contain model.safetensors from the HuggingFace
LiquidAI/LFM2-VL-1.6B checkpoint. Only the language model (text decoder)
weights are extracted — vision tower and projector weights are not included.
"""

import argparse
import os
from typing import Dict

import torch

from executorch.examples.models.checkpoint import get_mapped_key
from safetensors.torch import load_file

# HuggingFace key -> ET/Meta key mapping for LFM2.5-VL language model.
# Keys use {} as a placeholder for the layer index (handled by get_mapped_key).
# The "model.language_model." prefix is specific to the VL wrapper; text-only
# LFM2 uses "model." directly.
_LFM2_5_VL_TO_META = {
    "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
    "model.language_model.embedding_norm.weight": "norm.weight",
    "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.language_model.layers.{}.self_attn.out_proj.weight": "layers.{}.attention.wo.weight",
    "model.language_model.layers.{}.self_attn.q_layernorm.weight": "layers.{}.attention.q_norm_fn.weight",
    "model.language_model.layers.{}.self_attn.k_layernorm.weight": "layers.{}.attention.k_norm_fn.weight",
    "model.language_model.layers.{}.operator_norm.weight": "layers.{}.attention_norm.weight",
    "model.language_model.layers.{}.ffn_norm.weight": "layers.{}.ffn_norm.weight",
    "model.language_model.layers.{}.feed_forward.w1.weight": "layers.{}.feed_forward.w1.weight",
    "model.language_model.layers.{}.feed_forward.w2.weight": "layers.{}.feed_forward.w2.weight",
    "model.language_model.layers.{}.feed_forward.w3.weight": "layers.{}.feed_forward.w3.weight",
    "model.language_model.layers.{}.conv.conv.weight": "layers.{}.conv.conv.weight",
    "model.language_model.layers.{}.conv.out_proj.weight": "layers.{}.conv.out_proj.weight",
}


def lfm2_5_vl_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from LFM2.5-VL HF format to Meta's ET format.
    Only extracts language model (text decoder) weights.

    Args:
        state_dict: State dict from model.safetensors (full VL model).

    Returns:
        State dict in ET construct_transformer format.
    """
    converted = {}

    for key, value in state_dict.items():
        # Skip vision tower and projector — not part of the text decoder
        if not key.startswith("model.language_model."):
            continue

        try:
            new_key = get_mapped_key(key, _LFM2_5_VL_TO_META)
        except KeyError:
            # Fallback: strip "model.language_model." prefix
            new_key = key.removeprefix("model.language_model.")

        # Split conv in_proj: [3*dim, dim] -> B_proj, C_proj, x_proj each [dim, dim]
        if new_key.endswith(".conv.in_proj.weight"):
            for name, split_value in zip(
                ["B_proj", "C_proj", "x_proj"], torch.chunk(value, 3, dim=0)
            ):
                converted[new_key.replace("in_proj", name)] = split_value
        else:
            converted[new_key] = value

    # lm_head is not in safetensors (tied embeddings) — use tok_embeddings
    if "output.weight" not in converted:
        converted["output.weight"] = converted["tok_embeddings.weight"]

    return converted


def load_checkpoint(input_dir: str) -> Dict:
    print("Loading checkpoint from safetensors...")
    return load_file(os.path.join(input_dir, "model.safetensors"))


def convert_weights(input_dir: str, output_file: str) -> None:
    print("Loading checkpoint...")
    sd = load_checkpoint(input_dir)
    print("Converting weights...")
    sd = lfm2_5_vl_to_meta(sd)
    print("Saving checkpoint...")
    torch.save(sd, output_file)
    print(f"Saved {len(sd)} tensors to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LFM2.5-VL text decoder weights to Meta/ET format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing model.safetensors (HuggingFace LFM2-VL-1.6B).",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to the output .pt checkpoint file.",
    )
    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
