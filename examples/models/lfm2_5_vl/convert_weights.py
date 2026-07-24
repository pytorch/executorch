# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert LFM2.5-VL text decoder weights from HuggingFace to ET format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from executorch.examples.models.checkpoint import get_mapped_key
from safetensors.torch import load_file

_LFM2_5_VL_TO_META: dict[str, str] = {
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
    "model.language_model.lm_head.weight": "output.weight",
}

_IN_PROJ_SPLITS = ("B_proj", "C_proj", "x_proj")


def lfm2_5_vl_to_meta(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract and remap language model weights from a full VL state dict."""
    converted: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if not key.startswith("model.language_model."):
            continue

        try:
            new_key = get_mapped_key(key, _LFM2_5_VL_TO_META)
        except Exception:
            new_key = key.removeprefix("model.language_model.")

        if new_key.endswith(".conv.in_proj.weight"):
            for name, chunk in zip(_IN_PROJ_SPLITS, torch.chunk(value, 3, dim=0)):
                converted[new_key.replace("in_proj", name)] = chunk
        else:
            converted[new_key] = value

    if "output.weight" not in converted:
        converted["output.weight"] = converted["tok_embeddings.weight"]

    return converted


def convert_weights(input_dir: str, output_file: str) -> None:
    sd = load_file(str(Path(input_dir) / "model.safetensors"))
    sd = lfm2_5_vl_to_meta(sd)
    torch.save(sd, output_file)
    print(f"Saved {len(sd)} tensors to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LFM2.5-VL weights to ET format.")
    parser.add_argument("input_dir", help="Directory containing model.safetensors.")
    parser.add_argument("output", help="Output .pt checkpoint path.")
    args = parser.parse_args()
    convert_weights(args.input_dir, args.output)


if __name__ == "__main__":
    main()
