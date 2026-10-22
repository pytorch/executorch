# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantize Muse Glimmer into an unfused, backend-independent checkpoint.

Weights are streamed and quantized individually; backend projection fusion
happens when the checkpoint is loaded.
"""

import argparse
import gc
import os
import shutil

import torch
from executorch.examples.models.muse_glimmer.model.model import (
    load_unfused_bf16_state_dict,
)
from executorch.extension.llm.export.quant import (
    QuantConfig,
    quantize_stream,
    QuantRecipe,
    QuantRule,
)

# Quantization recipes keep norms unquantized and use wider weights for the
# sensitive projections. Atomic weights allow per-projection bit widths.

_INT4 = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
_INT4_HQQ = QuantConfig(bits=4, group_size=32, symmetric=False, method="hqq")
_INT8 = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
_INT8_PER_AXIS = QuantConfig(bits=8, group_size=6656, symmetric=True, method="min_max")
_INT6 = QuantConfig(bits=6, group_size=32, symmetric=True, method="min_max")
_INT8_GS128 = QuantConfig(bits=8, group_size=128, symmetric=True, method="min_max")
_INT4_GS64 = QuantConfig(bits=4, group_size=64, symmetric=False, method="min_max")
_INT4_GS128 = QuantConfig(bits=4, group_size=128, symmetric=False, method="min_max")
_INT6_GS64 = QuantConfig(bits=6, group_size=64, symmetric=True, method="min_max")
_INT6_GS128 = QuantConfig(bits=6, group_size=128, symmetric=True, method="min_max")
_EDGE_LAYERS = set(range(13)) | set(range(39, 52))

RECIPE_NAMES = (
    "default",
    "sensitive",
    "int4",
    "gguf",
    "gguf_int8",
    "gguf_int6_gs64",
    "gguf_int6_gs128",
    "gguf_int6only_gs128",
)


def build_recipes(use_int6: bool = False) -> dict[str, QuantRecipe]:
    """Build the Muse Glimmer quant recipes.

    When ``use_int6`` is set, the weights that are otherwise int8 (the
    embedding and the edge-layer o_proj/down_proj) are quantized to 6-bit at
    group_size 32 instead. MLX backend only. ``use_int6`` does not affect the
    'gguf' recipe, which specifies its int6 layers explicitly.
    """
    embed_cfg = _INT6 if use_int6 else _INT8_PER_AXIS
    wide_cfg = _INT6 if use_int6 else _INT8

    default = QuantRecipe(
        rules=[
            QuantRule(r"embed_tokens\.weight", embed_cfg),
            QuantRule(r".*norm\.weight", None),
            QuantRule(r".*\.weight", _INT4),
        ]
    )
    sensitive = QuantRecipe(
        rules=[
            QuantRule(r"embed_tokens\.weight", embed_cfg),
            QuantRule(r".*norm\.weight", None),
            QuantRule(r".*\.(o_proj|down_proj)\.weight", wide_cfg, layers=_EDGE_LAYERS),
            QuantRule(r".*\.weight", _INT4_HQQ),
        ]
    )

    # Mixed recipes keep v_proj, down_proj, and lm_head at higher precision.
    def _gguf_style(
        wide_cfg: QuantConfig, narrow_cfg: QuantConfig = _INT4
    ) -> QuantRecipe:
        return QuantRecipe(
            rules=[
                QuantRule(r"embed_tokens\.weight", narrow_cfg),
                QuantRule(r".*norm\.weight", None),
                QuantRule(r".*\.v_proj\.weight", wide_cfg),
                QuantRule(r".*\.down_proj\.weight", wide_cfg),
                QuantRule(r"lm_head\.weight", wide_cfg),
                QuantRule(r".*\.weight", narrow_cfg),
            ]
        )

    # All quantizable weights at int4 (including the embedding); norms
    # unquantized. MLX-only (int4 embedding is not supported by the CUDA
    # embedding packer).
    int4 = QuantRecipe(
        rules=[
            QuantRule(r".*norm\.weight", None),
            QuantRule(r".*\.weight", _INT4),
        ]
    )

    return {
        "default": default,
        "sensitive": sensitive,
        "int4": int4,
        "gguf": _gguf_style(_INT6),
        "gguf_int8": _gguf_style(_INT8_GS128),
        "gguf_int6_gs64": _gguf_style(_INT6_GS64, narrow_cfg=_INT4_GS64),
        "gguf_int6_gs128": _gguf_style(_INT6_GS128, narrow_cfg=_INT4_GS128),
        # int6 wide layers at gs128, int4 narrow layers stay at gs32.
        "gguf_int6only_gs128": _gguf_style(_INT6_GS128),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize Muse Glimmer and save as an atomic quantized checkpoint."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to consolidated Muse Glimmer checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="./muse_glimmer_int4",
        help="Output directory.",
    )
    parser.add_argument(
        "--quant-recipe",
        default="default",
        choices=RECIPE_NAMES,
        help="'default': int4 min_max linears + int8 per-axis embedding. "
        "'sensitive': int8 for edge-layer o_proj/down_proj, int4 hqq elsewhere. "
        "'int4': int4 everywhere incl. embedding (MLX-only). "
        "'gguf': int6 v_proj/down_proj/lm_head, int4 elsewhere. "
        "'gguf_int8': same but int8 gs128 on those layers. "
        "'gguf_int6_gs128': like gguf but groupsize 128 for both int4 and int6.",
    )
    parser.add_argument(
        "--int6",
        action="store_true",
        help="Quantize the otherwise-int8 weights (embedding + edge-layer "
        "o_proj/down_proj) to 6-bit at group_size 32 instead. MLX backend only.",
    )
    args = parser.parse_args()

    recipe = build_recipes(args.int6)[args.quant_recipe]

    # Load the checkpoint as a fully-unfused (atomic) bf16 state dict. The full
    # model is never instantiated; the weights stay mmap-backed so peak memory is
    # the quantized output plus a small working set.
    print("Loading checkpoint (atomic, unfused)...")
    state_dict, _config = load_unfused_bf16_state_dict(args.checkpoint_dir)

    print(f"Quantizing (model-free) with recipe '{args.quant_recipe}'...")
    quantized: dict[str, torch.Tensor] = {}
    for fqn, value in quantize_stream(state_dict.items(), recipe, dtype=torch.bfloat16):
        quantized[fqn] = value
        print(f"  Quantized: {fqn}", end="\r")
    print()
    del state_dict
    gc.collect()

    os.makedirs(args.output, exist_ok=True)
    safetensors_path = os.path.join(args.output, "model.safetensors")
    print("Saving atomic quantized checkpoint...")
    from safetensors.torch import save_file
    from torchao.prototype.safetensors.safetensors_support import (
        flatten_tensor_state_dict,
    )

    tensors_data, metadata = flatten_tensor_state_dict(quantized)
    # flatten_tensor_state_dict clones every quantized tensor, so drop the
    # originals before writing.
    n_tensors = len(quantized)
    del quantized
    gc.collect()
    save_file(tensors_data, safetensors_path, metadata=metadata)

    # The checkpoint is atomic / backend-agnostic: no fusion layout is recorded
    # (fusion is decided at load time per backend). Copy params.json verbatim.
    params_src = os.path.join(args.checkpoint_dir, "params.json")
    if os.path.exists(params_src):
        shutil.copy2(params_src, os.path.join(args.output, "params.json"))

    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"Saved {n_tensors} tensors ({size_mb:.1f} MB) to {args.output}/")
    print(f"Done. Use with: python export.py --prequantized {args.output}")


if __name__ == "__main__":
    main()
