# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantize Gemma 4 31B-IT and save as a quantized checkpoint.

Produces a packing-agnostic safetensors file (int values + per-group scales +
JSON header) that can later be loaded and packed for any backend via
``quant.load()`` and ``quant.pack_model()``.

No CUDA is needed — quantization runs on CPU. CUDA is only required at
load-and-pack time.

Usage:
    python quantize_and_save.py \\
        --model-dir ~/local/scripts/models/gemma-4-31B-it \\
        --output ./gemma4_31b_int4 \\
        --quant-recipe default
"""

import argparse
import os
import shutil

import torch.nn as nn

from executorch.examples.models.gemma4_31b.model import Gemma4_31B
from executorch.examples.models.gemma4_31b.quant import (
    QuantConfig,
    quantize_model,
    QuantRecipe,
    QuantRule,
    save,
)

# ---------------------------------------------------------------------------
# Production recipes for Gemma 4 31B.
#
# Layer sensitivity:
#   - v_proj and down_proj are the most sensitive to quantization error
#     (first/last quarter of layers especially so).
#   - q_proj, k_proj, o_proj, gate_proj, up_proj tolerate 4-bit well.
#   - embed_tokens is an index lookup — INT8 per-axis is nearly lossless.
#   - Norms and layer_scalar are tiny and must stay unquantized.

_INT4 = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
_INT4_HQQ = QuantConfig(bits=4, group_size=32, symmetric=True, method="hqq")
_INT8 = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
_INT8_PER_AXIS = QuantConfig(bits=8, group_size=5376, symmetric=True, method="min_max")
_EDGE_LAYERS = set(range(15)) | set(range(45, 60))

GEMMA4_31B_DEFAULT_RECIPE = QuantRecipe(
    rules=[
        QuantRule(r"embed_tokens\.weight", _INT8_PER_AXIS),
        QuantRule(r".*norm\.weight", None),
        QuantRule(r".*\.weight", _INT4),
    ]
)

GEMMA4_31B_SENSITIVE_RECIPE = QuantRecipe(
    rules=[
        QuantRule(r"embed_tokens\.weight", _INT8_PER_AXIS),
        QuantRule(r".*norm\.weight", None),
        QuantRule(r".*\.(v_proj|down_proj)\.weight", _INT8, layers=_EDGE_LAYERS),
        QuantRule(r".*\.weight", _INT4_HQQ),
    ]
)

_RECIPES = {
    "default": GEMMA4_31B_DEFAULT_RECIPE,
    "sensitive": GEMMA4_31B_SENSITIVE_RECIPE,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize Gemma 4 31B-IT and save as a quantized checkpoint."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="HuggingFace Gemma 4 31B-IT model dir.",
    )
    parser.add_argument(
        "--output",
        default="./gemma4_31b_int4",
        help="Output directory.",
    )
    parser.add_argument(
        "--quant-recipe",
        default="default",
        choices=list(_RECIPES),
        help="'default': int4 min_max linears + int8 per-axis embedding. "
        "'sensitive': int8 for edge-layer v_proj/down_proj, int4 hqq elsewhere.",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda"],
        help="Target backend (the quantized checkpoint is backend-agnostic, "
        "but this may influence default recipe selection in the future).",
    )
    args = parser.parse_args()

    recipe = _RECIPES[args.quant_recipe]

    print("Loading checkpoint (lazy, shard-by-shard)...")
    model, _ = Gemma4_31B.from_hf_checkpoint(args.model_dir)

    if model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr():
        print("Untying embed_tokens / lm_head...")
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())

    print(f"Quantizing with recipe '{args.quant_recipe}'...")
    quantized, unquantized = quantize_model(model, recipe)

    os.makedirs(args.output, exist_ok=True)
    safetensors_path = os.path.join(args.output, "model.safetensors")
    print("Saving quantized checkpoint...")
    n_tensors = save(quantized, unquantized, safetensors_path)

    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(args.model_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, filename))

    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"Saved {n_tensors} tensors ({size_mb:.1f} MB) to {args.output}/")
    print(f"Done. Use with: python export.py --prequantized {args.output}")


if __name__ == "__main__":
    main()
