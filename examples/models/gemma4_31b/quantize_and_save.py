# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantize Gemma 4 31B-IT and save as a quantized checkpoint.

Produces a safetensors file containing torchao tensor subclasses
(``Int4Tensor``, ``IntxUnpackedToInt8Tensor``) that can be loaded and
packed for any backend via the generic ``load_and_pack_for_*`` APIs with
Gemma4-specific custom packers.

The default recipe runs on CPU. The sensitive recipe requires CUDA for
HQQ asymmetric quantization.

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
from executorch.examples.models.gemma4_31b.pack_vision import (
    quantize_vision_position_table,
)
from executorch.examples.models.gemma4_31b.quant import (
    QuantConfig,
    quantize_model,
    QuantRecipe,
    QuantRule,
)

# ---------------------------------------------------------------------------
# Production recipes for Gemma 4 31B (vision + text in one rule set).
#
# Layer sensitivity (text decoder):
#   - v_proj and down_proj are the most sensitive to quantization error
#     (first/last quarter of layers especially so).
#   - q_proj, k_proj, o_proj, gate_proj, up_proj tolerate 4-bit well.
#   - embed_tokens is an index lookup — INT8 per-axis is nearly lossless.
#   - Norms and layer_scalar are tiny and must stay unquantized.
#
# Vision modality:
#   - Vision tower linears are small + accuracy-sensitive; they stay bf16.
#   - The vision multimodal projector (``embed_vision.*``) also stays bf16.
#   - The patch_embedder's position_embedding_table is the one "real" quant
#     on the vision side: bf16 → INT8 per-channel, applied explicitly before
#     the generic quantize_model parameter walk.

_INT4 = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
_INT4_HQQ = QuantConfig(bits=4, group_size=32, symmetric=False, method="hqq")
_INT8 = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
_INT8_PER_AXIS = QuantConfig(  # group_size = hidden_size (5376) for Gemma 4 31B
    bits=8, group_size=5376, symmetric=True, method="min_max"
)
_EDGE_LAYERS = set(range(15)) | set(range(45, 60))

# Shared vision rules: every vision-side weight stays bf16. The PE table is
# absent from the parameter walk after quantize_gemma4_vision_position_table
# replaces it with int8 buffers.


def quantize_gemma4_vision_position_table(model: nn.Module) -> None:
    vision_tower = getattr(model, "vision_tower", None)
    if vision_tower is not None:
        quantize_vision_position_table(vision_tower)


_VISION_RULES = [
    QuantRule(r"vision_tower\..*", None),
    QuantRule(r"embed_vision\..*", None),
]

GEMMA4_31B_DEFAULT_RECIPE = QuantRecipe(
    rules=[
        QuantRule(r"embed_tokens\.weight", _INT8_PER_AXIS),
        *_VISION_RULES,
        QuantRule(r".*norm\.weight", None),
        QuantRule(r".*\.weight", _INT4),
    ],
)

GEMMA4_31B_SENSITIVE_RECIPE = QuantRecipe(
    rules=[
        QuantRule(r"embed_tokens\.weight", _INT8_PER_AXIS),
        *_VISION_RULES,
        QuantRule(r".*norm\.weight", None),
        QuantRule(r".*\.(v_proj|down_proj)\.weight", _INT8, layers=_EDGE_LAYERS),
        QuantRule(r".*\.weight", _INT4_HQQ),
    ],
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

    # Single quantization entry point. ``quantize_model`` handles both
    # modalities in one pass:
    #   - text decoder linears -> INT4 / INT8 per the recipe;
    #   - vision tower + embed_vision linears -> stay bf16 (recipe rule);
    #   - vision PE table -> INT8 per-channel (explicit pre-quantization call).
    print(f"Quantizing with recipe '{args.quant_recipe}'...")
    quantize_gemma4_vision_position_table(model)
    state_dict = quantize_model(model, recipe, verbose=True)

    os.makedirs(args.output, exist_ok=True)
    safetensors_path = os.path.join(args.output, "model.safetensors")
    print("Saving quantized checkpoint...")
    from safetensors.torch import save_file
    from torchao.prototype.safetensors.safetensors_support import (
        flatten_tensor_state_dict,
    )

    tensors_data, metadata = flatten_tensor_state_dict(state_dict)
    save_file(tensors_data, safetensors_path, metadata=metadata)
    n_tensors = len(state_dict)

    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(args.model_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, filename))

    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"Saved {n_tensors} tensors ({size_mb:.1f} MB) to {args.output}/")
    print(f"Done. Use with: python export.py --prequantized {args.output}")


if __name__ == "__main__":
    main()
