#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared quantization and export utilities for Gemma 4."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def parse_quantize(quantize: str) -> tuple:
    """Parse composite --quantize flag into (linear_mode, emb_mode).

    Examples:
        "8da8w"       -> ("8da8w", None)
        "8da8w+emb8"  -> ("8da8w", "emb8")
        "emb4"        -> (None, "emb4")
        "none"        -> (None, None)
    """
    if quantize == "none":
        return None, None
    parts = quantize.split("+")
    linear_mode = None
    emb_mode = None
    for part in parts:
        if part in ("8da8w", "8da4w"):
            linear_mode = part
        elif part in ("emb8", "emb4"):
            emb_mode = part
    return linear_mode, emb_mode


def apply_linear_quantization(
    model: nn.Module,
    mode: str,
    group_size: int = 128,
) -> nn.Module:
    """Apply TorchAO dynamic quantization to Linear layers.

    Uses the shared quantize_model_() from extension/llm/export/quantize.py.
    """
    from executorch.extension.llm.export.quantize import quantize_model_

    logger.info(f"Applying linear quantization: {mode}")
    # Only pass group_size for modes that use it (8da4w).
    # 8da8w uses PerAxis(0) — passing group_size would override to PerGroup.
    qlinear_group_size = group_size if mode == "8da4w" else None
    quantize_model_(
        model,
        qlinear_config=mode,
        qlinear_group_size=qlinear_group_size,
        skip_incompatible_shapes=True,
    )
    return model


def apply_embedding_quantization(
    model: nn.Module,
    mode: str,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Apply embedding quantization (INT8 or INT4)."""
    from executorch.examples.models.llama.source_transformation.quantize import (
        EmbeddingQuantHandler,
        QuantizedGroupEmbedding,
    )

    bitwidth = 4 if mode == "emb4" else 8
    packed = bitwidth == 4
    logger.info(f"Quantizing embeddings to int{bitwidth} (packed={packed})")

    model = EmbeddingQuantHandler(
        model,
        bitwidth=bitwidth,
        group_size=None,
        precision=dtype,
        packed=packed,
    ).quantized_model()

    for _name, mod in model.named_modules():
        if isinstance(mod, QuantizedGroupEmbedding):
            if mod.scales.dtype == torch.bfloat16:
                mod.register_buffer("scales", mod.scales.to(torch.float16))

    return model
