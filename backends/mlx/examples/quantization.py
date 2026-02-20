#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared quantization utilities for MLX LLM export scripts.
"""

import argparse
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def add_quantization_args(parser: argparse.ArgumentParser) -> None:
    """Add common quantization arguments to an argparse parser."""
    parser.add_argument(
        "--quantize-linear",
        type=str,
        choices=["int4", "int8"],
        default=None,
        help="Quantization method for linear layers",
    )
    parser.add_argument(
        "--quantize-embeddings",
        type=str,
        choices=["int4", "int8"],
        default=None,
        help="Quantization method for embedding layers",
    )
    parser.add_argument(
        "--linear-group-size",
        type=int,
        choices=[32, 64, 128],
        default=None,
        help="Group size for linear layer quantization (default: 32 for int4, 128 for int8)",
    )
    parser.add_argument(
        "--embeddings-group-size",
        type=int,
        choices=[32, 64, 128],
        default=None,
        help="Group size for embedding layer quantization (default: 32 for int4, 128 for int8)",
    )
    parser.add_argument(
        "--no-tie-word-embeddings",
        action="store_true",
        default=False,
        help="Disable tying lm_head weights to embedding after quantization, "
        "even if the model config has tie_word_embeddings=True",
    )


def _default_group_size(dtype_str: str) -> int:
    """Return the default group size for a given quantization dtype."""
    return 32 if dtype_str == "int4" else 128


def apply_quantization(
    model: torch.nn.Module,
    quantize_linear: Optional[str],
    quantize_embeddings: Optional[str],
    tie_word_embeddings: bool = False,
    linear_group_size: Optional[int] = None,
    embeddings_group_size: Optional[int] = None,
) -> None:
    """Apply TorchAO quantization to the model.

    Uses the HQQ (Half-Quadratic Quantization) scale-only algorithm for
    choosing quantization parameters.

    Args:
        model: The model to quantize. Expected to have model.model.embed_tokens
               and model.lm_head attributes for weight tying.
        quantize_linear: Quantization method for linear layers ("int4", "int8", or None)
        quantize_embeddings: Quantization method for embedding layers ("int4", "int8", or None)
        tie_word_embeddings: If True, re-tie lm_head.weight to embed_tokens.weight
            after quantization. Should be set from the HF model config's
            tie_word_embeddings field, and can be overridden with --no-tie-word-embeddings.
        linear_group_size: Group size for linear quantization. Defaults to 32 for int4, 128 for int8.
        embeddings_group_size: Group size for embedding quantization. Defaults to 32 for int4, 128 for int8.
    """
    if not quantize_linear and not quantize_embeddings:
        return

    logger.info("Applying quantization with TorchAO...")
    try:
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_
        from torchao.quantization.quantize_.workflows import IntxChooseQParamsAlgorithm

        qparams_algorithm = IntxChooseQParamsAlgorithm.HQQ_SCALE_ONLY

        if quantize_embeddings:
            embed_dtype = torch.int4 if quantize_embeddings == "int4" else torch.int8
            embed_group_size = embeddings_group_size or _default_group_size(
                quantize_embeddings
            )
            logger.info(
                f"Quantizing embedding layers with {quantize_embeddings} "
                f"(group size {embed_group_size})..."
            )
            quantize_(
                model,
                IntxWeightOnlyConfig(
                    weight_dtype=embed_dtype,
                    granularity=PerGroup(embed_group_size),
                    intx_choose_qparams_algorithm=qparams_algorithm,
                ),
                filter_fn=lambda m, fqn: isinstance(m, torch.nn.Embedding),
            )

        if quantize_linear:
            linear_dtype = torch.int4 if quantize_linear == "int4" else torch.int8
            linear_group_size = linear_group_size or _default_group_size(
                quantize_linear
            )
            logger.info(
                f"Quantizing linear layers with {quantize_linear} "
                f"(group size {linear_group_size})..."
            )
            quantize_(
                model,
                IntxWeightOnlyConfig(
                    weight_dtype=linear_dtype,
                    granularity=PerGroup(linear_group_size),
                    intx_choose_qparams_algorithm=qparams_algorithm,
                ),
                filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear),
            )

        if (
            tie_word_embeddings
            and hasattr(model, "lm_head")
            and hasattr(model, "model")
        ):
            embed = getattr(model.model, "embed_tokens", None)
            if embed is not None:
                model.lm_head.weight = embed.weight
                logger.info(
                    "Re-tied lm_head weights to embedding (tie_word_embeddings=True)"
                )

        logger.info("Applied quantization successfully")
    except ImportError:
        logger.error("TorchAO not installed. Run: pip install torchao")
        raise
