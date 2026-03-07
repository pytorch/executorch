#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Quantization argument helpers for MLX LLM export scripts.

Re-exports quantize_model_ from the shared ExecuTorch LLM export library
and provides add_quantization_args for MLX export CLI scripts.
"""

import argparse

from executorch.extension.llm.export.quantize import quantize_model_

__all__ = ["add_quantization_args", "quantize_model_"]


def add_quantization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--qlinear",
        type=str,
        choices=["4w", "8w", "nvfp4"],
        default=None,
        help="Quantization config for linear layers",
    )
    parser.add_argument(
        "--qembedding",
        type=str,
        choices=["4w", "8w", "nvfp4"],
        default=None,
        help="Quantization config for embedding layers",
    )
    parser.add_argument(
        "--qlinear-group-size",
        type=int,
        choices=[32, 64, 128],
        default=None,
        help="Group size for linear layer quantization (default: 32)",
    )
    parser.add_argument(
        "--qembedding-group-size",
        type=int,
        choices=[32, 64, 128],
        default=None,
        help="Group size for embedding layer quantization (default: 128)",
    )
    parser.add_argument(
        "--no-tie-word-embeddings",
        action="store_true",
        default=False,
        help="Disable tying lm_head weights to embedding after quantization, "
        "even if the model config has tie_word_embeddings=True",
    )
    parser.add_argument(
        "--nvfp4-per-tensor-scale",
        action="store_true",
        default=False,
        help="Enable per-tensor scale for NVFP4 quantization (improves accuracy)",
    )
