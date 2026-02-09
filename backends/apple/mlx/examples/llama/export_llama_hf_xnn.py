#!/usr/bin/env python3
# @nocommit
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export Llama model from HuggingFace using optimum-executorch's XNNPACK recipe.

Usage:
    python -m executorch.backends.apple.mlx.examples.llama.export_llama_hf_xnn \
        --model-id "unsloth/Llama-3.2-1B-Instruct" \
        --output-dir llama_hf_xnn

Requirements:
    pip install transformers torch optimum-executorch
"""

import argparse
import logging

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace Llama model using optimum-executorch XNNPACK recipe"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="llama_hf_xnn",
        help="Output directory for .pte file",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum sequence length for KV cache",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Model dtype",
    )
    parser.add_argument(
        "--qlinear",
        type=str,
        choices=["8da4w", "4w", "8w"],
        default=None,
        help="Quantization config for linear layers (8da4w recommended for XNNPACK)",
    )
    parser.add_argument(
        "--qembedding",
        type=str,
        choices=["4w", "8w"],
        default=None,
        help="Quantization config for embedding layers",
    )
    parser.add_argument(
        "--use-custom-sdpa-and-kv-cache",
        action="store_true",
        default=False,
        help="Use custom SDPA and KV cache implementations",
    )

    args = parser.parse_args()

    from optimum.exporters.executorch import main_export

    dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    dtype_str = dtype_map.get(args.dtype, "bfloat16")

    main_export(
        model_name_or_path=args.model_id,
        task="text-generation",
        recipe="xnnpack",
        output_dir=args.output_dir,
        dtype=dtype_str,
        max_seq_len=args.max_seq_len,
        qlinear=args.qlinear,
        qlinear_group_size=32,
        qembedding=args.qembedding,
        qembedding_group_size=32,
        use_custom_sdpa=args.use_custom_sdpa_and_kv_cache,
        use_custom_kv_cache=args.use_custom_sdpa_and_kv_cache,
    )


if __name__ == "__main__":
    main()
