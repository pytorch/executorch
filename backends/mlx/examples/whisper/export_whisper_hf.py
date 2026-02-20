#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export Whisper model from HuggingFace using optimum-executorch's Seq2SeqLMExportableModule.

This script exports a Whisper model from HuggingFace without any source modification,
leveraging optimum-executorch's infrastructure that is proven to work.

Usage:
    python -m executorch.backends.mlx.examples.whisper.export_whisper_hf \
        --model-id "openai/whisper-tiny" \
        --output whisper_hf.pte

Requirements:
    pip install transformers torch optimum-executorch
"""

import argparse
import logging
import os
from typing import Optional

import torch

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def export_whisper_hf(  # noqa: C901
    model_id: str,
    output_path: str,
    max_cache_length: int = 256,
    max_hidden_seq_length: int = 1500,
    dtype: str = "bf16",
    quantize_linear: Optional[str] = None,
    quantize_embeddings: Optional[str] = None,
) -> None:
    """
    Export a HuggingFace Whisper model using optimum-executorch's Seq2SeqLMExportableModule.

    Args:
        model_id: HuggingFace model ID (e.g., "openai/whisper-tiny")
        output_path: Path to save the .pte file
        max_cache_length: Maximum sequence length for generation (default: 256)
        max_hidden_seq_length: Maximum encoder output sequence length (default: 1500)
        dtype: Model dtype ("fp32", "fp16", "bf16")
        quantize_linear: Quantization method for linear layers ("int4", "int8", or None)
        quantize_embeddings: Quantization method for embedding layers ("int4", "int8", or None)
    """
    from optimum.exporters.executorch.tasks.asr import load_seq2seq_speech_model

    logger.info(f"Loading model using optimum-executorch: {model_id}")

    # Map dtype string to proper format for optimum-executorch
    dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    dtype_str = dtype_map.get(dtype, "float32")

    # Load using optimum-executorch's ASR task which handles all the necessary setup
    exportable = load_seq2seq_speech_model(
        model_id,
        dtype=dtype_str,
        max_cache_length=max_cache_length,
        max_hidden_seq_length=max_hidden_seq_length,
    )

    # Apply quantization if requested
    if quantize_linear or quantize_embeddings:
        logger.info("Applying quantization with TorchAO...")
        try:
            from torchao.quantization.granularity import PerGroup
            from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

            # Get the underlying model from the exportable module
            model = exportable.model

            # Quantize embedding layers
            if quantize_embeddings:
                embed_dtype = (
                    torch.int4 if quantize_embeddings == "int4" else torch.int8
                )
                embed_group_size = 32 if quantize_embeddings == "int4" else 128
                logger.info(
                    f"Quantizing embedding layers with {quantize_embeddings} (group size {embed_group_size})..."
                )
                quantize_(
                    model,
                    IntxWeightOnlyConfig(
                        weight_dtype=embed_dtype, granularity=PerGroup(embed_group_size)
                    ),
                    filter_fn=lambda m, fqn: isinstance(m, torch.nn.Embedding),
                )

            # Quantize linear layers
            if quantize_linear:
                linear_dtype = torch.int4 if quantize_linear == "int4" else torch.int8
                linear_group_size = 32 if quantize_linear == "int4" else 128
                logger.info(
                    f"Quantizing linear layers with {quantize_linear} (group size {linear_group_size})..."
                )
                quantize_(
                    model,
                    IntxWeightOnlyConfig(
                        weight_dtype=linear_dtype,
                        granularity=PerGroup(linear_group_size),
                    ),
                    filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear),
                )

            logger.info("Applied quantization successfully")
        except ImportError:
            logger.error("TorchAO not installed. Run: pip install torchao")
            raise

    # Export using optimum-executorch's export method
    logger.info("Exporting model with torch.export...")
    exported_progs = exportable.export()

    # Lower to MLX backend
    logger.info("Delegating to MLX backend...")
    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    # Match optimum-executorch's EdgeCompileConfig
    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )

    # Convert single exported program to dict format if needed
    if len(exported_progs) == 1:
        exported_progs = {"forward": next(iter(exported_progs.values()))}

    edge_program = exir.to_edge_transform_and_lower(
        exported_progs,
        partitioner=[MLXPartitioner()],
        compile_config=edge_config,
        constant_methods=exportable.metadata,
    )

    logger.info("Exporting to ExecuTorch...")
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )

    # Save the program
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    logger.info(f"Saved model to: {output_path}")
    logger.info(f"Program size: {len(executorch_program.buffer) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace Whisper model using optimum-executorch to MLX"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/whisper-tiny",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="whisper_hf.pte",
        help="Output .pte file path",
    )
    parser.add_argument(
        "--max-cache-length",
        type=int,
        default=256,
        help="Maximum sequence length for generation",
    )
    parser.add_argument(
        "--max-hidden-seq-length",
        type=int,
        default=1500,
        help="Maximum encoder output sequence length",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Model dtype",
    )
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

    args = parser.parse_args()

    export_whisper_hf(
        model_id=args.model_id,
        output_path=args.output,
        max_cache_length=args.max_cache_length,
        max_hidden_seq_length=args.max_hidden_seq_length,
        dtype=args.dtype,
        quantize_linear=args.quantize_linear,
        quantize_embeddings=args.quantize_embeddings,
    )


if __name__ == "__main__":
    main()
