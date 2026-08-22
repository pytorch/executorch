#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export Voxtral model from HuggingFace using optimum-executorch, delegated to
the MLX backend.

Voxtral is a multimodal audio-language model (mistralai/Voxtral-Mini-3B-2507).
The exported .pte contains three methods:
  - audio_encoder   : mel-spectrogram features  →  audio embeddings
  - token_embedding : token ids                  →  text embeddings
  - text_decoder    : embeddings + cache_position →  next-token logits

Usage:
    python -m executorch.backends.mlx.examples.voxtral.export_voxtral_hf \
        --model-id "mistralai/Voxtral-Mini-3B-2507" \
        --output voxtral_mlx.pte

Requirements:
    pip install transformers torch optimum-executorch mistral-common librosa
"""

import argparse
import logging
import os
from typing import Optional

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def export_voxtral_hf(
    model_id: str,
    output_dir: str,
    max_seq_len: int = 1024,
    dtype: str = "bf16",
    qlinear: Optional[str] = None,
    qembedding: Optional[str] = None,
    qlinear_group_size: Optional[int] = None,
    qembedding_group_size: Optional[int] = None,
    max_audio_len: int = 300,
) -> None:
    """
    Export a HuggingFace Voxtral model using optimum-executorch, delegated to
    the MLX backend. Outputs two files:
      - model.pte: the main model (audio_encoder, token_embedding, text_decoder)
      - preprocessor.pte: mel spectrogram preprocessor for raw audio

    Args:
        model_id: HuggingFace model ID (e.g., "mistralai/Voxtral-Mini-3B-2507")
        output_dir: Directory to save the .pte files
        max_seq_len: Maximum sequence length for KV cache
        dtype: Model dtype ("fp32", "fp16", "bf16")
        qlinear: Quantization for linear layers ("4w", "8w", "nvfp4", or None)
        qembedding: Quantization for embeddings ("4w", "8w", "nvfp4", or None)
        qlinear_group_size: Group size for linear quantization (default: auto)
        qembedding_group_size: Group size for embedding quantization (default: auto)
        max_audio_len: Maximum audio length in seconds for preprocessor
    """
    from optimum.exporters.executorch.tasks.multimodal_text_to_text import (
        load_multimodal_text_to_text_model,
    )

    os.makedirs(output_dir, exist_ok=True)

    # --- Export preprocessor ---
    from executorch.extension.audio.mel_spectrogram import export_processor

    export_processor(
        output_file=os.path.join(output_dir, "preprocessor.pte"),
        backend="mlx",
        feature_size=128,
        max_audio_len=max_audio_len,
        stack_output=True,
    )

    # --- Export model ---
    logger.info(f"Loading model using optimum-executorch: {model_id}")

    dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    dtype_str = dtype_map.get(dtype, "bfloat16")

    exportable = load_multimodal_text_to_text_model(
        model_id,
        dtype=dtype_str,
        max_seq_len=max_seq_len,
    )

    # Apply quantization if requested
    from executorch.backends.mlx.llm.quantization import quantize_model_

    quantize_model_(
        exportable.model,
        qlinear_config=qlinear,
        qlinear_group_size=qlinear_group_size,
        qembedding_config=qembedding,
        qembedding_group_size=qembedding_group_size,
    )

    logger.info("Exporting model with torch.export...")
    exported_progs = exportable.export()
    logger.info(f"Exported methods: {list(exported_progs.keys())}")

    logger.info("Delegating to MLX backend...")
    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )

    edge_program = exir.to_edge_transform_and_lower(
        exported_progs,
        transform_passes=get_default_passes(),
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

    model_path = os.path.join(output_dir, "model.pte")
    with open(model_path, "wb") as f:
        f.write(executorch_program.buffer)

    logger.info(f"Saved model to: {model_path}")
    logger.info(f"Model size: {len(executorch_program.buffer) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace Voxtral model using optimum-executorch to MLX"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="mistralai/Voxtral-Mini-3B-2507",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="voxtral_mlx",
        help="Output directory for model.pte and preprocessor.pte",
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
    from executorch.backends.mlx.llm.quantization import add_quantization_args

    add_quantization_args(parser)
    parser.add_argument(
        "--max-audio-len",
        type=int,
        default=300,
        help="Maximum audio length in seconds for preprocessor",
    )

    args = parser.parse_args()

    export_voxtral_hf(
        model_id=args.model_id,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        qlinear=args.qlinear,
        qembedding=args.qembedding,
        qlinear_group_size=args.qlinear_group_size,
        qembedding_group_size=args.qembedding_group_size,
        max_audio_len=args.max_audio_len,
    )


if __name__ == "__main__":
    main()
