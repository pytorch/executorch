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

import torch

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def export_preprocessor(
    output_path: str,
    feature_size: int = 128,
    max_audio_len: int = 300,
) -> None:
    """
    Export the Voxtral audio preprocessor (mel spectrogram) to MLX.

    Args:
        output_path: Path to save the preprocessor .pte file
        feature_size: Mel spectrogram feature dimension (128 for Voxtral)
        max_audio_len: Maximum audio length in seconds
    """
    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.extension.audio.mel_spectrogram import WhisperAudioProcessor
    from torch.export import Dim

    logger.info("Exporting audio preprocessor with MLX backend...")

    model = WhisperAudioProcessor(
        feature_size=feature_size,
        max_audio_len=max_audio_len,
        stack_output=True,
    )

    audio_tensor = torch.randn(93680)
    shapes_collection = torch.export.ShapesCollection()
    max_n_chunks = int(model.max_audio_len * model.n_samples)
    shapes_collection[audio_tensor] = {0: Dim.DYNAMIC(max=max_n_chunks)}

    with torch.no_grad(), torch.fx.experimental._config.patch(
        backed_size_oblivious=True
    ):
        ep = torch.export.export(
            model, (audio_tensor,), dynamic_shapes=shapes_collection, strict=True
        )

        edge_program = exir.to_edge_transform_and_lower(
            ep,
            partitioner=[MLXPartitioner()],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        executorch_program = edge_program.to_executorch(
            config=ExecutorchBackendConfig(
                extract_delegate_segments=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            )
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    logger.info(f"Saved preprocessor to: {output_path}")
    logger.info(
        f"Preprocessor size: {len(executorch_program.buffer) / 1024 / 1024:.2f} MB"
    )


def export_voxtral_hf(
    model_id: str,
    output_dir: str,
    max_seq_len: int = 1024,
    dtype: str = "bf16",
    quantize_linear: Optional[str] = None,
    quantize_linear_group_size: int = 32,
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
        quantize_linear: Quantization for linear layers ("int4", "int8", or None)
        quantize_linear_group_size: Group size for linear quantization
        max_audio_len: Maximum audio length in seconds for preprocessor
    """
    from optimum.exporters.executorch.tasks.multimodal_text_to_text import (
        load_multimodal_text_to_text_model,
    )

    os.makedirs(output_dir, exist_ok=True)

    # --- Export preprocessor ---
    export_preprocessor(
        output_path=os.path.join(output_dir, "preprocessor.pte"),
        max_audio_len=max_audio_len,
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
    if quantize_linear:
        logger.info("Applying quantization with TorchAO...")
        try:
            from torchao.quantization.granularity import PerGroup
            from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

            model = exportable.model
            linear_dtype = torch.int4 if quantize_linear == "int4" else torch.int8
            config = IntxWeightOnlyConfig(
                weight_dtype=linear_dtype,
                granularity=PerGroup(quantize_linear_group_size),
                intx_choose_qparams_algorithm="hqq_scale_only",
            )

            logger.info(
                f"Quantizing linear layers with {quantize_linear} "
                f"(group_size={quantize_linear_group_size}, algorithm=hqq_scale_only)..."
            )
            quantize_(
                model,
                config,
                filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear),
            )

            logger.info("Applied quantization successfully")
        except ImportError:
            logger.error("TorchAO not installed. Run: pip install torchao")
            raise

    logger.info("Exporting model with torch.export...")
    exported_progs = exportable.export()
    logger.info(f"Exported methods: {list(exported_progs.keys())}")

    logger.info("Delegating to MLX backend...")
    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )

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
    parser.add_argument(
        "--quantize-linear",
        type=str,
        choices=["int4", "int8"],
        default=None,
        help="Quantization method for linear layers",
    )
    parser.add_argument(
        "--quantize-linear-group-size",
        type=int,
        default=32,
        help="Group size for linear quantization",
    )
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
        quantize_linear=args.quantize_linear,
        quantize_linear_group_size=args.quantize_linear_group_size,
        max_audio_len=args.max_audio_len,
    )


if __name__ == "__main__":
    main()
