#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export Llama model from HuggingFace using HF's StaticCache for state management.

This script exports a Llama model from HuggingFace without any source modification,
using HuggingFace's native StaticCache for KV cache management during inference.

Key features:
- No custom attention or cache implementations
- Uses HF's StaticCache via cache_implementation="static"
- Works with any HF Llama-compatible model
- Targets MLX backend

Usage:
    python -m executorch.examples.models.llama.export_llama_hf \
        --model-id "meta-llama/Llama-3.2-1B-Instruct" \
        --output llama_hf.pte

Requirements:
    pip install transformers torch
"""

import argparse
import logging
import os
from typing import Optional

import torch
import torch.nn as nn

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class LlamaHFExportable(nn.Module):
    """
    Exportable wrapper for HuggingFace Llama that uses StaticCache.

    This module wraps a HuggingFace Llama model and exposes a trace-friendly
    forward method that uses StaticCache for KV cache management.
    """

    def __init__(
        self,
        model_id: str,
        max_batch_size: int = 1,
        max_cache_len: int = 2048,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        from transformers import AutoModelForCausalLM, GenerationConfig

        logger.info(f"Loading model: {model_id}")

        # Use HF's cache_implementation="static" via GenerationConfig
        generation_config = GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            max_length=max_cache_len,
            cache_config={
                "batch_size": max_batch_size,
                "max_cache_len": max_cache_len,
            },
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="eager",  # Required for export
            generation_config=generation_config,
        )
        self.model.eval()
        self.config = self.model.config
        self.max_cache_len = max_cache_len

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with StaticCache.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            cache_position: Position indices for the cache [seq_len]

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            cache_position=cache_position,
            return_dict=True,
        )

        return outputs.logits

    def get_example_inputs(self, prefill_len: int = 4):
        """Get example inputs for tracing."""
        input_ids = torch.randint(0, self.config.vocab_size, (1, prefill_len))
        cache_position = torch.arange(prefill_len, dtype=torch.long)
        return (input_ids, cache_position)


def export_llama_hf(
    model_id: str,
    output_path: str,
    max_seq_len: int = 2048,
    dtype: str = "fp32",
    quantize: Optional[str] = None,
) -> None:
    """
    Export a HuggingFace Llama model using StaticCache.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        output_path: Path to save the .pte file
        max_seq_len: Maximum sequence length for KV cache
        dtype: Model dtype ("fp32", "fp16", "bf16")
        quantize: Quantization method ("int4", "int8", or None)
    """
    from transformers import AutoTokenizer

    # Map dtype string to torch dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    # Create exportable model
    model = LlamaHFExportable(
        model_id=model_id,
        max_batch_size=1,
        max_cache_len=max_seq_len,
        dtype=torch_dtype,
    )

    # Apply quantization if requested
    if quantize:
        logger.info(f"Applying {quantize} quantization...")
        try:
            from torchao.quantization.granularity import PerGroup
            from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_

            if quantize == "int4":
                quantize_(
                    model,
                    IntxWeightOnlyConfig(
                        weight_dtype=torch.int4, granularity=PerGroup(64)
                    ),
                )
            elif quantize == "int8":
                quantize_(
                    model,
                    IntxWeightOnlyConfig(
                        weight_dtype=torch.int8, granularity=PerGroup(64)
                    ),
                )
            else:
                logger.warning(f"Unknown quantization method: {quantize}")
        except ImportError:
            logger.error("TorchAO not installed. Run: pip install torchao")
            raise

    # Get example inputs for export
    example_inputs = model.get_example_inputs(prefill_len=4)

    # Set up dynamic shapes
    batch_dim = torch.export.Dim("batch", min=1, max=1)  # Fixed batch size
    seq_dim = torch.export.Dim("seq_len", min=1, max=max_seq_len)

    dynamic_shapes = {
        "input_ids": {0: batch_dim, 1: seq_dim},
        "cache_position": {0: seq_dim},
    }

    logger.info("Exporting model with torch.export...")
    with torch.no_grad():
        ep = torch.export.export(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,  # Allow non-strict mode for HF models
        )
        ep = ep.run_decompositions({})

    logger.info("Delegating to MLX backend...")
    import executorch.exir as exir
    from executorch.backends.apple.mlx import MLXPartitioner
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig

    edge_config = EdgeCompileConfig(
        _core_aten_ops_exception_list=[
            torch.ops.aten.scaled_dot_product_attention.default,
        ]
    )

    edge_program = exir.to_edge_transform_and_lower(
        ep,
        partitioner=[MLXPartitioner()],
        compile_config=edge_config,
    )

    logger.info("Exporting to ExecuTorch...")
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
        )
    )

    # Save the program
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    logger.info(f"Saved model to: {output_path}")
    logger.info(f"Program size: {len(executorch_program.buffer) / 1024 / 1024:.2f} MB")

    # Save tokenizer alongside for inference
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_path = output_path.replace(".pte", "_tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Saved tokenizer to: {tokenizer_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace Llama model using StaticCache to MLX"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llama_hf.pte",
        help="Output .pte file path",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length for KV cache",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Model dtype",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["int4", "int8"],
        default=None,
        help="Quantization method",
    )

    args = parser.parse_args()

    export_llama_hf(
        model_id=args.model_id,
        output_path=args.output,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()
