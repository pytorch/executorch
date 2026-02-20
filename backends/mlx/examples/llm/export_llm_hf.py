#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export Llama model from HuggingFace to MLX backend.

By default, uses optimum-executorch's CausalLMExportableModule which provides
a proven export pipeline. Optional flags enable custom MLX-optimized components:

  --use-custom-sdpa   Register MLX attention (mlx::custom_sdpa) which handles
                      K/V slicing and causal masking internally.
  --use-custom-kv-cache  Replace HF's StaticCache with HFStaticCache that uses
                         mlx::kv_cache_update for optimized cache updates.

When neither flag is set, the script behaves identically to the original
optimum-executorch export pipeline.

Usage:
    # Baseline (optimum-executorch pipeline):
    python -m executorch.backends.mlx.examples.llm.export_llm_hf \\
        --model-id "unsloth/Llama-3.2-1B-Instruct" \\
        --output llama_hf.pte

    # With custom MLX components:
    python -m executorch.backends.mlx.examples.llm.export_llm_hf \\
        --model-id "unsloth/Llama-3.2-1B-Instruct" \\
        --output llama_hf_mlx.pte \\
        --use-custom-sdpa \\
        --use-custom-kv-cache

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


def _export_with_optimum(
    model_id: str,
    output_path: str,
    max_seq_len: int,
    dtype: str,
    quantize_linear: Optional[str],
    quantize_embeddings: Optional[str],
    no_tie_word_embeddings: bool = False,
) -> None:
    """
    Export using optimum-executorch's CausalLMExportableModule.

    This is the default pipeline when no custom flags are set.
    """
    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from optimum.exporters.executorch.tasks.causal_lm import load_causal_lm_model

    dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    dtype_str = dtype_map.get(dtype, "bfloat16")

    logger.info(f"Loading model using optimum-executorch: {model_id}")
    exportable = load_causal_lm_model(
        model_id,
        dtype=dtype_str,
        max_seq_len=max_seq_len,
    )

    from executorch.backends.mlx.examples.llm.quantize import apply_quantization

    apply_quantization(
        exportable.model,
        quantize_linear,
        quantize_embeddings,
        tie_word_embeddings=getattr(
            exportable.model.config, "tie_word_embeddings", False
        )
        and not no_tie_word_embeddings,
    )

    logger.info("Exporting model with torch.export...")
    exported_progs = exportable.export()

    logger.info("Delegating to MLX backend...")
    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )

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

    _save_program(executorch_program, output_path)


def _export_with_custom_components(
    model_id: str,
    output_path: str,
    max_seq_len: int,
    dtype: str,
    quantize_linear: Optional[str],
    quantize_embeddings: Optional[str],
    use_custom_sdpa: bool,
    use_custom_kv_cache: bool,
    no_tie_word_embeddings: bool = False,
) -> None:
    """
    Export using direct HF model with custom MLX components.

    Used when --use-custom-sdpa and/or --use-custom-kv-cache are set.
    """
    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from transformers import AutoModelForCausalLM
    from transformers.integrations.executorch import (
        TorchExportableModuleWithStaticCache,
    )

    torch_dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = torch_dtype_map.get(dtype, torch.bfloat16)

    if use_custom_sdpa:
        from executorch.backends.mlx.examples.attention import register_mlx_attention

        register_mlx_attention()
        logger.info("Registered MLX custom SDPA attention")

    attn_implementation = "mlx" if use_custom_sdpa else None

    logger.info(f"Loading HuggingFace model: {model_id}")
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    model.generation_config.cache_implementation = "static"
    model.generation_config.cache_config = {
        "batch_size": 1,
        "max_cache_len": max_seq_len,
    }
    model.eval()

    logger.info("Creating TorchExportableModuleWithStaticCache wrapper...")
    exportable = TorchExportableModuleWithStaticCache(
        model=model,
        batch_size=1,
        max_cache_len=max_seq_len,
    )

    if use_custom_kv_cache:
        from executorch.backends.mlx.examples.source_transformation import (
            replace_hf_cache_with_mlx,
        )

        logger.info("Replacing HuggingFace StaticCache with HFStaticCache...")
        replace_hf_cache_with_mlx(
            exportable,
            model.config,
            max_batch_size=1,
            max_cache_len=max_seq_len,
            dtype=torch_dtype,
        )
        logger.info("  HFStaticCache installed successfully")

    from executorch.backends.mlx.examples.llm.quantize import apply_quantization

    apply_quantization(
        exportable.model,
        quantize_linear,
        quantize_embeddings,
        tie_word_embeddings=getattr(model.config, "tie_word_embeddings", False)
        and not no_tie_word_embeddings,
    )

    logger.info("Exporting model with torch.export...")
    seq_length = 3
    example_input_ids = torch.zeros((1, seq_length), dtype=torch.long)
    example_cache_position = torch.arange(seq_length, dtype=torch.long)

    seq_len_dim = torch.export.Dim("seq_length_dim", max=max_seq_len - 1)
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "cache_position": {0: seq_len_dim},
    }

    with torch.no_grad():
        exported_program = torch.export.export(
            exportable,
            args=(),
            kwargs={
                "input_ids": example_input_ids,
                "cache_position": example_cache_position,
            },
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )

    logger.info("Export completed successfully")
    for sym, constraint in exported_program.range_constraints.items():
        logger.info(f"  Range constraint: {sym}: {constraint}")

    logger.info("Delegating to MLX backend...")
    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )

    edge_program = exir.to_edge_transform_and_lower(
        {"forward": exported_program},
        partitioner=[MLXPartitioner()],
        compile_config=edge_config,
    )

    logger.info("Exporting to ExecuTorch...")
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
        )
    )

    _save_program(executorch_program, output_path)


def _save_program(executorch_program, output_path: str) -> None:
    """Save the ExecuTorch program to disk."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    logger.info(f"Saved model to: {output_path}")
    logger.info(f"Program size: {len(executorch_program.buffer) / 1024 / 1024:.2f} MB")


def export_llama_hf(
    model_id: str,
    output_path: str,
    max_seq_len: int = 1024,
    dtype: str = "bf16",
    quantize_linear: Optional[str] = None,
    quantize_embeddings: Optional[str] = None,
    use_custom_sdpa: bool = False,
    use_custom_kv_cache: bool = False,
    no_tie_word_embeddings: bool = False,
) -> None:
    """
    Export a HuggingFace Llama model to ExecuTorch with MLX backend.

    Args:
        model_id: HuggingFace model ID
        output_path: Path to save the .pte file
        max_seq_len: Maximum sequence length for KV cache
        dtype: Model dtype ("fp32", "fp16", "bf16")
        quantize_linear: Quantization for linear layers ("int4", "int8", or None)
        quantize_embeddings: Quantization for embeddings ("int4", "int8", or None)
        use_custom_sdpa: Use MLX custom SDPA (mlx::custom_sdpa)
        use_custom_kv_cache: Use MLX custom KV cache (mlx::kv_cache_update)
    """
    if use_custom_sdpa or use_custom_kv_cache:
        logger.info(
            f"Using custom components: sdpa={use_custom_sdpa}, "
            f"kv_cache={use_custom_kv_cache}"
        )
        _export_with_custom_components(
            model_id=model_id,
            output_path=output_path,
            max_seq_len=max_seq_len,
            dtype=dtype,
            quantize_linear=quantize_linear,
            quantize_embeddings=quantize_embeddings,
            use_custom_sdpa=use_custom_sdpa,
            use_custom_kv_cache=use_custom_kv_cache,
            no_tie_word_embeddings=no_tie_word_embeddings,
        )
    else:
        logger.info("Using optimum-executorch pipeline (no custom components)")
        _export_with_optimum(
            model_id=model_id,
            output_path=output_path,
            max_seq_len=max_seq_len,
            dtype=dtype,
            quantize_linear=quantize_linear,
            quantize_embeddings=quantize_embeddings,
            no_tie_word_embeddings=no_tie_word_embeddings,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace Llama model to MLX backend"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .pte file path",
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
    from executorch.backends.mlx.examples.llm.quantize import add_quantization_args

    add_quantization_args(parser)
    parser.add_argument(
        "--use-custom-sdpa",
        action="store_true",
        default=False,
        help="Use MLX custom SDPA (mlx::custom_sdpa) for attention",
    )
    parser.add_argument(
        "--use-custom-kv-cache",
        action="store_true",
        default=False,
        help="Use MLX custom KV cache (mlx::kv_cache_update)",
    )

    args = parser.parse_args()

    export_llama_hf(
        model_id=args.model_id,
        output_path=args.output,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        quantize_linear=args.quantize_linear,
        quantize_embeddings=args.quantize_embeddings,
        use_custom_sdpa=args.use_custom_sdpa,
        use_custom_kv_cache=args.use_custom_kv_cache,
        no_tie_word_embeddings=args.no_tie_word_embeddings,
    )


if __name__ == "__main__":
    main()
