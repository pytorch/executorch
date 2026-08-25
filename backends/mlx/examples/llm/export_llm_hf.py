#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export LLM model from HuggingFace to MLX backend.

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


def resolve_prefill_chunk_size(
    prefill_chunk_size: Optional[int],
    max_ctx_len: int,
    sliding_window: Optional[int],
) -> int:
    """Resolve the largest single forward step this export will trace.

    The result bounds the traced ``seq_len`` dimension and, for sliding-window
    models, doubles as the ring buffer's ``max_write_len``. ``None`` means "as
    large as the model allows".
    """
    ceiling = (
        max_ctx_len if sliding_window is None else min(max_ctx_len, sliding_window)
    )
    if prefill_chunk_size is None:
        return ceiling
    if prefill_chunk_size < 1 or prefill_chunk_size > ceiling:
        limit = (
            f"max_ctx_len {max_ctx_len}"
            if sliding_window is None or max_ctx_len <= sliding_window
            else f"sliding window {sliding_window}"
        )
        raise ValueError(
            f"--prefill-chunk-size {prefill_chunk_size} must be in [1, {ceiling}], "
            f"bounded by the {limit}. It is the largest step this export traces, "
            "and a ring layer holds window + chunk - 1 slots."
        )
    return prefill_chunk_size


def _export_with_optimum(
    model_id: str,
    revision: Optional[str],
    output_path: str,
    max_ctx_len: int,
    dtype: str,
    qlinear: Optional[str],
    qembedding: Optional[str],
    no_tie_word_embeddings: bool = False,
    qlinear_group_size: Optional[int] = None,
    qembedding_group_size: Optional[int] = None,
) -> None:
    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from optimum.exporters.executorch.tasks.causal_lm import load_causal_lm_model

    dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    dtype_str = dtype_map.get(dtype, "bfloat16")

    logger.info(f"Loading model using optimum-executorch: {model_id}")
    exportable = load_causal_lm_model(
        model_id,
        revision=revision,
        dtype=dtype_str,
        max_seq_len=max_ctx_len,
    )

    from executorch.backends.mlx.llm.quantization import quantize_model_

    quantize_model_(
        exportable.model,
        qlinear_config=qlinear,
        qlinear_group_size=qlinear_group_size,
        qembedding_config=qembedding,
        qembedding_group_size=qembedding_group_size,
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

    # optimum drives its own torch.export call, so it owns the seq-len bound.
    constant_methods = dict(exportable.metadata)
    constant_methods["get_max_ctx_len"] = max_ctx_len

    edge_program = exir.to_edge_transform_and_lower(
        exported_progs,
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=edge_config,
        constant_methods=constant_methods,
    )

    logger.info("Exporting to ExecuTorch...")
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )

    _save_program(executorch_program, output_path)


def build_hf_exported_program(
    model_id: str,
    revision: Optional[str],
    max_ctx_len: int,
    dtype: str,
    qlinear: Optional[str],
    qembedding: Optional[str],
    use_custom_sdpa: bool,
    use_custom_kv_cache: bool,
    no_tie_word_embeddings: bool = False,
    qlinear_group_size: Optional[int] = None,
    qembedding_group_size: Optional[int] = None,
    tap_layers: Optional[list[int]] = None,
    prefill_chunk_size: Optional[int] = None,
):
    """Build the torch.export program for an HF model with custom MLX components.

    Returns ``(exported_program, resolved_prefill_chunk_size)``. The resolved chunk
    is the traced ``seq_len`` upper bound and is what callers should publish as
    ``get_prefill_chunk_size``.
    """
    from transformers import AutoModelForCausalLM

    torch_dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = torch_dtype_map.get(dtype, torch.bfloat16)

    if use_custom_sdpa:
        from executorch.backends.mlx.llm.hf_attention import register_mlx_attention

        register_mlx_attention()
        logger.info("Registered MLX custom SDPA attention")

    attn_implementation = "mlx" if use_custom_sdpa else None

    logger.info(f"Loading HuggingFace model: {model_id}")
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # Check if model uses sliding window attention. Multimodal configs like
    # Gemma 4 keep transformer attributes under text_config.
    text_config = model.config.get_text_config()
    sliding_window = getattr(text_config, "sliding_window", None)
    effective_cache_len = max_ctx_len
    prefill_chunk_size = resolve_prefill_chunk_size(
        prefill_chunk_size, max_ctx_len, sliding_window
    )
    if sliding_window is not None:
        logger.info(
            f"Model has sliding_window={sliding_window}; "
            f"cache length {effective_cache_len}"
        )

    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    model.generation_config.cache_config = {
        "batch_size": 1,
        "max_cache_len": effective_cache_len,
    }
    text_config = model.config.get_text_config()
    text_config.use_cache = True
    model.eval()

    from executorch.backends.mlx.llm.exportable import (
        create_hf_exportable,
        install_mlx_cache,
    )

    exportable = create_hf_exportable(
        model=model,
        max_cache_len=effective_cache_len,
        tap_layers=tap_layers,
    )

    if use_custom_kv_cache:
        install_mlx_cache(
            exportable,
            config=model.config,
            max_batch_size=1,
            max_cache_len=effective_cache_len,
            dtype=torch_dtype,
            prefill_chunk_size=prefill_chunk_size,
        )
        logger.info("  MLX cache installed successfully")

        if use_custom_sdpa and sliding_window is not None:
            from executorch.backends.mlx.llm.hf_attention import (
                register_mlx_sliding_window_attention,
            )

            register_mlx_sliding_window_attention(exportable)
            model.config._attn_implementation = "mlx_sliding_window"
            text_config._attn_implementation = "mlx_sliding_window"
            logger.info("Registered MLX sliding-window SDPA")

    from executorch.backends.mlx.llm.quantization import quantize_model_

    quantize_model_(
        exportable.model,
        qlinear_config=qlinear,
        qlinear_group_size=qlinear_group_size,
        qembedding_config=qembedding,
        qembedding_group_size=qembedding_group_size,
        tie_word_embeddings=getattr(model.config, "tie_word_embeddings", False)
        and not no_tie_word_embeddings,
    )

    logger.info("Exporting model with torch.export...")
    seq_length = 3
    example_input_ids = torch.zeros((1, seq_length), dtype=torch.long)
    example_cache_position = torch.arange(seq_length, dtype=torch.long)

    # prefill_chunk_size is the largest single step: it bounds the traced seq_len
    # and sizes the ring buffer as window + chunk - 1.
    seq_len_dim = torch.export.Dim("seq_length_dim", max=prefill_chunk_size)
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

    return exported_program, prefill_chunk_size


def _export_with_custom_components(
    model_id: str,
    revision: Optional[str],
    output_path: str,
    max_ctx_len: int,
    dtype: str,
    qlinear: Optional[str],
    qembedding: Optional[str],
    use_custom_sdpa: bool,
    use_custom_kv_cache: bool,
    no_tie_word_embeddings: bool = False,
    qlinear_group_size: Optional[int] = None,
    qembedding_group_size: Optional[int] = None,
    tap_layers: Optional[list[int]] = None,
    prefill_chunk_size: Optional[int] = None,
) -> None:
    """Export using direct HF model with custom MLX components."""
    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    exported_program, prefill_chunk_size = build_hf_exported_program(
        model_id=model_id,
        revision=revision,
        max_ctx_len=max_ctx_len,
        dtype=dtype,
        qlinear=qlinear,
        qembedding=qembedding,
        use_custom_sdpa=use_custom_sdpa,
        use_custom_kv_cache=use_custom_kv_cache,
        no_tie_word_embeddings=no_tie_word_embeddings,
        qlinear_group_size=qlinear_group_size,
        qembedding_group_size=qembedding_group_size,
        tap_layers=tap_layers,
        prefill_chunk_size=prefill_chunk_size,
    )

    logger.info("Delegating to MLX backend...")
    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )

    constant_methods = {
        "get_max_ctx_len": max_ctx_len,
        "get_prefill_chunk_size": prefill_chunk_size,
    }

    edge_program = exir.to_edge_transform_and_lower(
        {"forward": exported_program},
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=edge_config,
        constant_methods=constant_methods,
    )

    logger.info("Exporting to ExecuTorch...")
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
        )
    )

    _save_program(executorch_program, output_path)


def _export_with_offgraph_cache(
    model_id: str,
    revision: Optional[str],
    output_path: str,
    max_ctx_len: int,
    dtype: str,
    qlinear: Optional[str],
    qembedding: Optional[str],
    no_tie_word_embeddings: bool = False,
    qlinear_group_size: Optional[int] = None,
    qembedding_group_size: Optional[int] = None,
    prefill_chunk_size: Optional[int] = None,
) -> None:
    """Export using the off-graph KV cache op (kvcache::update_and_attend)."""
    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.llm.hf_attention import (
        OffGraphExportWrapper,
        register_mlx_offgraph_attention,
    )
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from transformers import AutoModelForCausalLM

    torch_dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = torch_dtype_map.get(dtype, torch.bfloat16)

    register_mlx_offgraph_attention()
    logger.info("Registered MLX off-graph attention (update_and_attend)")

    logger.info(f"Loading HuggingFace model: {model_id}")
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "mlx_offgraph",
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    from executorch.backends.mlx.llm.quantization import quantize_model_

    quantize_model_(
        model,
        qlinear_config=qlinear,
        qlinear_group_size=qlinear_group_size,
        qembedding_config=qembedding,
        qembedding_group_size=qembedding_group_size,
        tie_word_embeddings=getattr(model.config, "tie_word_embeddings", False)
        and not no_tie_word_embeddings,
    )

    exportable = OffGraphExportWrapper(model)

    from executorch.backends.mlx.llm.cache import resolve_hf_cache_layout

    layer_types, cache_kv_heads, cache_head_dims = resolve_hf_cache_layout(model.config)
    text_config = model.config.get_text_config()
    sliding_window = getattr(text_config, "sliding_window", None) or 0
    cache_windows = [
        sliding_window if t == "sliding_attention" else 0 for t in layer_types
    ]
    sliding = [w for w in cache_windows if w > 0]
    prefill_chunk_size = resolve_prefill_chunk_size(
        prefill_chunk_size, max_ctx_len, min(sliding) if sliding else None
    )

    kv_metadata = {
        "get_n_caches": len(layer_types),
        "get_kv_heads": torch.tensor(cache_kv_heads, dtype=torch.int32),
        "get_head_dims": torch.tensor(cache_head_dims, dtype=torch.int32),
        "get_windows": torch.tensor(cache_windows, dtype=torch.int32),
        "get_prefill_chunk_size": prefill_chunk_size,
        "get_max_ctx_len": max_ctx_len,
    }
    logger.info(
        f"KV cache layout: {len(layer_types)} caches, "
        f"{sum(1 for w in cache_windows if w)} sliding (window {sliding_window})"
    )

    logger.info("Exporting model with torch.export...")
    seq_length = 3
    example_input_ids = torch.zeros((1, seq_length), dtype=torch.long)
    example_cache_position = torch.arange(seq_length, dtype=torch.long)

    seq_len_dim = torch.export.Dim("seq_length_dim", max=prefill_chunk_size)
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

    logger.info("Delegating to MLX backend...")
    edge_program = exir.to_edge_transform_and_lower(
        {"forward": exported_program},
        transform_passes=get_default_passes(),
        constant_methods=kv_metadata,
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
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
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)
    logger.info(f"Saved model to: {output_path}")
    logger.info(f"Program size: {len(executorch_program.buffer) / 1024 / 1024:.2f} MB")


def export_llama_hf(
    model_id: str,
    revision: Optional[str],
    output_path: str,
    max_ctx_len: int = 1024,
    dtype: str = "bf16",
    qlinear: Optional[str] = None,
    qembedding: Optional[str] = None,
    use_custom_sdpa: bool = False,
    use_custom_kv_cache: bool = False,
    use_offgraph_cache: bool = False,
    no_tie_word_embeddings: bool = False,
    qlinear_group_size: Optional[int] = None,
    qembedding_group_size: Optional[int] = None,
    tap_layers: Optional[list[int]] = None,
    prefill_chunk_size: Optional[int] = None,
) -> None:
    if use_offgraph_cache:
        if use_custom_sdpa or use_custom_kv_cache:
            raise ValueError(
                "--use-offgraph-cache is exclusive with --use-custom-sdpa / "
                "--use-custom-kv-cache (it replaces both)"
            )
        logger.info("Using off-graph KV cache (update_and_attend)")
        _export_with_offgraph_cache(
            model_id=model_id,
            revision=revision,
            output_path=output_path,
            max_ctx_len=max_ctx_len,
            dtype=dtype,
            qlinear=qlinear,
            qembedding=qembedding,
            no_tie_word_embeddings=no_tie_word_embeddings,
            qlinear_group_size=qlinear_group_size,
            qembedding_group_size=qembedding_group_size,
            prefill_chunk_size=prefill_chunk_size,
        )
    elif use_custom_sdpa or use_custom_kv_cache or tap_layers is not None:
        logger.info(
            f"Using custom components: sdpa={use_custom_sdpa}, "
            f"kv_cache={use_custom_kv_cache}, tap_layers={tap_layers}, "
            f"prefill_chunk_size={prefill_chunk_size}"
        )
        _export_with_custom_components(
            model_id=model_id,
            revision=revision,
            output_path=output_path,
            max_ctx_len=max_ctx_len,
            dtype=dtype,
            qlinear=qlinear,
            qembedding=qembedding,
            use_custom_sdpa=use_custom_sdpa,
            use_custom_kv_cache=use_custom_kv_cache,
            no_tie_word_embeddings=no_tie_word_embeddings,
            qlinear_group_size=qlinear_group_size,
            qembedding_group_size=qembedding_group_size,
            tap_layers=tap_layers,
            prefill_chunk_size=prefill_chunk_size,
        )
    else:
        logger.info("Using optimum-executorch pipeline (no custom components)")
        if prefill_chunk_size is not None:
            logger.warning(
                "--prefill-chunk-size is ignored on the optimum-executorch "
                "pipeline, which drives its own torch.export call and so owns "
                "the seq_len bound. Pass --use-custom-kv-cache or "
                "--use-offgraph-cache to control it."
            )
        _export_with_optimum(
            model_id=model_id,
            revision=revision,
            output_path=output_path,
            max_ctx_len=max_ctx_len,
            dtype=dtype,
            qlinear=qlinear,
            qembedding=qembedding,
            no_tie_word_embeddings=no_tie_word_embeddings,
            qlinear_group_size=qlinear_group_size,
            qembedding_group_size=qembedding_group_size,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace Llama model to MLX backend"
    )
    parser.add_argument("--model-id", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--max-ctx-len",
        type=int,
        default=1024,
        help="Maximum context length / KV cache capacity",
    )
    parser.add_argument(
        "--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="bf16"
    )
    from executorch.backends.mlx.llm.quantization import add_quantization_args

    add_quantization_args(parser)
    parser.add_argument("--use-custom-sdpa", action="store_true", default=False)
    parser.add_argument("--use-custom-kv-cache", action="store_true", default=False)
    parser.add_argument(
        "--tap-layers",
        type=str,
        default=None,
        help="Comma-separated layer indices whose hidden states are concatenated and returned alongside logits. E.g. '1,9,17,25,33'",
    )
    parser.add_argument("--use-offgraph-cache", action="store_true", default=False)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=512,
        help="Max tokens per forward step. Bounds the traced seq_len dimension "
        "and, for sliding-window models, sizes the ring buffer as window + chunk - 1. "
        "May not exceed the sliding window or the context length.",
    )

    args = parser.parse_args()
    tap_layers = (
        [int(x) for x in args.tap_layers.split(",")] if args.tap_layers else None
    )

    export_llama_hf(
        model_id=args.model_id,
        revision=args.revision,
        output_path=args.output,
        max_ctx_len=args.max_ctx_len,
        dtype=args.dtype,
        qlinear=args.qlinear,
        qembedding=args.qembedding,
        use_custom_sdpa=args.use_custom_sdpa,
        use_custom_kv_cache=args.use_custom_kv_cache,
        use_offgraph_cache=args.use_offgraph_cache,
        no_tie_word_embeddings=args.no_tie_word_embeddings,
        qlinear_group_size=args.qlinear_group_size,
        qembedding_group_size=args.qembedding_group_size,
        tap_layers=tap_layers,
        prefill_chunk_size=args.prefill_chunk_size,
    )


if __name__ == "__main__":
    main()
