# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a Muse Glimmer target checkpoint to ExecuTorch.

Supports GGUF, consolidated BF16, prequantized, and MLX-affine inputs. Vision
exports add ``vision_encoder`` and write ``pos_embed.bin``.
"""

import argparse
import gc

import torch
import torch.nn as nn
from executorch.examples.models.muse_glimmer.export import common
from executorch.examples.models.muse_glimmer.model.model import (
    materialize_runtime_buffers,
    MuseGlimmerConfig,
    MuseGlimmerModel,
)

# Smallest CUDA chunk that runs the dynamic ``prefill`` method (shorter tails
# fall back to the static ``decode`` method).
_CUDA_MIN_PREFILL_CHUNK = 5


# Input formats share the loading and backend-packing pipeline in
# ``checkpoint_loader``.


def load_prequantized_model(
    prequantized_dir: str,
    max_seq_len: int = 16384,
    backend: str = "cuda",
) -> tuple[MuseGlimmerModel, MuseGlimmerConfig]:
    """Load an atomic quantized checkpoint and pack for the target backend."""
    from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
        load_prequantized_model as _load_prequantized_model,
    )

    return _load_prequantized_model(prequantized_dir, max_seq_len, backend)


def load_and_quantize(
    checkpoint_dir: str,
    recipe_name: str,
    max_seq_len: int = 16384,
    backend: str = "cuda",
) -> tuple[MuseGlimmerModel, MuseGlimmerConfig]:
    """Load bf16 checkpoint, quantize, pack — one shot."""
    from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
        load_and_quantize as _load_and_quantize,
    )
    from executorch.examples.models.muse_glimmer.loaders.quantize_and_save import (
        build_recipes,
    )

    recipe = build_recipes()[recipe_name]
    return _load_and_quantize(checkpoint_dir, recipe, max_seq_len, backend)


# Backend dispatch helpers


_SUPPORTED_BACKENDS = ("cuda", "mlx")


# Export + lower


def export_and_lower(
    model: MuseGlimmerModel,
    config: MuseGlimmerConfig,
    output_dir: str,
    backend: str = "cuda",
    sample: bool = True,
    use_turboquant: bool = False,
    activation_dtype: torch.dtype = torch.bfloat16,
    max_prefill_chunk: int = 512,
    vision_model: nn.Module | None = None,
    pos_embed_table: torch.Tensor | None = None,
    max_vision_patches: int = 16384,
    vision_fp32_mm: str = "none",
) -> None:
    if backend == "cuda":
        _export_cuda(
            model,
            config,
            output_dir,
            sample=sample,
            use_turboquant=use_turboquant,
            vision_model=vision_model,
            pos_embed_table=pos_embed_table,
            max_vision_patches=max_vision_patches,
            vision_fp32_mm=vision_fp32_mm,
        )
    elif backend == "mlx":
        _export_mlx(
            model,
            config,
            output_dir,
            activation_dtype=activation_dtype,
            max_prefill_chunk=max_prefill_chunk,
            vision_model=vision_model,
            pos_embed_table=pos_embed_table,
            max_vision_patches=max_vision_patches,
        )
    else:
        raise ValueError(
            f"Unsupported backend: {backend!r}. Supported: {_SUPPORTED_BACKENDS}."
        )


def _solo_constant_methods(
    *,
    config: MuseGlimmerConfig,
    max_prefill: int,
    activation_dtype: torch.dtype,
    mutable_buffer_metadata: str,
    has_vision: bool,
    max_vision_patches: int,
) -> dict[str, object]:
    """Constant methods baked into the .pte and read back by the runners.

    ``use_kv_cache`` / ``use_sdpa_with_kv_cache`` / ``enable_dynamic_shape`` /
    ``get_vocab_size`` are the standard ExecuTorch LLM metadata keys read by
    ``llm::get_llm_metadata``; the rest are muse_glimmer-specific. CUDA additionally
    sets ``get_min_prefill_chunk``, since only it has a separate static
    ``decode`` method for short prefill tails.
    """
    constant_methods: dict[str, object] = {
        "get_max_seq_len": config.max_seq_len,
        "get_vocab_size": config.vocab_size,
        "get_max_prefill_chunk": max_prefill,
        "get_activation_dtype": common.activation_dtype_tag(activation_dtype),
        "get_mutable_buffer_metadata": mutable_buffer_metadata,
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": False,
        "enable_dynamic_shape": True,
        "has_vision_encoder": bool(has_vision),
    }
    if has_vision:
        constant_methods["get_vision_hidden_size"] = config.dim
        constant_methods["get_max_vision_patches"] = int(max_vision_patches)
    return constant_methods


def _export_cuda(
    model: MuseGlimmerModel,
    config: MuseGlimmerConfig,
    output_dir: str,
    sample: bool = True,
    use_turboquant: bool = False,
    vision_model: nn.Module | None = None,
    pos_embed_table: torch.Tensor | None = None,
    max_vision_patches: int = 16384,
    vision_fp32_mm: str = "none",
) -> None:
    import torch._inductor.config as inductor_config
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
    from torch.export import Dim, export

    inductor_config.coordinate_descent_tuning = False
    inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"
    # Disable precompiled headers: the PCH path shells out to `openssl sha512`
    # and does `stdout.split()[-1]`, which raises IndexError when the
    # preprocessed header is missing/empty in some environments (flaky). PCH is
    # only a compile-time optimization, so turning it off is safe.
    inductor_config.aot_inductor.precompile_headers = False
    if hasattr(inductor_config, "cpp_cache_precompile_headers"):
        inductor_config.cpp_cache_precompile_headers = False

    if vision_fp32_mm == "native":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
    from executorch.examples.models.muse_glimmer.source_transformations.cuda import (
        add_on_device_sampler,
        cuda_source_transformations,
        vision_cuda_source_transformations,
    )

    # Always applied: bounds global-attention SDPA to the valid context via a
    # runtime kv_len (O(context) decode). With use_turboquant=True it also swaps
    # the global KV caches for TurboQuant TQ4.
    cuda_source_transformations(model, use_turboquant=use_turboquant)

    # Max prefill chunk must fit in the ring buffer (2 * sliding_window)
    max_prefill = min(config.max_seq_len - 1, model._sliding_window * 2)

    has_vision = vision_model is not None
    programs: dict[str, "torch.export.ExportedProgram"] = {}

    if sample:
        add_on_device_sampler(model)

    hidden = config.dim
    temp = torch.tensor([1.0], dtype=torch.float32)
    et_seq = Dim("embed_seq_len", min=1, max=max_prefill)
    print(f"Exporting embed_text (T in [1, {max_prefill}])...")
    embed_method = model.embed_text_forward if sample else model.embed_text
    with common.BoundMethodForward(model, embed_method), torch.no_grad():
        programs["embed_text"] = export(
            model,
            (torch.zeros((1, max_prefill), dtype=torch.long),),
            dynamic_shapes=({1: et_seq},),
            strict=True,
        )

    seq_dim = Dim("seq_len", min=5, max=max_prefill)
    print(f"Exporting forward_from_embeddings (T in [5, {max_prefill}], embeds)...")
    forward_method = (
        model.forward_from_embeddings if sample else model.prefill_from_embeds
    )
    forward_args = (
        torch.zeros((1, max_prefill, hidden), dtype=torch.bfloat16),
        torch.arange(max_prefill, dtype=torch.long),
    ) + ((temp,) if sample else ())
    forward_dynamic_shapes = ({1: seq_dim}, {0: seq_dim}) + ((None,) if sample else ())
    with common.BoundMethodForward(model, forward_method), torch.no_grad():
        programs["forward_from_embeddings"] = export(
            model,
            forward_args,
            dynamic_shapes=forward_dynamic_shapes,
            strict=True,
        )

    print("Exporting decode_from_embedding (T=1, embeds)...")
    decode_method = model.decode_from_embedding if sample else model.prefill_from_embeds
    decode_args = (
        torch.zeros((1, 1, hidden), dtype=torch.bfloat16),
        torch.tensor([0], dtype=torch.long),
    ) + ((temp,) if sample else ())
    with common.BoundMethodForward(model, decode_method), torch.no_grad():
        programs["decode_from_embedding"] = export(
            model,
            decode_args,
            strict=True,
        )

    if has_vision:
        if vision_fp32_mm != "none":
            # CUDA-only: applied as a source transform so vision_tower.py stays
            # backend-neutral for the MLX export, which lowers the encoder with
            # no transform at all.
            vision_cuda_source_transformations(
                vision_model,
                start=0,
                end=34,
                use_triton_mm=vision_fp32_mm == "triton",
            )
        programs["vision_encoder"] = common.export_vision_encoder(
            vision_model, pos_embed_table, max_vision_patches
        )

    mutable_buffer_metadata = common.mutable_buffer_metadata(model)
    del model
    if has_vision:
        del vision_model
    gc.collect()
    torch.cuda.empty_cache()

    def _partitioner_for(name: str) -> "CudaPartitioner":
        return CudaPartitioner(
            [
                CudaBackend.generate_method_name_compile_spec(name),
                CompileSpec("low_memory_mode", b"ON"),
            ]
        )

    constant_methods = _solo_constant_methods(
        config=config,
        max_prefill=max_prefill,
        activation_dtype=torch.bfloat16,
        mutable_buffer_metadata=mutable_buffer_metadata,
        has_vision=bool(has_vision),
        max_vision_patches=max_vision_patches,
    )
    constant_methods["get_min_prefill_chunk"] = _CUDA_MIN_PREFILL_CHUNK
    constant_methods["use_sampling"] = sample

    print(
        f"Lowering {len(programs)} methods to ExecuTorch (CUDA): "
        f"{', '.join(programs.keys())}..."
    )
    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner={name: [_partitioner_for(name)] for name in programs},
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods,
    )
    del programs
    gc.collect()
    torch.cuda.empty_cache()

    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
            ),
            emit_mutable_buffer_names=True,
            # Keep method outputs device-resident so the CUDA backend does not
            # insert a per-step boundary D2H copy. The runner reads the sampled
            # token back with its own on-device D2H (solo.cpp read_token).
            propagate_device_config=PropagateDeviceConfig(
                skip_d2h_for_method_outputs=True,
            ),
        ),
    )

    del et_prog
    gc.collect()
    torch.cuda.empty_cache()

    common.save_pte(et_program, output_dir, pos_embed_table if has_vision else None)

    # GPU peak memory marker — emitted on its own line for CI grep.
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"EXPORT_GPU_PEAK_MEMORY_MB: {peak_mb:.1f}")
    print("Done.")


def _export_mlx(
    model: MuseGlimmerModel,
    config: MuseGlimmerConfig,
    output_dir: str,
    activation_dtype: torch.dtype = torch.bfloat16,
    max_prefill_chunk: int = 512,
    vision_model: nn.Module | None = None,
    pos_embed_table: torch.Tensor | None = None,
    max_vision_patches: int = 4096,
) -> None:
    """Export to .pte via torch.export + MLX backend.

    MLX exports ``embed_text`` and dynamic ``forward_from_embeddings``. The
    latter handles both prefill and single-token decode so one delegated method
    owns the KV cache, and returns last-token logits for host-side sampling.

    Vision (``vision_model`` given) is purely additive: it appends a third
    ``vision_encoder`` method (the 9-input host-precompute graph). The runner
    runs ``vision_encoder`` on host-preprocessed patches, splices the image
    soft-tokens into the ``embed_text`` output at ``<|patch|>`` positions, then
    runs ``forward_from_embeddings`` on the spliced embeddings. The vision
    encoder lowers to MLX with no source transform (LayerNorm, bidirectional
    SDPA with a bool mask, GELU, quantized/bf16 Linear and 2D RoPE all have
    direct MLX handlers).

    CUDA additionally exports a static ``decode_from_embedding`` method, with
    on-device sampling by default.
    """
    import gc

    # Register GGUF pattern handlers + custom op so ExportableGGUFTensor weights
    # (Q4_K, Q6_K) lower to the MLX GGUF custom kernels.
    import executorch.backends.mlx.custom_kernel_ops.gguf.patterns  # noqa: F401

    # Register the Int4 export op + MLX pattern handlers so ExportableInt4Tensor
    # weights lower to the MLX quantized matmul. IntxUnpackedToInt8Tensor (int8
    # embedding) already lowers via dequantize_affine -> linear, no import needed.
    import executorch.backends.mlx.patterns  # noqa: F401
    import executorch.extension.llm.export.gguf  # noqa: F401
    import executorch.extension.llm.export.int4  # noqa: F401

    # Register the NVFP4 / MX export ops so block-scaled MLX checkpoints
    # (ExportableNVFP4Tensor / ExportableMXTensor) lower to the MLX
    # mode="nvfp4"/"mxfp8" quantized matmul + gather kernels.
    import executorch.extension.llm.export.mx  # noqa: F401
    import executorch.extension.llm.export.nvfp4  # noqa: F401
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.examples.models.muse_glimmer.source_transformations.mlx import (
        mlx_source_transformations,
    )
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim, export

    has_vision = vision_model is not None

    max_prefill = max_prefill_chunk
    seq_dim = Dim("seq_len", min=1, max=max_prefill)

    # Size sliding-window ring buffers to the prefill chunk (window +
    # max_write_len - 1) rather than 2*window, since the runner chunks prefill
    # to get_max_prefill_chunk == max_prefill.
    mlx_source_transformations(model, dtype=activation_dtype, max_write_len=max_prefill)
    materialize_runtime_buffers(model, dtype=activation_dtype)

    # Convert everything to the activation dtype: unquantized floating params +
    # buffers (norms, RoPE, softcap) AND the quantized-weight scales. GGUF
    # weights are packed with float16 scales; ``model.to`` downcasts them (and
    # everything else) to the target dtype. torchao intx tensors convert their
    # scale/zero_point while preserving the integer qdata. Keeping scales in the
    # activation dtype avoids MLX promoting (e.g. float16 ⊕ bfloat16 → float32)
    # in quantized_matmul, which would otherwise feed float32 into the fused
    # q5_k/q6_k kernels that are templated on the activation dtype.
    model.to(activation_dtype)
    mutable_buffer_metadata = common.mutable_buffer_metadata(model)

    programs: dict[str, "torch.export.ExportedProgram"] = {}

    # Unified MLX contract: embed_text + embeds-input forward_from_embeddings,
    # with vision_encoder added only when a vision tower is present. Only
    # forward_from_embeddings mutates the KV cache (decode reuses it), mirroring
    # the gemma4 MLX vision export.
    hidden = config.dim

    et_seq = Dim("embed_seq_len", min=1, max=max_prefill)
    print(f"Exporting embed_text (T in [1, {max_prefill}])...")
    with common.BoundMethodForward(model, model.mlx_embed_text), torch.no_grad():
        programs["embed_text"] = export(
            model,
            (torch.zeros((1, max_prefill), dtype=torch.long),),
            dynamic_shapes=({1: et_seq},),
            strict=True,
        )

    print(f"Exporting forward_from_embeddings (T in [1, {max_prefill}], embeds)...")
    with common.BoundMethodForward(model, model.mlx_prefill_forward), torch.no_grad():
        programs["forward_from_embeddings"] = export(
            model,
            (
                torch.zeros((1, max_prefill, hidden), dtype=activation_dtype),
                torch.arange(max_prefill, dtype=torch.long),
            ),
            dynamic_shapes=({1: seq_dim}, {0: seq_dim}),
            strict=True,
        )

    if has_vision:
        programs["vision_encoder"] = common.export_vision_encoder(
            vision_model, pos_embed_table, max_vision_patches
        )

    del model
    if has_vision:
        del vision_model
    gc.collect()

    constant_methods = _solo_constant_methods(
        config=config,
        max_prefill=max_prefill,
        activation_dtype=activation_dtype,
        mutable_buffer_metadata=mutable_buffer_metadata,
        has_vision=bool(has_vision),
        max_vision_patches=max_vision_patches,
    )
    constant_methods["use_sampling"] = False

    print(
        f"Lowering {len(programs)} method(s) to ExecuTorch (MLX): "
        f"{', '.join(programs.keys())}..."
    )
    et_prog = to_edge_transform_and_lower(
        programs,
        transform_passes=get_default_passes(),
        partitioner={name: [MLXPartitioner()] for name in programs},
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods,
    )
    del programs
    gc.collect()

    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                share_mutable_buffers=True,
            ),
            emit_mutable_buffer_names=True,
        ),
    )

    del et_prog
    gc.collect()

    common.save_pte(et_program, output_dir, pos_embed_table if has_vision else None)
    print("Done.")


def main() -> None:
    from executorch.examples.models.muse_glimmer.loaders.quantize_and_save import (
        RECIPE_NAMES,
    )

    parser = argparse.ArgumentParser(description="Export Muse Glimmer to ExecuTorch.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Path to consolidated checkpoint. Triggers load + quantize + export.",
    )
    src.add_argument(
        "--prequantized",
        default=None,
        help="Path to a quantized checkpoint directory. Skips quantization.",
    )
    src.add_argument(
        "--gguf",
        default=None,
        help="Path to a GGUF file (e.g. Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf). Streams and "
        "converts weights for the target backend, preserving quant bit-widths.",
    )
    src.add_argument(
        "--mlx",
        default=None,
        help="Path to an MLX affine-quantized safetensors dir (mlx_lm output, "
        "e.g. muse-glimmer-target-mlx-q5-lang). Converts weights to torchao intx "
        "and packs for the selected backend.",
    )
    parser.add_argument(
        "--output-dir",
        default="./muse_glimmer_exports",
        help="Output directory for model.pte / model.ptd.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=131072,
        help="KV cache size.",
    )
    parser.add_argument(
        "--quant-recipe",
        default="default",
        choices=RECIPE_NAMES,
        help="Quantization recipe (only with --checkpoint-dir).",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=list(_SUPPORTED_BACKENDS),
        help="Target backend for export.",
    )
    parser.add_argument(
        "--logits",
        action="store_true",
        help="Export full logits instead of on-device sampling (for NLL "
        "validation). Default samples on-device and returns a token.",
    )
    parser.add_argument(
        "--turboquant",
        action="store_true",
        help="Use TurboQuant TQ4 KV cache on global (NoPE) layers (CUDA).",
    )
    parser.add_argument(
        "--activation-dtype",
        default=None,
        choices=list(common.ACTIVATION_DTYPES),
        help="Activation compute dtype (MLX backend only). Defaults to 'float16' "
        "for --backend mlx and 'bfloat16' otherwise. For 'float16' the KV cache "
        "and all unquantized weights/buffers are converted to fp16 too.",
    )
    parser.add_argument(
        "--max-prefill-chunk",
        type=int,
        default=512,
        help="Max prefill chunk size (MLX backend only): the largest number of "
        "tokens processed per prefill forward, and the dynamic seq_len upper "
        "bound baked into the .pte (get_max_prefill_chunk). Larger values improve "
        "prefill GPU utilization/throughput at the cost of more memory. "
        "Default 512.",
    )
    parser.add_argument(
        "--mmproj",
        default=None,
        help="Optional path to the vision projector GGUF "
        "(e.g. mmproj-Muse-Glimmer-30B-Q4_K_M.gguf). When given (CUDA or "
        "MLX backend), a `vision_encoder` method is exported and the runner can "
        "take image input. Omit to export the text-only (TITO) model.",
    )
    parser.add_argument(
        "--max-vision-patches",
        type=int,
        default=16384,
        help="Max input patches to the vision encoder (upper bound on the "
        "dynamic num_patches dim before 2x2 downsampling). Default 16384 "
        "corresponds to 4096 output image tokens.",
    )
    parser.add_argument(
        "--vision-fp32-mm",
        choices=("triton", "native", "none"),
        default="none",
        help="Optional FP32-output linear implementation for vision blocks "
        "0-34. The default preserves the original all-BF16 encoder.",
    )
    args = parser.parse_args()

    if args.backend == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required for the cuda backend.")

    if args.activation_dtype is None:
        args.activation_dtype = "float16" if args.backend == "mlx" else "bfloat16"
    activation_dtype = common.ACTIVATION_DTYPES[args.activation_dtype]
    if args.backend != "mlx" and activation_dtype != torch.bfloat16:
        parser.error("--activation-dtype is only supported with --backend mlx.")
    if args.backend != "mlx" and args.max_prefill_chunk != 512:
        parser.error("--max-prefill-chunk is only supported with --backend mlx.")
    if args.gguf:
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_gguf_model,
        )

        model, config = load_gguf_model(
            args.gguf,
            max_seq_len=args.max_seq_len,
            backend=args.backend,
            activation_dtype=activation_dtype,
        )
    elif args.mlx:
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_mlx_model,
        )

        model, config = load_mlx_model(
            args.mlx,
            backend=args.backend,
            max_seq_len=args.max_seq_len,
            activation_dtype=activation_dtype,
        )
    elif args.prequantized:
        model, config = load_prequantized_model(
            args.prequantized,
            max_seq_len=args.max_seq_len,
            backend=args.backend,
        )
    else:
        model, config = load_and_quantize(
            args.checkpoint_dir,
            args.quant_recipe,
            max_seq_len=args.max_seq_len,
            backend=args.backend,
        )

    vision_model = None
    pos_embed_table = None
    if args.mmproj:
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_mmproj_vision_model,
        )

        vision_model, pos_embed_table, _ = load_mmproj_vision_model(
            args.mmproj,
            backend=args.backend,
            activation_dtype=activation_dtype,
        )

    export_and_lower(
        model,
        config,
        args.output_dir,
        backend=args.backend,
        sample=not args.logits,
        use_turboquant=args.turboquant,
        activation_dtype=activation_dtype,
        max_prefill_chunk=args.max_prefill_chunk,
        vision_model=vision_model,
        pos_embed_table=pos_embed_table,
        max_vision_patches=args.max_vision_patches,
        vision_fp32_mm=args.vision_fp32_mm,
    )


if __name__ == "__main__":
    main()
