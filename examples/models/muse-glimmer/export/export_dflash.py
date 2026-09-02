# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export Muse Glimmer target and DFlash draft models to one ExecuTorch file.

Target and draft inputs may be GGUF or MLX-affine checkpoints. The export
shares token embeddings and the output head; ``--mmproj`` adds vision.
"""

from __future__ import annotations

import argparse
import gc

import torch

from executorch.examples.models.muse_glimmer.export import common


def validate_dflash_export_options(backend: str) -> None:
    if backend not in {"cuda", "mlx"}:
        raise ValueError(f"Unsupported DFlash backend: {backend}")


def _share_graph_mutable_buffers(backend: str) -> bool:
    # CUDA/AOTI lifts state into delegates and shares it by FQN at runtime.
    # MLX keeps mutable state at the graph level and needs the shared arena.
    return backend != "cuda"


def _embed_text_max_len(
    backend: str, target_max_len: int, max_target_prefill: int
) -> int:
    return max_target_prefill if backend == "cuda" else target_max_len


def _max_draft_prefill_len(draft_config, max_target_prefill: int) -> int:
    if (
        draft_config.sliding_window
        and draft_config.sliding_window_pattern is not None
        and all(draft_config.sliding_window_pattern)
    ):
        return min(draft_config.sliding_window, max_target_prefill)
    return max_target_prefill


def export_dflash(
    output_dir: str,
    target_gguf: str | None = None,
    target_mlx: str | None = None,
    draft_gguf: str | None = None,
    draft_mlx: str | None = None,
    max_seq_len: int = 131072,
    activation_dtype: torch.dtype = torch.bfloat16,
    backend: str = "mlx",
    mmproj: str | None = None,
    max_vision_patches: int = 16384,
) -> None:
    """Export DFlash target + draft to one CUDA or MLX .pte.

    The target comes from either a GGUF (``target_gguf``) or an MLX
    affine-quantized safetensors dir (``target_mlx``); the draft likewise from a
    DFlash GGUF (``draft_gguf``) or an MLX affine-quantized dir (``draft_mlx``,
    an mlx_lm.convert output dir). Exactly one of each pair must be given, and
    both must already be quantized. Supplying ``mmproj`` reuses the ordinary Muse Glimmer
    mmproj loader and vision export contract to add ``vision_encoder``.
    """
    validate_dflash_export_options(backend)

    import executorch.extension.llm.export.gguf  # noqa: F401
    import executorch.extension.llm.export.int4  # noqa: F401
    from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
        load_gguf_model,
        load_mlx_model,
    )
    from executorch.examples.models.muse_glimmer.loaders.dflash_loader import (
        load_dflash_gguf,
        load_dflash_mlx,
    )

    # -- Load target model --
    print("=" * 60)
    print("Loading target model...")
    print("=" * 60)
    if target_mlx is not None:
        target_model, target_config = load_mlx_model(
            target_mlx,
            backend=backend,
            max_seq_len=max_seq_len,
            activation_dtype=activation_dtype,
        )
    else:
        target_model, target_config = load_gguf_model(
            target_gguf,
            max_seq_len=max_seq_len,
            backend=backend,
            activation_dtype=activation_dtype,
        )

    # All transformed target entry points cast through this shared contract.
    target_model.activation_dtype = activation_dtype

    # -- Load draft model --
    print("=" * 60)
    print("Loading draft model...")
    print("=" * 60)
    if draft_mlx is not None:
        draft_model, draft_config = load_dflash_mlx(
            draft_mlx,
            backend=backend,
            max_seq_len=max_seq_len,
            activation_dtype=activation_dtype,
        )
    else:
        draft_model, draft_config = load_dflash_gguf(
            draft_gguf,
            backend=backend,
            max_seq_len=max_seq_len,
            activation_dtype=activation_dtype,
        )

    vision_model = None
    pos_embed_table = None
    if mmproj is not None:
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_mmproj_vision_model,
        )

        print("=" * 60)
        print("Loading vision encoder...")
        print("=" * 60)
        vision_model, pos_embed_table, _ = load_mmproj_vision_model(
            mmproj,
            backend=backend,
            activation_dtype=activation_dtype,
        )

    backend_export = _export_dflash_mlx if backend == "mlx" else _export_dflash_cuda
    backend_export(
        target_model,
        target_config,
        draft_model,
        draft_config,
        vision_model,
        pos_embed_table,
        output_dir,
        max_seq_len,
        activation_dtype,
        max_vision_patches,
    )


def _dflash_constant_methods(
    *,
    target_config,
    draft_config,
    max_prefill: int,
    activation_dtype: torch.dtype,
    mutable_buffer_metadata: str,
    exported_block_size: int,
    has_vision: bool,
    max_vision_patches: int,
) -> dict[str, object]:
    """Constant methods baked into the .pte and read back by the runners.

    ``use_kv_cache`` / ``use_sdpa_with_kv_cache`` / ``enable_dynamic_shape`` /
    ``get_vocab_size`` are the standard ExecuTorch LLM metadata keys read by
    ``llm::get_llm_metadata``; the rest are muse_glimmer-specific.
    """
    constant_methods: dict[str, object] = {
        "get_max_seq_len": target_config.max_seq_len,
        "get_vocab_size": target_config.vocab_size,
        "get_max_prefill_chunk": max_prefill,
        "get_activation_dtype": common.activation_dtype_tag(activation_dtype),
        "get_mutable_buffer_metadata": mutable_buffer_metadata,
        "get_block_size": exported_block_size,
        "get_mask_token_id": draft_config.mask_token_id,
        "get_n_target_layers": draft_config.n_target_layers,
        "get_draft_sliding_window": (
            draft_config.sliding_window or 0
            if draft_config.sliding_window_pattern is not None
            and all(draft_config.sliding_window_pattern)
            else 0
        ),
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": False,
        "enable_dynamic_shape": True,
        "has_vision_encoder": has_vision,
    }
    if has_vision:
        constant_methods["get_vision_hidden_size"] = target_config.dim
        constant_methods["get_max_vision_patches"] = int(max_vision_patches)
    return constant_methods


def _export_dflash_mlx(
    target_model,
    target_config,
    draft_model,
    draft_config,
    vision_model,
    pos_embed_table,
    output_dir: str,
    max_seq_len: int,
    activation_dtype: torch.dtype,
    max_vision_patches: int,
) -> None:
    """Export the MLX DFlash contract.

    Always ``embed_text`` + ``target_forward_from_embeddings`` + ``draft_forward``,
    with ``vision_encoder`` appended when a vision tower is present.
    """
    import executorch.backends.mlx.custom_kernel_ops.gguf.patterns  # noqa: F401
    import executorch.backends.mlx.patterns  # noqa: F401

    # Register block-scaled MLX target/draft checkpoint export ops.
    import executorch.extension.llm.export.mx  # noqa: F401
    import executorch.extension.llm.export.nvfp4  # noqa: F401
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.examples.models.muse_glimmer.model.dflash_model import (
        MuseGlimmerWithDFlash,
    )
    from executorch.examples.models.muse_glimmer.model.model import (
        materialize_runtime_buffers,
    )
    from executorch.examples.models.muse_glimmer.source_transformations.mlx import (
        dflash_mlx_source_transformations,
        mlx_source_transformations_dflash_target,
    )
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
    from torch.export import Dim, export

    max_prefill = 512

    print("Applying target MLX source transforms (with hidden tapping)...")
    mlx_source_transformations_dflash_target(
        target_model,
        target_layer_ids=draft_config.target_layers,
        dtype=activation_dtype,
        max_write_len=max_prefill,
    )
    materialize_runtime_buffers(target_model, dtype=activation_dtype)

    print("Applying draft MLX source transforms...")
    draft_model = dflash_mlx_source_transformations(
        draft_model,
        max_context_length=max_seq_len,
        dtype=activation_dtype,
        sliding_window=draft_config.sliding_window,
        sliding_window_pattern=draft_config.sliding_window_pattern,
        max_write_len=max_prefill,
    )

    combined = MuseGlimmerWithDFlash(
        target_model, draft_model, target_config, draft_config
    )

    # Normalize every floating param/buffer AND quantized-weight scale to the
    # activation dtype (mirrors export_solo.py's terminal ``model.to``). Critical
    # for fp16: the MLX draft stores scales as float16 and the target as
    # bfloat16, so without this a float16/bfloat16 mix promotes to float32 in MLX
    # quantized_matmul -- which surfaces as an SDPA dtype mismatch (query in the
    # activation dtype, cached K/V in float32) when exporting draft_forward.
    combined.to(activation_dtype)

    # -- Export the target contract --
    target_seq_dim = Dim("target_seq_len", min=1, max=max_prefill)
    target_tokens = torch.tensor([[0, 1]], dtype=torch.long)
    target_input_pos = torch.arange(target_tokens.size(1), dtype=torch.long)

    print("=" * 60)
    print("Exporting target_forward_from_embeddings...")
    print("=" * 60)
    with common.BoundMethodForward(
        combined, combined.target_forward_from_embeddings
    ), torch.no_grad():
        target_from_embeddings_ep = export(
            combined,
            (
                torch.zeros(
                    (1, target_tokens.size(1), target_config.dim),
                    dtype=target_model.activation_dtype,
                ),
                target_input_pos,
            ),
            dynamic_shapes=({1: target_seq_dim}, {0: target_seq_dim}),
            strict=True,
        )

    embed_seq_dim = Dim("embed_seq_len", min=1, max=max_prefill)
    print("=" * 60)
    print("Exporting embed_text...")
    print("=" * 60)
    with common.BoundMethodForward(combined, combined.embed_text), torch.no_grad():
        embed_text_ep = export(
            combined,
            (target_tokens,),
            dynamic_shapes=({1: embed_seq_dim},),
            strict=True,
        )

    vision_ep = None
    if vision_model is not None:
        print("=" * 60)
        vision_ep = common.export_vision_encoder(
            vision_model, pos_embed_table, max_vision_patches
        )
        print("=" * 60)

    # -- Export draft method --
    print("=" * 60)
    print("Exporting draft_forward...")
    print("=" * 60)
    exported_block_size = draft_config.block_size
    n_target = draft_config.n_target_layers
    dim = draft_config.dim

    # Draft block length is dynamic: a single artifact can run any block length
    # in [2, block_size] at runtime (min 2 avoids 0/1 specialization; max is the
    # native trained block size). target_hidden keeps its dynamic new_ctx_len.
    # Both dims feed the RoPE arange over (ctx_start + new_ctx_len + block_len).
    block_dim = Dim("block_len", min=2, max=exported_block_size)
    new_ctx_dim = Dim("new_ctx_len", min=1, max=max_prefill)

    # input_pos for draft: [ctx_start_pos] — cache write offset
    draft_input_pos = torch.tensor([0], dtype=torch.long)

    with common.BoundMethodForward(combined, combined.draft_forward), torch.no_grad():
        draft_ep = export(
            combined,
            (
                torch.tensor([[0] * exported_block_size], dtype=torch.long),
                torch.randn(1, 2, n_target * dim, dtype=activation_dtype),
                draft_input_pos,
            ),
            dynamic_shapes=({1: block_dim}, {1: new_ctx_dim}, None),
            strict=True,
        )

    mutable_buffer_metadata = common.mutable_buffer_metadata(combined)
    del combined, target_model, draft_model
    if vision_model is not None:
        del vision_model
    gc.collect()

    # -- Lower to ExecuTorch --
    print("=" * 60)
    print("Lowering to ExecuTorch with MLX backend...")
    print("=" * 60)
    methods = {
        "target_forward_from_embeddings": target_from_embeddings_ep,
        "embed_text": embed_text_ep,
        "draft_forward": draft_ep,
    }
    if vision_ep is not None:
        methods["vision_encoder"] = vision_ep
    partitioner = {name: [MLXPartitioner()] for name in methods}

    constant_methods = _dflash_constant_methods(
        target_config=target_config,
        draft_config=draft_config,
        max_prefill=max_prefill,
        activation_dtype=activation_dtype,
        mutable_buffer_metadata=mutable_buffer_metadata,
        exported_block_size=exported_block_size,
        has_vision=vision_ep is not None,
        max_vision_patches=max_vision_patches,
    )

    et_prog = to_edge_transform_and_lower(
        methods,
        transform_passes=get_default_passes(),
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods,
    )

    del target_from_embeddings_ep, embed_text_ep, draft_ep, vision_ep
    gc.collect()

    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=False,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                share_mutable_buffers=_share_graph_mutable_buffers("mlx"),
            ),
            emit_mutable_buffer_names=True,
            propagate_device_config=PropagateDeviceConfig(),
        ),
    )

    del et_prog
    gc.collect()

    common.save_pte(et_program, output_dir, pos_embed_table)
    print(f"Done. Methods: {', '.join(methods)}")


def _export_dflash_cuda(
    target_model,
    target_config,
    draft_model,
    draft_config,
    vision_model,
    pos_embed_table,
    output_dir: str,
    max_seq_len: int,
    activation_dtype: torch.dtype,
    max_vision_patches: int,
) -> None:
    """Export the CUDA DFlash contract.

    Text and vision share the embeddings-based target methods. Vision adds
    ``vision_encoder`` without changing the text contract.
    """
    # Register CUDA quantized tensor linear dispatch before torch.export.
    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
    import torch._inductor.config as inductor_config
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.examples.models.gemma4_31b.cuda_packers import (
        convert_quantized_tensors_for_cuda,
    )
    from executorch.examples.models.muse_glimmer.model.dflash_model import (
        MuseGlimmerWithDFlash,
    )
    from executorch.examples.models.muse_glimmer.source_transformations.cuda import (
        add_dflash_hidden_tapping,
        cuda_source_transformations,
        dflash_cuda_source_transformations,
        materialize_dflash_runtime_buffers,
    )
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
    from torch.export import Dim, export

    max_prefill = 512
    has_vision = vision_model is not None

    print("Applying target CUDA source transforms (with hidden tapping)...")
    cuda_source_transformations(target_model)
    add_dflash_hidden_tapping(target_model, draft_config.target_layers)

    print("Applying draft CUDA source transforms...")
    dflash_cuda_source_transformations(draft_model)

    print("Converting draft quantized tensors for CUDA...")
    convert_quantized_tensors_for_cuda(draft_model)
    materialize_dflash_runtime_buffers(draft_model, dtype=activation_dtype)

    combined = MuseGlimmerWithDFlash(
        target_model, draft_model, target_config, draft_config
    )
    max_target_prefill = min(
        target_config.max_seq_len - 1, target_model._sliding_window * 2
    )

    # See _export_dflash_mlx: normalize params/buffers/scales to one dtype
    # before export so no mixed-precision promotion leaks into the graph.
    combined.to(activation_dtype)

    # -- Export the target contract --
    target_max_len = 4
    target_seq_dim = Dim("target_seq_len", min=1, max=target_max_len)
    target_tokens = torch.arange(target_max_len, dtype=torch.long).unsqueeze(0)
    target_input_pos = torch.arange(target_tokens.size(1), dtype=torch.long)
    print("=" * 60)
    print("Exporting target_forward_from_embeddings...")
    print("=" * 60)
    with common.BoundMethodForward(
        combined, combined.target_forward_from_embeddings
    ), torch.no_grad():
        target_from_embeddings_ep = export(
            combined,
            (
                torch.zeros(
                    (1, target_tokens.size(1), target_config.dim),
                    dtype=target_model.activation_dtype,
                ),
                target_input_pos,
            ),
            dynamic_shapes=({1: target_seq_dim}, {0: target_seq_dim}),
            strict=True,
        )

    # ``embed_text`` serves both the fixed-width verifier path and the
    # variable-width prefill path, so its input needs the same upper bound as
    # ``target_prefill_from_embeddings``.
    embed_max_len = _embed_text_max_len("cuda", target_max_len, max_target_prefill)
    embed_seq_dim = Dim("embed_seq_len", min=1, max=embed_max_len)
    print("=" * 60)
    print("Exporting embed_text...")
    print("=" * 60)
    with common.BoundMethodForward(combined, combined.embed_text), torch.no_grad():
        embed_text_ep = export(
            combined,
            (target_tokens,),
            dynamic_shapes=({1: embed_seq_dim},),
            strict=True,
        )

    vision_ep = None
    if has_vision:
        print("=" * 60)
        vision_ep = common.export_vision_encoder(
            vision_model, pos_embed_table, max_vision_patches
        )
        print("=" * 60)

    # -- Export the prefill contract --
    target_prefill_dim = Dim("target_prefill_seq_len", min=5, max=max_target_prefill)
    print("=" * 60)
    print(
        "Exporting target_prefill_from_embeddings "
        f"(T in [5, {max_target_prefill}])..."
    )
    print("=" * 60)
    with common.BoundMethodForward(
        combined, combined.target_prefill_from_embeddings
    ), torch.no_grad():
        target_prefill_from_embeddings_ep = export(
            combined,
            (
                torch.zeros(
                    (1, max_target_prefill, target_config.dim),
                    dtype=activation_dtype,
                ),
                torch.arange(max_target_prefill, dtype=torch.long),
            ),
            dynamic_shapes=({1: target_prefill_dim}, {0: target_prefill_dim}),
            strict=True,
        )

    # -- Export draft methods --
    print("=" * 60)
    print("Exporting draft_forward...")
    print("=" * 60)
    exported_block_size = min(draft_config.block_size, 4)
    n_target = draft_config.n_target_layers
    dim = draft_config.dim
    new_ctx_max = 4

    block_dim = Dim("block_len", min=2, max=exported_block_size)
    draft_input_pos = torch.tensor([0], dtype=torch.long)

    with common.BoundMethodForward(combined, combined.draft_forward), torch.no_grad():
        # Keep every input byte size fixed for CUDA Graph replay. The
        # valid-length scalar preserves cache/attention semantics when the
        # previous speculative step committed fewer than four positions.
        draft_ep = export(
            combined,
            (
                torch.tensor([[0] * exported_block_size], dtype=torch.long),
                torch.randn(1, new_ctx_max, n_target * dim, dtype=activation_dtype),
                draft_input_pos,
                torch.tensor([2], dtype=torch.long),
            ),
            dynamic_shapes=({1: block_dim}, None, None, None),
            strict=True,
        )

    max_draft_prefill = _max_draft_prefill_len(draft_config, max_target_prefill)
    draft_prefill_dim = Dim("draft_prefill_seq_len", min=5, max=max_draft_prefill)
    print("=" * 60)
    print(f"Exporting draft_prefill (T in [5, {max_draft_prefill}])...")
    print("=" * 60)
    with common.BoundMethodForward(combined, combined.draft_prefill), torch.no_grad():
        draft_prefill_ep = export(
            combined,
            (
                torch.tensor([[0] * exported_block_size], dtype=torch.long),
                torch.randn(
                    1,
                    max_draft_prefill,
                    n_target * dim,
                    dtype=activation_dtype,
                ),
                draft_input_pos,
            ),
            dynamic_shapes=(
                {1: block_dim},
                {1: draft_prefill_dim},
                None,
            ),
            strict=True,
        )

    mutable_buffer_metadata = common.mutable_buffer_metadata(combined)
    del combined, target_model, draft_model
    if vision_model is not None:
        del vision_model
    gc.collect()

    # -- Lower to ExecuTorch --
    print("=" * 60)
    print("Lowering to ExecuTorch with CUDA backend...")
    print("=" * 60)
    inductor_config.coordinate_descent_tuning = False
    inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"
    inductor_config.aot_inductor.precompile_headers = False
    if hasattr(inductor_config, "cpp_cache_precompile_headers"):
        inductor_config.cpp_cache_precompile_headers = False

    def cuda_partitioner(method: str) -> CudaPartitioner:
        return CudaPartitioner(
            [
                CudaBackend.generate_method_name_compile_spec(method),
                CompileSpec("low_memory_mode", b"ON"),
            ]
        )

    methods = {
        "target_forward_from_embeddings": target_from_embeddings_ep,
        "target_prefill_from_embeddings": target_prefill_from_embeddings_ep,
        "embed_text": embed_text_ep,
        "draft_forward": draft_ep,
        "draft_prefill": draft_prefill_ep,
    }
    if vision_ep is not None:
        methods["vision_encoder"] = vision_ep
    partitioner = {name: [cuda_partitioner(name)] for name in methods}

    constant_methods = _dflash_constant_methods(
        target_config=target_config,
        draft_config=draft_config,
        max_prefill=max_prefill,
        activation_dtype=activation_dtype,
        mutable_buffer_metadata=mutable_buffer_metadata,
        exported_block_size=exported_block_size,
        has_vision=has_vision,
        max_vision_patches=max_vision_patches,
    )
    constant_methods.update(
        {
            "get_min_target_prefill_chunk": 5,
            "get_max_target_prefill_chunk": max_target_prefill,
            "get_min_draft_prefill_chunk": 5,
            "get_max_draft_prefill_chunk": max_draft_prefill,
        }
    )

    et_prog = to_edge_transform_and_lower(
        methods,
        transform_passes=None,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods,
    )

    del (
        target_prefill_from_embeddings_ep,
        target_from_embeddings_ep,
        embed_text_ep,
        draft_ep,
        draft_prefill_ep,
        vision_ep,
    )
    gc.collect()

    # The whole target path is device-resident so the DFlash decode loop never
    # round-trips activations through the host: embed_text takes the candidate
    # token ids straight from CUDA and hands its embeddings to the target without
    # a copy. Host callers (prefill, the image splice, the host-sampler baseline)
    # stage their own inputs with clone_tensor_ptr_to and read outputs back with
    # an explicit device-to-host copy.
    device_resident = PropagateDeviceConfig(
        skip_h2d_for_method_inputs=True,
        skip_d2h_for_method_outputs=True,
    )
    propagate_device_config = {
        "embed_text": device_resident,
        "target_forward_from_embeddings": device_resident,
        "target_prefill_from_embeddings": device_resident,
        # draft_forward/draft_prefill still take host tokens and host target
        # hidden state; only their logits stay on device for CUDA sampling.
        "draft_forward": PropagateDeviceConfig(skip_d2h_for_method_outputs=True),
        "draft_prefill": PropagateDeviceConfig(skip_d2h_for_method_outputs=True),
    }

    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                share_mutable_buffers=_share_graph_mutable_buffers("cuda"),
            ),
            emit_mutable_buffer_names=True,
            propagate_device_config=propagate_device_config,
        ),
    )

    del et_prog
    gc.collect()

    common.save_pte(et_program, output_dir, pos_embed_table)
    print(f"Done. Methods: {', '.join(methods)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DFlash (target + draft) to ExecuTorch CUDA or MLX."
    )
    target_src = parser.add_mutually_exclusive_group(required=True)
    target_src.add_argument(
        "--target-gguf",
        default=None,
        help="Path to target GGUF (e.g. Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf).",
    )
    target_src.add_argument(
        "--target-mlx",
        default=None,
        help="Path to target MLX affine-quantized safetensors dir "
        "(e.g. an mlx_lm.convert output dir).",
    )
    draft_src = parser.add_mutually_exclusive_group(required=True)
    draft_src.add_argument(
        "--draft-gguf",
        default=None,
        help="Path to draft DFlash GGUF (e.g. dflash-Muse-Glimmer-30B-Q4_K_M.gguf).",
    )
    draft_src.add_argument(
        "--draft-mlx",
        default=None,
        help="Path to an MLX affine-quantized DFlash draft dir "
        "(e.g. dflash-3l_wind2048-mlx-q5). Already quantized.",
    )
    parser.add_argument(
        "--backend",
        default="mlx",
        choices=["cuda", "mlx"],
        help="ExecuTorch backend. Defaults to MLX.",
    )
    parser.add_argument(
        "--output-dir",
        default="./dflash_exports_mlx",
        help="Output directory for model.pte / model.ptd.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=131072,
        help="KV cache size for target model.",
    )
    parser.add_argument(
        "--mmproj",
        default=None,
        help="Optional Muse Glimmer vision mmproj GGUF; adds vision_encoder.",
    )
    parser.add_argument(
        "--max-vision-patches",
        type=int,
        default=16384,
        help="Maximum patch count accepted by the exported vision encoder.",
    )
    parser.add_argument(
        "--activation-dtype",
        default=None,
        choices=list(common.ACTIVATION_DTYPES),
        help="Activation / KV-cache / unquantized-weight dtype. Defaults to "
        "float16 for MLX and bfloat16 for CUDA.",
    )
    args = parser.parse_args()

    try:
        validate_dflash_export_options(args.backend)
    except ValueError as error:
        parser.error(str(error))
    if args.backend == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required for the cuda backend.")
    if args.backend == "cuda" and (args.target_mlx or args.draft_mlx):
        parser.error("MLX checkpoint directories are only supported by --backend mlx.")
    if args.activation_dtype is None:
        args.activation_dtype = "float16" if args.backend == "mlx" else "bfloat16"
    if args.backend == "cuda" and args.activation_dtype != "bfloat16":
        parser.error("The CUDA DFlash path currently requires bfloat16 activations.")

    activation_dtype = common.ACTIVATION_DTYPES[args.activation_dtype]

    export_dflash(
        target_gguf=args.target_gguf,
        target_mlx=args.target_mlx,
        draft_gguf=args.draft_gguf,
        draft_mlx=args.draft_mlx,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        activation_dtype=activation_dtype,
        backend=args.backend,
        mmproj=args.mmproj,
        max_vision_patches=args.max_vision_patches,
    )


if __name__ == "__main__":
    main()
