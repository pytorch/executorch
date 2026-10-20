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
from typing import TYPE_CHECKING

import torch
from executorch.examples.models.muse_glimmer.export import common

if TYPE_CHECKING:
    from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig


def validate_dflash_export_options(backend: str, use_offgraph_kv_cache=False) -> None:
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


def _export_cuda_sampler_methods(
    max_draft_tokens: int, vocab_size: int
) -> dict[str, torch.export.ExportedProgram]:
    from executorch.examples.models.muse_glimmer.model.dflash_token_sampler import (
        DFlashSpeculativeVerifier,
        DFlashTokenSampler,
    )
    from torch.export import Dim, export

    proposals = Dim("dflash_proposals", min=1, max=max_draft_tokens)
    temperature = torch.tensor([1.0])
    top_k = torch.tensor([0], dtype=torch.long)
    top_p = torch.tensor([1.0])
    return {
        "dflash_sample_tokens": export(
            DFlashTokenSampler(),
            (torch.zeros(max_draft_tokens, vocab_size), temperature, top_k, top_p),
            dynamic_shapes=({0: proposals}, None, None, None),
            strict=True,
        ),
        "dflash_verify_speculative": export(
            DFlashSpeculativeVerifier(),
            (
                torch.zeros(max_draft_tokens + 1, vocab_size),
                torch.zeros(max_draft_tokens, vocab_size),
                torch.zeros(max_draft_tokens + 1, dtype=torch.long),
                temperature,
                top_k,
                top_p,
                torch.tensor([False]),
            ),
            dynamic_shapes=(
                {0: proposals + 1},
                {0: proposals},
                {0: proposals + 1},
                None,
                None,
                None,
                None,
            ),
            strict=True,
        ),
    }


VERIFY_METHOD = "dflash_verify"
VERIFY_ARGMAX_METHOD = "dflash_verify_argmax"


def cuda_owner_verifier_programs(
    vocab_size: int, max_owners: int, block: int
) -> dict[str, torch.export.ExportedProgram]:
    """Batched on-device verification of up to ``max_owners`` draft blocks."""
    from executorch.examples.models.muse_glimmer.model.dflash_token_sampler import (
        DFlashOwnerArgmaxVerifier,
        DFlashOwnerVerifier,
    )
    from torch.export import Dim, export

    owners = (
        Dim("verify_owners", min=1, max=max_owners)
        if max_owners > 1
        else Dim.STATIC
    )
    # A one-proposal width would specialize its size-1 proposal axis.
    width = Dim("verify_width", min=3, max=block)
    logits = torch.zeros(max_owners, block, vocab_size)
    candidates = torch.zeros(max_owners, block, dtype=torch.long)
    params = torch.zeros(max_owners, block + 3)
    return {
        VERIFY_METHOD: export(
            DFlashOwnerVerifier(),
            (logits, candidates, params),
            dynamic_shapes=(
                {0: owners, 1: width},
                {0: owners, 1: width},
                {0: owners, 1: width + 3},
            ),
            strict=True,
        ),
        VERIFY_ARGMAX_METHOD: export(
            DFlashOwnerArgmaxVerifier(),
            (logits, candidates),
            dynamic_shapes=({0: owners, 1: width}, {0: owners, 1: width}),
            strict=True,
        ),
    }


def _cuda_propagate_device_config() -> dict[str, PropagateDeviceConfig]:
    """Keeps the speculative chain's vocabulary-wide tensors on the device."""
    from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig

    device_resident = PropagateDeviceConfig(
        skip_h2d_for_method_inputs=True,
        skip_d2h_for_method_outputs=True,
    )
    draft = PropagateDeviceConfig(skip_d2h_for_method_outputs=True)
    return {
        "embed_text": device_resident,
        "target_forward_from_embeddings": device_resident,
        "target_prefill_from_embeddings": device_resident,
        "draft_forward": draft,
        "draft_prefill": draft,
        "dflash_sample_tokens": device_resident,
        "dflash_verify_speculative": device_resident,
    }


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
    use_offgraph_kv_cache: bool = False,
    max_step_tokens: int = 4096,
    max_cells: int = 65536,
    draft_max_cells: int | None = None,
    max_draft_owners: int = 4,
    forward_sizes=None,
) -> None:
    """Export DFlash target + draft to one CUDA or MLX .pte.

    The target comes from either a GGUF (``target_gguf``) or an MLX
    affine-quantized safetensors dir (``target_mlx``); the draft likewise from a
    DFlash GGUF (``draft_gguf``) or an MLX affine-quantized dir (``draft_mlx``,
    an mlx_lm.convert output dir). Exactly one of each pair must be given, and
    both must already be quantized. Supplying ``mmproj`` reuses the ordinary Muse Glimmer
    mmproj loader and vision export contract to add ``vision_encoder``.
    """
    validate_dflash_export_options(backend, use_offgraph_kv_cache)
    if use_offgraph_kv_cache and backend == "cuda" and mmproj is not None:
        raise ValueError("the CUDA batching DFlash export is text-only")

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
            max_seq_len=None if use_offgraph_kv_cache else max_seq_len,
            activation_dtype=activation_dtype,
            **({"defer_runtime_buffers": True} if use_offgraph_kv_cache else {}),
        )
    else:
        target_model, target_config = load_gguf_model(
            target_gguf,
            max_seq_len=None if use_offgraph_kv_cache else max_seq_len,
            backend=backend,
            activation_dtype=activation_dtype,
            **({"defer_runtime_buffers": True} if use_offgraph_kv_cache else {}),
        )

    # All transformed target entry points cast through this shared contract.
    target_model.activation_dtype = activation_dtype
    if use_offgraph_kv_cache:
        max_seq_len = target_config.max_seq_len

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

    if use_offgraph_kv_cache and backend == "cuda":
        from executorch.examples.models.muse_glimmer.export.export_solo import (
            DEFAULT_FORWARD_SIZES,
            export_cuda_batching,
        )

        export_cuda_batching(
            target_model,
            target_config,
            output_dir,
            max_step_tokens=max_step_tokens,
            max_cells=max_cells,
            forward_sizes=(
                DEFAULT_FORWARD_SIZES if forward_sizes is None else forward_sizes
            ),
            dflash_draft=draft_model,
            max_draft_owners=max_draft_owners,
            draft_max_cells=draft_max_cells,
        )
        return

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
        **({"use_offgraph_kv_cache": True} if use_offgraph_kv_cache else {}),
    )


def prepare_cuda_packed_draft(
    target_model,
    target_config,
    draft_model,
    *,
    max_step_tokens: int,
    max_cells: int,
    max_owners: int,
):
    """Trace the packed draft and publish its independent cell geometry."""
    from executorch.examples.models.muse_glimmer.export.export_solo import cell_manifest
    from executorch.examples.models.muse_glimmer.model.dflash_model import (
        DFlashPackedExportModel,
    )
    from executorch.examples.models.muse_glimmer.source_transformations.cuda import (
        add_dflash_hidden_tapping,
        enable_dflash_offgraph_kv_cache,
        offgraph_kv_cache_geometry,
    )
    from torch.export import Dim, export

    config = draft_model.config
    block = config.block_size
    if (
        type(block) is not int
        or block < 2
        or type(max_owners) is not int
        or not 1 <= max_owners <= 4
    ):
        raise ValueError("packed DFlash requires a trained block >= 2 and 1..4 owners")
    if type(max_step_tokens) is not int or max_step_tokens < 2:
        raise ValueError("packed DFlash max_step_tokens must be an integer >= 2")
    max_write = max_step_tokens + max_owners * block
    if type(max_cells) is not int or max_cells <= max_write:
        raise ValueError("draft max_cells must exceed context plus all owner blocks")
    if config.dim != target_config.dim:
        raise ValueError("draft and target hidden dimensions must agree")
    if target_model.activation_dtype != torch.bfloat16:
        raise ValueError("CUDA packed DFlash requires bfloat16 activations")
    if not 0 <= config.mask_token_id < target_config.vocab_size:
        raise ValueError("draft mask_token_id must be in the target vocabulary")
    add_dflash_hidden_tapping(target_model, list(config.target_layers))
    from executorch.examples.models.gemma4_31b.cuda_packers import (
        convert_quantized_tensors_for_cuda,
    )

    convert_quantized_tensors_for_cuda(draft_model)
    draft_model.to(torch.bfloat16)
    manifest = cell_manifest(
        enable_dflash_offgraph_kv_cache(draft_model, max_write), max_cells
    )
    combined = DFlashPackedExportModel(target_model, draft_model)
    owners = (
        Dim.STATIC if max_owners == 1 else Dim("draft_owners", min=1, max=max_owners)
    )
    context = Dim("draft_new_context", min=0, max=max_step_tokens)
    kv_width = (
        context + block
        if max_owners == 1
        else Dim("draft_kv_width", min=block, max=max_write)
    )
    sample_owners = min(2, max_owners)
    sample_queries = sample_owners * block
    with torch.no_grad():
        program = export(
            combined,
            (
                torch.zeros((sample_owners, block), dtype=torch.long),
                torch.zeros(
                    (1, 2, config.n_target_layers * config.dim), dtype=torch.bfloat16
                ),
                torch.arange(2 + sample_queries, dtype=torch.long),
                torch.arange(2 + sample_queries, dtype=torch.long),
            ),
            dynamic_shapes=(
                {0: owners} if max_owners > 1 else None,
                {1: context},
                {0: kv_width},
                {0: kv_width},
            ),
            strict=True,
        )
    programs = {"draft_forward": program}
    geometry = offgraph_kv_cache_geometry(draft_model)
    metadata = {
        key.replace("get_", "get_draft_", 1): value for key, value in geometry.items()
    }
    metadata.update(
        {
            "get_block_size": block,
            "get_mask_token_id": config.mask_token_id,
            "get_n_target_layers": config.n_target_layers,
            "get_dflash_target_layers": torch.tensor(
                config.target_layers, dtype=torch.int32
            ),
            "get_draft_offgraph_kv_max_cells": max_cells,
            "get_draft_max_step_tokens": max_write,
            "get_draft_max_owners": max_owners,
            "get_draft_dflash_block_size": block,
            "get_dflash_target_cache_key": "muse_glimmer_target",
            "get_dflash_draft_cache_key": "muse_glimmer_draft",
        }
    )
    return programs, manifest, metadata


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
    use_offgraph_kv_cache: bool = False,
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

    max_prefill = min(512, target_config.max_seq_len) if use_offgraph_kv_cache else 512

    print("Applying target MLX source transforms (with hidden tapping)...")
    mlx_source_transformations_dflash_target(
        target_model,
        target_layer_ids=draft_config.target_layers,
        dtype=activation_dtype,
        max_write_len=max_prefill,
        use_offgraph_kv_cache=use_offgraph_kv_cache,
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
        use_offgraph_kv_cache=use_offgraph_kv_cache,
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
    with (
        common.BoundMethodForward(combined, combined.target_forward_from_embeddings),
        torch.no_grad(),
    ):
        target_from_embeddings_ep = export(
            combined,
            (
                torch.zeros(
                    (1, target_tokens.size(1), target_config.dim),
                    dtype=target_model.activation_dtype,
                ),
                target_input_pos,
            )
            + ((target_input_pos.clone(),) if use_offgraph_kv_cache else ()),
            dynamic_shapes=({1: target_seq_dim}, {0: target_seq_dim})
            + (
                ({0: Dim("selected_rows", min=1, max=max_prefill)},)
                if use_offgraph_kv_cache
                else ()
            ),
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
    draft_extra, draft_shapes = (), None
    sample_queries = exported_block_size
    sample_owners, owner_dim = 1, Dim.STATIC
    if use_offgraph_kv_cache:
        if not 2 <= exported_block_size <= max_prefill:
            raise ValueError("DFlash block must fit the target forward bound")
        max_owners = max_prefill // exported_block_size
        owner_dim = Dim("draft_owners", min=1, max=max_owners) if max_owners > 1 else 1
        sample_owners = min(2, max_owners)
        sample_queries *= sample_owners
        new_ctx_dim = Dim("new_ctx_len", min=0, max=max_prefill)
        draft_input_pos = torch.arange(2 + sample_queries, dtype=torch.long)
        draft_extra = (torch.arange(2 + sample_queries, dtype=torch.long),)
        draft_shapes = {0: Dim.AUTO}

    with common.BoundMethodForward(combined, combined.draft_forward), torch.no_grad():
        draft_ep = export(
            combined,
            (
                torch.zeros((sample_owners, exported_block_size), dtype=torch.long),
                torch.randn(1, 2, n_target * dim, dtype=activation_dtype),
                draft_input_pos,
            )
            + draft_extra,
            dynamic_shapes=(
                {0: owner_dim, 1: block_dim},
                {1: new_ctx_dim},
                draft_shapes,
            )
            + ((draft_shapes,) if use_offgraph_kv_cache else ()),
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
        (
            "forward_from_embeddings"
            if use_offgraph_kv_cache
            else "target_forward_from_embeddings"
        ): target_from_embeddings_ep,
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

    if use_offgraph_kv_cache:
        from executorch.examples.models.muse_glimmer.export.export_solo import (
            _solo_constant_methods,
        )
        from executorch.extension.llm.export.model_metadata import write_cache_geometry

        constant_methods.pop("get_mutable_buffer_metadata")
        constant_methods.update(
            _solo_constant_methods(
                config=target_config,
                max_prefill=max_prefill,
                activation_dtype=activation_dtype,
                mutable_buffer_metadata=None,
                has_vision=vision_ep is not None,
                max_vision_patches=max_vision_patches,
                use_offgraph_kv_cache=True,
            )
        )
        draft_geometry = write_cache_geometry(
            [draft_config.n_kv_heads] * draft_config.n_layers,
            [draft_config.head_dim] * draft_config.n_layers,
            [
                draft_config.sliding_window if sliding else 0
                for sliding in (
                    draft_config.sliding_window_pattern
                    or [False] * draft_config.n_layers
                )
            ],
        )
        constant_methods.update(
            {
                key.replace("get_", "get_draft_", 1): value
                for key, value in draft_geometry.items()
            }
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
    exported_block_size = min(draft_config.block_size, 16)
    target_max_len = exported_block_size
    target_seq_dim = Dim("target_seq_len", min=1, max=target_max_len)
    target_tokens = torch.arange(target_max_len, dtype=torch.long).unsqueeze(0)
    target_input_pos = torch.arange(target_tokens.size(1), dtype=torch.long)
    print("=" * 60)
    print("Exporting target_forward_from_embeddings...")
    print("=" * 60)
    with (
        common.BoundMethodForward(combined, combined.target_forward_from_embeddings),
        torch.no_grad(),
    ):
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
    min_prefill = exported_block_size + 1
    target_prefill_dim = Dim(
        "target_prefill_seq_len", min=min_prefill, max=max_target_prefill
    )
    print("=" * 60)
    print(
        f"Exporting target_prefill_from_embeddings (T in [{min_prefill}, {max_target_prefill}])..."
    )
    print("=" * 60)
    with (
        common.BoundMethodForward(combined, combined.target_prefill_from_embeddings),
        torch.no_grad(),
    ):
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
    n_target = draft_config.n_target_layers
    dim = draft_config.dim
    new_ctx_max = exported_block_size

    block_dim = Dim("block_len", min=2, max=exported_block_size)
    draft_input_pos = torch.tensor([0], dtype=torch.long)

    with common.BoundMethodForward(combined, combined.draft_forward), torch.no_grad():
        # Keep every input byte size fixed for CUDA Graph replay. The
        # valid-length scalar preserves cache/attention semantics when the
        # previous speculative step committed fewer positions than the input width.
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
    draft_prefill_dim = Dim(
        "draft_prefill_seq_len", min=min_prefill, max=max_draft_prefill
    )
    print("=" * 60)
    print(f"Exporting draft_prefill (T in [{min_prefill}, {max_draft_prefill}])...")
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

    print("=" * 60)
    print("Exporting DFlash sampling methods...")
    print("=" * 60)
    sampler_methods = _export_cuda_sampler_methods(
        max_draft_tokens=exported_block_size - 1,
        vocab_size=target_config.vocab_size,
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
        **sampler_methods,
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
            "get_min_target_prefill_chunk": min_prefill,
            "get_max_target_prefill_chunk": max_target_prefill,
            "get_min_draft_prefill_chunk": min_prefill,
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
        sampler_methods,
        vision_ep,
    )
    gc.collect()

    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                share_mutable_buffers=_share_graph_mutable_buffers("cuda"),
            ),
            emit_mutable_buffer_names=True,
            propagate_device_config=_cuda_propagate_device_config(),
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
    parser.add_argument("--use-offgraph-kv-cache", action="store_true")
    parser.add_argument("--max-step-tokens", type=int, default=4096)
    parser.add_argument("--max-cells", type=int, default=65536)
    parser.add_argument("--draft-max-cells", type=int, default=None)
    parser.add_argument("--max-draft-owners", type=int, default=4)
    parser.add_argument("--forward-sizes", default="1,2,4,8,16,32,64")
    args = parser.parse_args()

    try:
        validate_dflash_export_options(args.backend, args.use_offgraph_kv_cache)
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
        use_offgraph_kv_cache=args.use_offgraph_kv_cache,
        max_step_tokens=args.max_step_tokens,
        max_cells=args.max_cells,
        draft_max_cells=args.draft_max_cells,
        max_draft_owners=args.max_draft_owners,
        forward_sizes=[int(n) for n in args.forward_sizes.split(",") if n],
    )


if __name__ == "__main__":
    main()
