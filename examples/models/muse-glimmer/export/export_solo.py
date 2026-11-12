# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a Muse Glimmer target checkpoint to ExecuTorch.

Supports GGUF, consolidated BF16, prequantized, and MLX-affine inputs. Vision
exports add ``vision_encoder`` and write ``pos_embed.bin``.

``--use-offgraph-kv-cache`` keeps the KV cache out of the program and exports
the batching runtime's contract. On MLX that is the selected-logits decoder. On
CUDA it is the artifact CudaExecutor (backends/cuda/batching) drives: the text
forwards plus, with ``--mmproj``, ``vision_encoder``/``embed_text``/
``forward_from_embeddings``; without the flag CUDA keeps its in-graph prefill
and decode methods:

- ``forward_{N}`` for each N in ``--forward-sizes`` (default 1, 2, 4, ..., 64):
  T = K = N, static, each captured into a CUDA graph at runtime, its quantized
  linears traced through the smallest QuantizedGemmFamily bucket holding N.
- ``forward_others``: T in [max N, ``--max-step-tokens``], dynamic.
- ``sample`` / ``sample_argmax`` (extension/llm/batching/sampler.py), so logits
  stay on the device; ``--logits`` returns them to the host instead.

Every sequence of a batch shares one pool of ``--max-cells`` per-token cells.
The sizes stop at 64, the widest quantized GEMM bucket: above it the quantized
linears take dequant + matmul, which ``forward_others`` runs unpadded.
"""

import argparse
import gc
import json
import types
import warnings

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
    max_seq_len: int | None = 16384,
    backend: str = "cuda",
    *,
    defer_runtime_buffers: bool = False,
) -> tuple[MuseGlimmerModel, MuseGlimmerConfig]:
    """Load an atomic quantized checkpoint and pack for the target backend."""
    from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
        load_prequantized_model as _load_prequantized_model,
    )

    return _load_prequantized_model(
        prequantized_dir,
        max_seq_len,
        backend,
        defer_runtime_buffers=defer_runtime_buffers,
    )


def load_and_quantize(
    checkpoint_dir: str,
    recipe_name: str,
    max_seq_len: int | None = 16384,
    backend: str = "cuda",
    *,
    defer_runtime_buffers: bool = False,
) -> tuple[MuseGlimmerModel, MuseGlimmerConfig]:
    """Load bf16 checkpoint, quantize, pack — one shot."""
    from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
        load_and_quantize as _load_and_quantize,
    )
    from executorch.examples.models.muse_glimmer.loaders.quantize_and_save import (
        build_recipes,
    )

    recipe = build_recipes()[recipe_name]
    return _load_and_quantize(
        checkpoint_dir,
        recipe,
        max_seq_len,
        backend,
        defer_runtime_buffers=defer_runtime_buffers,
    )


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
    use_offgraph_kv_cache: bool = False,
    activation_dtype: torch.dtype = torch.bfloat16,
    max_prefill_chunk: int = 512,
    vision_model: nn.Module | None = None,
    pos_embed_table: torch.Tensor | None = None,
    max_vision_patches: int = 16384,
    vision_fp32_mm: str = "none",
    forward_sizes=None,
    max_step_tokens: int = 4096,
    max_cells: int = 65536,
    kv_layout: str = "cell",
    max_kv_seqs: int = 4,
) -> None:
    if use_offgraph_kv_cache and use_turboquant:
        raise ValueError("off-graph KV cache and TurboQuant are mutually exclusive")
    if backend == "mlx" and use_offgraph_kv_cache:
        if activation_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "off-graph MLX export requires float16 or bfloat16 activations"
            )
        if not 1 <= max_prefill_chunk <= config.max_seq_len:
            raise ValueError("max_prefill_chunk must be within [1, max_seq_len]")
    if backend == "cuda" and use_offgraph_kv_cache:
        if vision_fp32_mm != "none":
            raise ValueError("CUDA batching does not support vision_fp32_mm")
        export_cuda_batching(
            model,
            config,
            output_dir,
            max_step_tokens=max_step_tokens,
            max_cells=max_cells,
            forward_sizes=forward_sizes or DEFAULT_FORWARD_SIZES,
            device_sampling=sample,
            kv_layout=kv_layout,
            max_kv_seqs=max_kv_seqs,
            vision_model=vision_model,
            pos_embed_table=pos_embed_table,
            max_vision_patches=max_vision_patches,
        )
    elif backend == "cuda":
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
            use_offgraph_kv_cache=use_offgraph_kv_cache,
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
    mutable_buffer_metadata: str | None,
    has_vision: bool,
    max_vision_patches: int,
    use_offgraph_kv_cache: bool = False,
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
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": False,
        "enable_dynamic_shape": True,
        "has_vision_encoder": bool(has_vision),
    }
    if use_offgraph_kv_cache:
        from executorch.extension.llm.export.model_metadata import (
            write_activation_dtype,
            write_cache_geometry,
            write_logits_to_keep_mode,
            write_max_context_len,
            write_max_seq_len,
        )

        dtype_names = {torch.float16: "fp16", torch.bfloat16: "bf16"}
        constant_methods.update(write_activation_dtype(dtype_names[activation_dtype]))
        constant_methods.update(write_max_context_len(config.max_seq_len))
        constant_methods.update(write_max_seq_len(max_prefill))
        constant_methods.update(write_logits_to_keep_mode("selected"))
        constant_methods.update(
            write_cache_geometry(
                [config.n_kv_heads] * config.n_layers,
                [config.head_dim] * config.n_layers,
                [config.layer_window_size(i) for i in range(config.n_layers)],
            )
        )
    else:
        constant_methods["get_activation_dtype"] = common.activation_dtype_tag(
            activation_dtype
        )
        if mutable_buffer_metadata is not None:
            constant_methods["get_mutable_buffer_metadata"] = mutable_buffer_metadata
    if has_vision:
        constant_methods["get_vision_hidden_size"] = config.dim
        constant_methods["get_max_vision_patches"] = int(max_vision_patches)
    return constant_methods


# CUDA batching export (--use-offgraph-kv-cache), over a cell-layout KV cache.

# Read by CudaExecutor: the cell pool's size, which fixes its step buffers.
MAX_CELLS_METHOD = "get_offgraph_kv_max_cells"
# Read by CudaExecutor in place of the cell count when each sequence keeps a
# private history: the sequence columns of every layer's storage.
MAX_KV_SEQS_METHOD = "get_offgraph_kv_max_seqs"
# Read by MuseGlimmerCudaDFlashExecutor for the draft's own sequence columns,
# mirroring kMaxSeqs in runtime/cuda_dflash_executor.h.
DRAFT_MAX_KV_SEQS_METHOD = "get_draft_offgraph_kv_max_seqs"
KV_LAYOUTS = ("cell", "sequence")
# Static methods by default: powers of two up to the widest GEMM bucket.
DEFAULT_FORWARD_SIZES = (1, 2, 4, 8, 16, 32, 64)
# The device samplers' widest call: one row per resident session, and the cell
# cache holds at most 64 sequences.
MAX_SAMPLE_ROWS = 64
SAMPLE_METHOD = "sample"
ARGMAX_METHOD = "sample_argmax"


def step_width_spec(method: str) -> bytes:
    """``delegate_input:dim`` of input_pos[T], the step's token count.

    Every forward method's delegate keeps the exported (tokens, input_pos,
    logits_to_keep) input order. The CUDA backend's CheckOffGraphKVStepWidthPass
    fails the export if this is wrong.
    """
    return b"1:0"


# Image prompts reach the batching cache through this method.
EMBEDDINGS_METHOD = "forward_from_embeddings"


def step_forward(
    model: MuseGlimmerModel,
    tokens: torch.Tensor,
    input_pos: torch.Tensor,
    logits_to_keep: torch.Tensor,
) -> torch.Tensor:
    """Logits for the selected tokens of one packed step: [K, vocab] float32.

    The whole step runs through the decoder so every token's K/V reaches the
    cache; only the rows the batch samples from go through the LM head.
    """
    return embeddings_step_forward(
        model, model.embed_text(tokens), input_pos, logits_to_keep
    )


def embeddings_step_forward(
    model: MuseGlimmerModel,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    logits_to_keep: torch.Tensor,
) -> torch.Tensor:
    """``step_forward`` from decoder inputs [1, T, dim], e.g. spliced images."""
    taps = None
    if hasattr(model, "_dflash_target_layer_ids"):
        sliding_mask, global_mask = model._build_masks(input_pos)
        tapped = []
        for i, layer in enumerate(model.layers):
            if i in model._dflash_target_layer_ids:
                tapped.append(x)
            x = layer(x, input_pos, sliding_mask, global_mask)
        taps = torch.cat(tapped, dim=-1)
    else:
        x = model._run_blocks(x, input_pos)
    x = model.output_norm(x[:, logits_to_keep, :])
    logits = model._soft_cap(model.lm_head(x))[0]
    return logits if taps is None else (logits, taps)


def add_kv_layout_arguments(parser) -> None:
    parser.add_argument(
        "--kv-layout",
        choices=KV_LAYOUTS,
        default="cell",
        help="CUDA with --use-offgraph-kv-cache: cell shares one pool of "
        "--max-cells among every sequence; sequence keeps --max-kv-seqs private "
        "histories of up to --max-seq-len each, as MLX batching does.",
    )
    parser.add_argument(
        "--max-kv-seqs",
        type=int,
        default=4,
        help="CUDA with --kv-layout sequence: sequences resident at once.",
    )


def batched_sequence_manifest(
    sequence_manifest: str,
    max_seqs: int,
    causal: bool = True,
    context_window: bool = False,
) -> str:
    """The batched-sequence lowering manifest: one column per sequence."""
    from executorch.backends.cuda.passes.lower_offgraph_kv import (
        batched_sequence_max_segments,
    )

    manifest = json.loads(sequence_manifest)
    manifest["layout"] = "batched-sequence"
    manifest["max_seqs"] = max_seqs
    manifest["max_segments"] = batched_sequence_max_segments(max_seqs)
    manifest["causal"] = causal
    if context_window:
        manifest["context_window"] = True
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def cell_manifest(sequence_manifest: str, max_cells: int) -> str:
    """The cell-layout lowering manifest for the model's off-graph layers."""
    manifest = json.loads(sequence_manifest)
    manifest["layout"] = "cell"
    manifest["max_cells"] = max_cells
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def sampler_programs(vocab_size: int, max_rows: int = MAX_SAMPLE_ROWS):
    """The device samplers, over up to ``max_rows`` float32 logits rows."""
    from executorch.extension.llm.batching.sampler import (
        BatchArgmax,
        BatchSampler,
        NUM_PARAMS,
    )
    from torch.export import Dim, export

    rows = Dim("sample_rows", min=1, max=max_rows)
    logits = torch.zeros(max_rows, vocab_size)
    params = torch.zeros(max_rows, NUM_PARAMS)
    return {
        SAMPLE_METHOD: export(
            BatchSampler(),
            (logits, params),
            dynamic_shapes=({0: rows}, {0: rows}),
            strict=True,
        ),
        ARGMAX_METHOD: export(
            BatchArgmax(), (logits,), dynamic_shapes=({0: rows},), strict=True
        ),
    }


def device_residency(forward_methods, device_sampling: bool, samplers=()):
    """Per-method ``PropagateDeviceConfig`` for ``to_executorch``.

    With device sampling the forwards keep their logits in device memory, and
    the samplers consume them there without a host round trip.
    """
    from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig

    device = {
        name: PropagateDeviceConfig(skip_d2h_for_method_outputs=device_sampling)
        for name in forward_methods
    }
    if device_sampling:
        for name in (SAMPLE_METHOD, ARGMAX_METHOD, *samplers):
            device[name] = PropagateDeviceConfig(skip_h2d_for_method_inputs=True)
    return device


# Keep target and draft assembly under one shared-weight collector.
def export_cuda_batching(  # noqa: C901
    model: MuseGlimmerModel,
    config: MuseGlimmerConfig,
    output_dir: str,
    max_step_tokens: int,
    max_cells: int,
    forward_sizes=DEFAULT_FORWARD_SIZES,
    device_sampling: bool = True,
    *,
    dflash_draft: nn.Module | None = None,
    max_draft_owners: int = 4,
    draft_max_cells: int | None = None,
    vision_model: nn.Module | None = None,
    pos_embed_table: torch.Tensor | None = None,
    max_vision_patches: int = 16384,
    kv_layout: str = "cell",
    max_kv_seqs: int = 4,
):
    """Exports the forward methods and writes the artifact.

    ``kv_layout`` picks the KV cache the program addresses: ``cell`` shares
    one pool of ``max_cells`` cells among every sequence, ``sequence`` keeps
    ``max_kv_seqs`` private histories of up to ``config.max_seq_len`` each.

    With ``vision_model`` the token forwards are unchanged, and image prompts run
    through three more methods: ``vision_encoder``, ``embed_text`` and a dynamic
    ``forward_from_embeddings`` over the same cache and weights.

    Returns the ExecutorchProgramManager so callers can inspect it.
    """
    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
    import torch._inductor.config as inductor_config
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.backends.cuda.passes.lower_offgraph_kv import (
        OFFGRAPH_KV_COMPILE_SPEC,
        OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC,
    )
    from executorch.examples.models.muse_glimmer.source_transformations.cuda import (
        enable_offgraph_kv_cache,
        offgraph_kv_cache_geometry,
    )
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.extension.llm.export.model_metadata import (
        write_logits_to_keep_mode,
        write_max_context_len,
    )
    from torch.export import Dim, export

    if kv_layout not in KV_LAYOUTS:
        raise ValueError(f"kv_layout must be one of {KV_LAYOUTS}, got {kv_layout!r}")
    sequence_layout = kv_layout == "sequence"
    if sequence_layout and max_kv_seqs <= 0:
        raise ValueError("the sequence layout needs max_kv_seqs > 0")
    has_vision = vision_model is not None and pos_embed_table is not None
    if (vision_model is None) != (pos_embed_table is None):
        raise ValueError("vision_model and pos_embed_table must be given together")
    if dflash_draft is not None:
        from executorch.examples.models.muse_glimmer.export.export_dflash import (
            prepare_cuda_packed_draft,
        )

        draft_programs, draft_manifest, draft_metadata = prepare_cuda_packed_draft(
            model,
            config,
            dflash_draft,
            max_step_tokens=max_step_tokens,
            max_cells=max_cells if draft_max_cells is None else draft_max_cells,
            max_owners=max_draft_owners,
            kv_layout=kv_layout,
            max_kv_seqs=max_kv_seqs,
        )

    forward_sizes = tuple(sorted({int(n) for n in forward_sizes}))
    widest_static = forward_sizes[-1] if forward_sizes else 0
    if not forward_sizes or forward_sizes[0] != 1:
        raise ValueError(f"forward_sizes must include 1, got {forward_sizes}")
    # forward_others is dynamic from the widest static size, and its linears
    # trace from five tokens.
    if widest_static < _CUDA_MIN_PREFILL_CHUNK - 1:
        raise ValueError(
            f"the widest forward size must be at least {_CUDA_MIN_PREFILL_CHUNK - 1}, "
            f"got {widest_static}"
        )
    if not (
        widest_static < max_step_tokens
        and (sequence_layout or max_step_tokens < max_cells)
    ):
        bound = config.max_seq_len if sequence_layout else max_cells
        raise ValueError(
            f"max_step_tokens must be in ({widest_static}, {bound}), "
            f"got {max_step_tokens}"
        )
    if max_step_tokens >= config.max_seq_len:
        raise ValueError("max_step_tokens must be below the model's context")

    inductor_config.coordinate_descent_tuning = False
    inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"
    # The PCH path shells out to `openssl sha512` and is flaky; it is only a
    # compile-time optimization.
    inductor_config.aot_inductor.precompile_headers = False
    if hasattr(inductor_config, "cpp_cache_precompile_headers"):
        inductor_config.cpp_cache_precompile_headers = False

    single_manifest = enable_offgraph_kv_cache(model, max_step_tokens)
    manifest = (
        batched_sequence_manifest(single_manifest, max_kv_seqs)
        if sequence_layout
        else cell_manifest(single_manifest, max_cells)
    )
    if dflash_draft is not None:
        # Deferred checkpoints still need scalar/RoPE buffers, after KV removal.
        materialize_runtime_buffers(model, dtype=torch.bfloat16)
    geometry = offgraph_kv_cache_geometry(model)

    # The selected logits feed the device samplers, which take at most
    # MAX_SAMPLE_ROWS rows, so a step never selects more. Bounding the selector
    # there instead of at the step width keeps the memory-planned FP32 logits
    # output at MAX_SAMPLE_ROWS * vocab rather than max_step_tokens * vocab.
    max_selected_rows = min(max_step_tokens, MAX_SAMPLE_ROWS)
    example_rows = _CUDA_MIN_PREFILL_CHUNK if sequence_layout else max_selected_rows
    width = Dim(
        "width", min=max(widest_static, _CUDA_MIN_PREFILL_CHUNK), max=max_step_tokens
    )
    rows = Dim("rows", min=_CUDA_MIN_PREFILL_CHUNK, max=max_selected_rows)
    programs = {}
    with (
        common.BoundMethodForward(model, types.MethodType(step_forward, model)),
        torch.no_grad(),
    ):
        for n in forward_sizes:
            print(f"Exporting forward_{n} (T={n})...")
            programs[f"forward_{n}"] = export(
                model,
                (
                    torch.zeros((1, n), dtype=torch.long),
                    torch.arange(n, dtype=torch.long),
                    torch.zeros(n, dtype=torch.long),
                ),
                strict=True,
            )
        print(
            f"Exporting forward_others (T in [{max(widest_static, _CUDA_MIN_PREFILL_CHUNK)}, "
            f"{max_step_tokens}])..."
        )
        programs["forward_others"] = export(
            model,
            (
                torch.zeros((1, max_step_tokens), dtype=torch.long),
                torch.arange(max_step_tokens, dtype=torch.long),
                torch.zeros(example_rows, dtype=torch.long),
            ),
            dynamic_shapes=({1: width}, {0: width}, {0: rows}),
            strict=True,
        )
    if has_vision:
        embeddings_width = Dim(
            "embeddings_width", min=_CUDA_MIN_PREFILL_CHUNK, max=max_step_tokens
        )
        with (
            common.BoundMethodForward(
                model, types.MethodType(embeddings_step_forward, model)
            ),
            torch.no_grad(),
        ):
            print(
                f"Exporting {EMBEDDINGS_METHOD} (T in [{_CUDA_MIN_PREFILL_CHUNK}, "
                f"{max_step_tokens}])..."
            )
            programs[EMBEDDINGS_METHOD] = export(
                model,
                (
                    torch.zeros((1, max_step_tokens, config.dim), dtype=torch.bfloat16),
                    torch.arange(max_step_tokens, dtype=torch.long),
                    torch.zeros(example_rows, dtype=torch.long),
                ),
                dynamic_shapes=(
                    {1: embeddings_width},
                    {0: embeddings_width},
                    {0: rows},
                ),
                strict=True,
            )
        with common.BoundMethodForward(model, model.embed_text), torch.no_grad():
            print("Exporting embed_text...")
            programs["embed_text"] = export(
                model,
                (torch.zeros((1, max_step_tokens), dtype=torch.long),),
                dynamic_shapes=({1: Dim("embed_width", min=1, max=max_step_tokens)},),
                strict=True,
            )
        programs["vision_encoder"] = common.export_vision_encoder(
            vision_model, pos_embed_table, max_vision_patches
        )
        del vision_model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    forward_methods = [
        name for name in programs if name not in ("embed_text", "vision_encoder")
    ]
    if dflash_draft is not None:
        programs.update(draft_programs)
        del draft_programs, dflash_draft
    draft_methods = [name for name in programs if name.startswith("draft_forward")]
    cache_methods = forward_methods + draft_methods
    verifiers = {}
    if device_sampling:
        print(f"Exporting {SAMPLE_METHOD} and {ARGMAX_METHOD}...")
        programs.update(sampler_programs(config.vocab_size))
        if "draft_forward" in programs:
            from executorch.examples.models.muse_glimmer.export.export_dflash import (
                cuda_owner_verifier_programs,
            )

            verifiers = cuda_owner_verifier_programs(
                config.vocab_size,
                max_draft_owners,
                draft_metadata["get_block_size"],
            )
            print(f"Exporting {', '.join(verifiers)}...")
            programs.update(verifiers)

    def partitioner(name: str) -> CudaPartitioner:
        if name not in cache_methods:
            return CudaPartitioner(
                [CudaBackend.generate_method_name_compile_spec(name)]
            )
        is_draft = name in draft_methods
        specs = [
            CudaBackend.generate_method_name_compile_spec(name),
            CompileSpec("low_memory_mode", b"ON"),
            CompileSpec(
                OFFGRAPH_KV_COMPILE_SPEC,
                (draft_manifest if is_draft else manifest).encode(),
            ),
            CompileSpec(
                OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC,
                b"2:0" if is_draft else step_width_spec(name),
            ),
        ]
        if draft_methods:
            specs.append(
                CompileSpec(
                    "llm_cache_registry_key",
                    b"muse_glimmer_draft" if is_draft else b"muse_glimmer_target",
                )
            )
        return CudaPartitioner(specs)

    constant_methods = _solo_constant_methods(
        config=config,
        max_prefill=max_step_tokens,
        activation_dtype=torch.bfloat16,
        mutable_buffer_metadata=None,
        has_vision=has_vision,
        max_vision_patches=max_vision_patches,
    )
    constant_methods.update(geometry)
    constant_methods.update(write_max_context_len(config.max_seq_len))
    constant_methods.update(write_logits_to_keep_mode("selected"))
    constant_methods["get_min_prefill_chunk"] = _CUDA_MIN_PREFILL_CHUNK
    constant_methods["use_sampling"] = False
    if sequence_layout:
        constant_methods[MAX_KV_SEQS_METHOD] = max_kv_seqs
    else:
        constant_methods[MAX_CELLS_METHOD] = max_cells
    if "draft_forward" in programs:
        constant_methods.update(draft_metadata)

    print(f"Lowering {', '.join(programs)} to ExecuTorch (CUDA)...")
    edge = to_edge_transform_and_lower(
        programs,
        partitioner={name: [partitioner(name)] for name in programs},
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods,
    )
    del programs
    gc.collect()
    torch.cuda.empty_cache()

    et_program = edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            propagate_device_config=device_residency(
                cache_methods, device_sampling, tuple(verifiers)
            ),
            emit_mutable_buffer_names=True,
        ),
    )
    del edge
    gc.collect()

    common.save_pte(et_program, output_dir, pos_embed_table)
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"EXPORT_GPU_PEAK_MEMORY_MB: {peak_mb:.1f}")
    print("Done.")
    return et_program


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

    max_prefill = min(config.max_seq_len - 1, model._sliding_window * 2)
    cuda_source_transformations(model, use_turboquant=use_turboquant)

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
    del model, vision_model
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
    use_offgraph_kv_cache: bool = False,
) -> None:
    """Export to .pte via torch.export + MLX backend.

    MLX exports ``embed_text`` and dynamic ``forward_from_embeddings``. The
    latter handles both prefill and single-token decode so one delegated method
    owns the KV cache, and returns last-token logits for host-side sampling.
    With ``use_offgraph_kv_cache``, the runtime owns the cache and the decoder
    takes a third Long[R] selected-row input, returning Float[1, R, V]. Its
    metadata separates forward width from context and uses integer ScalarType.

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
    # Trace dynamic axes with an example >= 2 to avoid specializing them to 1.
    # min=1 still admits decode; a width-one artifact is deliberately static.
    seq_dim = Dim.STATIC if max_prefill == 1 else Dim("seq_len", min=1, max=max_prefill)

    # Size sliding-window ring buffers to the prefill chunk (window +
    # max_write_len - 1) rather than 2*window, since the runner chunks prefill
    # to get_max_prefill_chunk == max_prefill.
    mlx_source_transformations(
        model,
        dtype=activation_dtype,
        max_write_len=max_prefill,
        use_offgraph_kv_cache=use_offgraph_kv_cache,
    )
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
    mutable_buffer_metadata = (
        None if use_offgraph_kv_cache else common.mutable_buffer_metadata(model)
    )

    programs: dict[str, "torch.export.ExportedProgram"] = {}

    # Unified MLX contract: embed_text + embeds-input forward_from_embeddings,
    # with vision_encoder added only when a vision tower is present. Only
    # forward_from_embeddings mutates the KV cache (decode reuses it), mirroring
    # the gemma4 MLX vision export.
    hidden = config.dim

    et_seq = (
        Dim.STATIC if max_prefill == 1 else Dim("embed_seq_len", min=1, max=max_prefill)
    )
    print(f"Exporting embed_text (T in [1, {max_prefill}])...")
    with common.BoundMethodForward(model, model.mlx_embed_text), torch.no_grad():
        programs["embed_text"] = export(
            model,
            (torch.zeros((1, max_prefill), dtype=torch.long),),
            dynamic_shapes=({1: et_seq},),
            strict=True,
        )

    print(f"Exporting forward_from_embeddings (T in [1, {max_prefill}], embeds)...")
    forward_args = (
        torch.zeros((1, max_prefill, hidden), dtype=activation_dtype),
        torch.arange(max_prefill, dtype=torch.long),
    )
    forward_dynamic_shapes = ({1: seq_dim}, {0: seq_dim})
    if use_offgraph_kv_cache:
        selected_dim = (
            Dim.STATIC
            if max_prefill == 1
            else Dim("selected_len", min=1, max=max_prefill)
        )
        # R is independent of T: callers may select reordered/repeated rows.
        # Both are nonempty and bounded by the advertised forward width.
        forward_args += (torch.arange(max_prefill, dtype=torch.long),)
        forward_dynamic_shapes += ({0: selected_dim},)
    with common.BoundMethodForward(model, model.mlx_prefill_forward), torch.no_grad():
        programs["forward_from_embeddings"] = export(
            model,
            forward_args,
            dynamic_shapes=forward_dynamic_shapes,
            strict=True,
        )

    if has_vision:
        if use_offgraph_kv_cache:
            # API callers can supply a tower loaded in a different dtype.
            # Batching image embeddings must match embed_text and the decoder.
            vision_model.to(activation_dtype)
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
        use_offgraph_kv_cache=use_offgraph_kv_cache,
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


def _validate_offgraph_prefill(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    native_context: int | None = None,
) -> None:
    if args.backend != "mlx" or not args.use_offgraph_kv_cache:
        return
    if args.max_prefill_chunk < 1:
        parser.error("--max-prefill-chunk must be positive.")
    if native_context is not None and args.max_prefill_chunk > native_context:
        parser.error(
            "--max-prefill-chunk must not exceed the checkpoint's native "
            f"context limit ({native_context})."
        )


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
        help="Context/KV cache size for CUDA and legacy MLX export. Ignored for MLX off-graph "
        "export, which uses the checkpoint's native context limit.",
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
        "--use-offgraph-kv-cache",
        action="store_true",
        help="Allocate the KV cache at runtime and export the batching runtime's "
        "contract: MLX's selected-logits decoder, or CUDA's "
        "forward_{N} + forward_others over a cell-layout cache (plus the vision "
        "methods with --mmproj). Without it CUDA "
        "exports its in-graph prefill and decode methods.",
    )
    parser.add_argument(
        "--forward-sizes",
        default=",".join(map(str, DEFAULT_FORWARD_SIZES)),
        help="CUDA with --use-offgraph-kv-cache: comma-separated token counts "
        "of the static forward_{N} methods.",
    )
    parser.add_argument(
        "--max-step-tokens",
        type=int,
        default=4096,
        help="CUDA with --use-offgraph-kv-cache: forward_others' widest step, "
        "the most tokens one forward carries.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=65536,
        help="CUDA with --use-offgraph-kv-cache: cells in the shared KV pool, "
        "the tokens every resident session together may hold. Memory grows with "
        "use, up to this.",
    )
    add_kv_layout_arguments(parser)
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

    _validate_offgraph_prefill(parser, args)

    if args.backend == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required for the cuda backend.")

    if args.activation_dtype is None:
        args.activation_dtype = "float16" if args.backend == "mlx" else "bfloat16"
    activation_dtype = common.ACTIVATION_DTYPES[args.activation_dtype]
    if args.backend != "mlx" and activation_dtype != torch.bfloat16:
        parser.error("--activation-dtype is only supported with --backend mlx.")
    if args.backend != "mlx" and args.max_prefill_chunk != 512:
        parser.error("--max-prefill-chunk is only supported with --backend mlx.")
    if args.use_offgraph_kv_cache and args.turboquant:
        parser.error("--use-offgraph-kv-cache cannot be combined with --turboquant.")
    use_native_context = args.backend == "mlx" and args.use_offgraph_kv_cache
    if use_native_context and args.max_seq_len != parser.get_default("max_seq_len"):
        warnings.warn(
            f"--max-seq-len={args.max_seq_len} is ignored for MLX off-graph export; "
            "using the checkpoint's native context limit.",
            stacklevel=2,
        )
    max_seq_len = None if use_native_context else args.max_seq_len
    loader_kwargs = {"defer_runtime_buffers": True} if use_native_context else {}
    if args.gguf:
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_gguf_model,
        )

        model, config = load_gguf_model(
            args.gguf,
            max_seq_len=max_seq_len,
            backend=args.backend,
            activation_dtype=activation_dtype,
            **loader_kwargs,
        )
    elif args.mlx:
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_mlx_model,
        )

        model, config = load_mlx_model(
            args.mlx,
            backend=args.backend,
            max_seq_len=max_seq_len,
            activation_dtype=activation_dtype,
            **loader_kwargs,
        )
    elif args.prequantized:
        model, config = load_prequantized_model(
            args.prequantized,
            max_seq_len=max_seq_len,
            backend=args.backend,
            **loader_kwargs,
        )
    else:
        model, config = load_and_quantize(
            args.checkpoint_dir,
            args.quant_recipe,
            max_seq_len=max_seq_len,
            backend=args.backend,
            **loader_kwargs,
        )

    _validate_offgraph_prefill(parser, args, config.max_seq_len)

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
        use_offgraph_kv_cache=args.use_offgraph_kv_cache,
        activation_dtype=activation_dtype,
        max_prefill_chunk=args.max_prefill_chunk,
        vision_model=vision_model,
        pos_embed_table=pos_embed_table,
        max_vision_patches=args.max_vision_patches,
        vision_fp32_mm=args.vision_fp32_mm,
        forward_sizes=[int(n) for n in args.forward_sizes.split(",") if n],
        max_step_tokens=args.max_step_tokens,
        max_cells=args.max_cells,
        kv_layout=args.kv_layout,
        max_kv_seqs=args.max_kv_seqs,
    )


if __name__ == "__main__":
    main()
