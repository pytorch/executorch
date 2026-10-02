# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a Muse Glimmer text model for batched serving on CUDA.

The artifact is what ``CudaExecutor`` (backends/cuda/batching) drives: two
methods over one set of weights, both taking ``(tokens[1, T], input_pos[T],
logits_to_keep[K])`` and returning float32 ``logits[K, vocab]`` for the
selected rows.

- ``decode``: T = K = 1, static. The runtime captures it into a CUDA graph.
- ``prefill``: T and K in [5, W], dynamic. Narrower slices run as decodes.

The KV cache lives off-graph in the cell layout: every sequence of a batch
shares one pool of ``max_cells`` per-token cells, and the runtime cache writes
each step's placement and visibility before the forward. Sampling happens on
the host, per session.

    python -m executorch.examples.models.muse_glimmer.export.export_solo_batching \\
        --gguf Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf --output-dir out/
"""

import argparse
import gc
import json
import types

import torch
from executorch.examples.models.muse_glimmer.export import common
from executorch.examples.models.muse_glimmer.export.export_solo import (
    _solo_constant_methods,
    load_and_quantize,
    load_prequantized_model,
)
from executorch.examples.models.muse_glimmer.model.model import (
    MuseGlimmerConfig,
    MuseGlimmerModel,
)

# Read by CudaExecutor: the cell pool's size, which fixes its step buffers.
MAX_CELLS_METHOD = "get_offgraph_kv_max_cells"
# The fewest tokens, and selected rows, prefill is exported for; CudaExecutor
# reads it from get_min_prefill_chunk and runs narrower slices as decodes. The
# quantized linears switch kernels at M <= 4 (backends/cuda/quantize_op_dispatch),
# so a dynamic width only traces from 5 -- as in the single-sequence export.
MIN_PREFILL_TOKENS = 5


def step_width_spec(method: str) -> bytes:
    """``delegate_input:dim`` of input_pos[T], the step's token count.

    It names the delegate's own input order, which partitioning decides: the
    static method takes (tokens, input_pos, logits_to_keep), while the dynamic
    one reads logits_to_keep's symbolic size first and takes (tokens,
    logits_to_keep, input_pos). The CUDA backend's
    CheckOffGraphKVStepWidthPass fails the export if this is wrong.
    """
    return b"2:0" if method == "prefill" else b"1:0"


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
    x = model._run_blocks(model.embed_text(tokens), input_pos)
    x = model.output_norm(x[:, logits_to_keep, :])
    return model._soft_cap(model.lm_head(x))[0]


def cell_manifest(sequence_manifest: str, max_cells: int) -> str:
    """The cell-layout lowering manifest for the model's off-graph layers."""
    manifest = json.loads(sequence_manifest)
    manifest["layout"] = "cell"
    manifest["max_cells"] = max_cells
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def export_batching(
    model: MuseGlimmerModel,
    config: MuseGlimmerConfig,
    output_dir: str,
    max_step_tokens: int,
    max_cells: int,
):
    """Exports ``decode`` and ``prefill`` and writes the artifact.

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

    if not MIN_PREFILL_TOKENS <= max_step_tokens <= max_cells:
        raise ValueError(
            f"max_step_tokens must be in [{MIN_PREFILL_TOKENS}, max_cells], "
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

    manifest = cell_manifest(enable_offgraph_kv_cache(model, max_step_tokens), max_cells)
    geometry = offgraph_kv_cache_geometry(model)

    width = Dim("width", min=MIN_PREFILL_TOKENS, max=max_step_tokens)
    rows = Dim("rows", min=MIN_PREFILL_TOKENS, max=max_step_tokens)
    programs = {}
    with common.BoundMethodForward(
        model, types.MethodType(step_forward, model)
    ), torch.no_grad():
        print("Exporting decode (T=1)...")
        programs["decode"] = export(
            model,
            (
                torch.zeros((1, 1), dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            ),
            strict=True,
        )
        print(f"Exporting prefill (T in [{MIN_PREFILL_TOKENS}, {max_step_tokens}])...")
        programs["prefill"] = export(
            model,
            (
                torch.zeros((1, max_step_tokens), dtype=torch.long),
                torch.arange(max_step_tokens, dtype=torch.long),
                torch.zeros(max_step_tokens, dtype=torch.long),
            ),
            dynamic_shapes=({1: width}, {0: width}, {0: rows}),
            strict=True,
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    def partitioner(name: str) -> CudaPartitioner:
        return CudaPartitioner(
            [
                CudaBackend.generate_method_name_compile_spec(name),
                CompileSpec("low_memory_mode", b"ON"),
                CompileSpec(OFFGRAPH_KV_COMPILE_SPEC, manifest.encode()),
                CompileSpec(OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC, step_width_spec(name)),
                CompileSpec("autotune_at_compile_time", b"OFF"),
            ]
        )

    constant_methods = _solo_constant_methods(
        config=config,
        max_prefill=max_step_tokens,
        activation_dtype=torch.bfloat16,
        mutable_buffer_metadata=None,
        has_vision=False,
        max_vision_patches=0,
    )
    constant_methods.update(geometry)
    constant_methods.update(write_max_context_len(config.max_seq_len))
    constant_methods.update(write_logits_to_keep_mode("selected"))
    constant_methods["get_min_prefill_chunk"] = MIN_PREFILL_TOKENS
    constant_methods["use_sampling"] = False
    constant_methods[MAX_CELLS_METHOD] = max_cells

    print("Lowering decode and prefill to ExecuTorch (CUDA)...")
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

    # Logits come back to the host: each session samples its own rows there.
    et_program = edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            emit_mutable_buffer_names=True,
        ),
    )
    del edge
    gc.collect()

    common.save_pte(et_program, output_dir, None)
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"EXPORT_GPU_PEAK_MEMORY_MB: {peak_mb:.1f}")
    print("Done.")
    return et_program


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Muse Glimmer for batched serving on CUDA."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf", default=None, help="Path to a GGUF checkpoint.")
    src.add_argument(
        "--prequantized", default=None, help="Path to a quantized checkpoint dir."
    )
    src.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Path to a consolidated bf16 checkpoint; quantized with --quant-recipe.",
    )
    parser.add_argument("--quant-recipe", default="default")
    parser.add_argument("--output-dir", default="./muse_glimmer_batching")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=131072,
        help="Model context: the most tokens one session may hold.",
    )
    parser.add_argument(
        "--max-step-tokens",
        type=int,
        default=512,
        help="prefill's widest step, W: the most tokens one forward carries.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=65536,
        help="Cells in the shared KV pool: the tokens every resident session "
        "together may hold. Memory grows with use, up to this.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required.")

    if args.gguf:
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_gguf_model,
        )

        model, config = load_gguf_model(
            args.gguf,
            max_seq_len=args.max_seq_len,
            backend="cuda",
            activation_dtype=torch.bfloat16,
        )
    elif args.prequantized:
        model, config = load_prequantized_model(
            args.prequantized, max_seq_len=args.max_seq_len, backend="cuda"
        )
    else:
        model, config = load_and_quantize(
            args.checkpoint_dir,
            args.quant_recipe,
            max_seq_len=args.max_seq_len,
            backend="cuda",
        )

    export_batching(
        model,
        config,
        args.output_dir,
        max_step_tokens=args.max_step_tokens,
        max_cells=args.max_cells,
    )


if __name__ == "__main__":
    main()
