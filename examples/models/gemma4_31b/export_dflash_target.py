
"""Export Gemma4-31B as a DFlash target model (logits + hidden states) to
the MLX backend.

Mirrors export.py's load_prequantized_model + _export_mlx flow, but builds
Gemma4_31BWithHidden instead of Gemma4_31B, and skips the sample=True /
SamplingHead branch entirely -- DFlash verification needs raw per-position
logits (see dflash_hidden_export.py's forward docstring for why), not an
on-device sampled token.

layer_ids should match the DFlash draft checkpoint's config.json
dflash_config.target_layer_ids exactly (e.g. z-lab/gemma-4-31B-it-DFlash:
[1, 12, 23, 35, 46, 57]) -- a mismatch here would silently condition the
draft on the wrong hidden states with no error, only degraded tau.

Usage:
    python3 export_dflash_target.py \\
        --prequantized ./gemma-4-31B-it-HQQ-INT4 \\
        --dflash-layers 1,12,23,35,46,57 \\
        --output-dir ./gemma4_31b_dflash_exports_mlx \\
        --max-seq-len 4096
"""

import argparse
import gc
import json
import os

import torch

from executorch.examples.models.gemma4_31b.dflash_hidden_export import (
    Gemma4_31BWithHidden,
)
from executorch.examples.models.gemma4_31b.export import _pack_for_backend
from executorch.examples.models.gemma4_31b.model import (
    Gemma4_31BConfig,
    materialize_runtime_buffers,
)


def load_prequantized_dflash_target(
    prequantized_dir: str,
    layer_ids: list,
    max_seq_len: int = 4096,
) -> tuple:
    """Load a quantized checkpoint into Gemma4_31BWithHidden, packed for MLX.

    Same as export.py's load_prequantized_model, except the meta-device
    model is Gemma4_31BWithHidden (adds no new parameters/buffers beyond
    Gemma4_31B, so _pack_for_backend's generic by-FQN packing is unaffected).
    """
    config = Gemma4_31BConfig.from_hf_config(
        os.path.join(prequantized_dir, "config.json")
    )
    config.max_seq_len = max_seq_len

    print("Building Gemma4_31BWithHidden on meta device...")
    with torch.device("meta"):
        model = Gemma4_31BWithHidden(config, layer_ids=layer_ids)

    safetensors_path = os.path.join(prequantized_dir, "model.safetensors")
    print(f"Loading quantized checkpoint from {safetensors_path}...")
    _pack_for_backend(model, safetensors_path, "mlx")
    model.eval()

    print(
        f"Model: {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
        f"dflash_layer_ids={layer_ids}"
    )
    return model, config


def export_dflash_target_mlx(
    model: "Gemma4_31BWithHidden",
    config: Gemma4_31BConfig,
    output_dir: str,
) -> None:
    """Export the DFlash target (logits, hidden) via torch.export + MLX backend.

    Adapted from export.py's _export_mlx: same source transforms and
    lowering pipeline, but no sample=True / SamplingHead branch (not
    applicable -- see load call site), and example_args/dynamic_shapes
    reflect forward(tokens, input_pos) -> (logits, hidden) instead of
    forward(tokens, input_pos, temperature) -> sampled_token.
    """
    import executorch.backends.mlx.custom_kernel_ops.gguf.patterns  # noqa: F401
    import executorch.extension.llm.export.gguf  # noqa: F401
    import executorch.extension.llm.export.int4  # noqa: F401

    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.examples.models.gemma4_31b.dflash_mlx_source_transformations import (
        dflash_mlx_source_transformations,
    )
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim, export

    # block_size (max verified block length) governs the upper bound here,
    # not max_prefill in the general sense -- DFlash target verification
    # forward passes are always <= block_size tokens. 256 matches export.py's
    # ceiling and comfortably covers any realistic block_size (e.g. 16).
    max_verify_len = 256

    dflash_mlx_source_transformations(
        model,
        dtype=torch.bfloat16,
        use_turboquant=False,
        max_write_len=max_verify_len,
    )

    materialize_runtime_buffers(model, dtype=torch.bfloat16)

    seq_dim = Dim("seq_len", min=1, max=max_verify_len)
    example_tokens = torch.tensor([[0, 1]], dtype=torch.long)
    example_input_pos = torch.tensor([0, 1], dtype=torch.long)
    example_args = (example_tokens, example_input_pos)
    dynamic_shapes = ({1: seq_dim}, {0: seq_dim})

    print(f"Exporting DFlash target (T in [1, {max_verify_len}])...")
    with torch.no_grad():
        exported = export(
            model,
            example_args,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )

    del model
    gc.collect()

    print("Lowering to ExecuTorch with MLX backend...")
    et_prog = to_edge_transform_and_lower(
        exported,
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods={
            "get_max_seq_len": config.max_seq_len,
            "get_vocab_size": config.vocab_size,
            "get_n_layers": config.num_hidden_layers,
            "get_max_prefill_chunk": max_verify_len,
            "use_kv_cache": True,
            "use_sdpa_with_kv_cache": False,
            "enable_dynamic_shape": True,
            "use_sampling": False,
        },
    )

    del exported
    gc.collect()
    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )

    del et_prog
    gc.collect()

    os.makedirs(output_dir, exist_ok=True)
    pte_path = os.path.join(output_dir, "model.pte")
    print(f"Saving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)
    print(f"  {os.path.getsize(pte_path) / 1024**2:.1f} MB")
    if et_program._tensor_data:
        et_program.write_tensor_data_to_file(output_dir)
        print(f"  Saved tensor data (.ptd) to {output_dir}/")
    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prequantized",
        required=True,
        help="Directory with quantized checkpoint (model.safetensors + config.json).",
    )
    p.add_argument(
        "--dflash-layers",
        required=True,
        help=(
            "Comma-separated 0-indexed target layer ids to tap for hidden "
            "states, matching the DFlash draft checkpoint's config.json "
            "dflash_config.target_layer_ids exactly (e.g. "
            "'1,12,23,35,46,57' for z-lab/gemma-4-31B-it-DFlash)."
        ),
    )
    p.add_argument("--output-dir", required=True, help="Output dir for model.pte/.ptd.")
    p.add_argument("--max-seq-len", type=int, default=4096)
    args = p.parse_args()

    layer_ids = [int(x) for x in args.dflash_layers.split(",")]

    model, config = load_prequantized_dflash_target(
        args.prequantized, layer_ids, max_seq_len=args.max_seq_len
    )
    export_dflash_target_mlx(model, config, args.output_dir)


if __name__ == "__main__":
    main()
