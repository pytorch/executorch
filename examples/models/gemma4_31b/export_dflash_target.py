"""Exports Gemma4-31B as a DFlash target model for MLX backend. 

The exported model returns full-sequence logits and hidden states from the configured target layers. These layer IDs must match the draft checkpoint. 
"""

import argparse
import gc
import os

import torch

from executorch.examples.models.gemma4_31b.dflash_export import Gemma4_31BWithHidden
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
    """Loads the prequantized target model with DFlash hidden-state outputs."""
    config = Gemma4_31BConfig.from_hf_config(
        os.path.join(prequantized_dir, "config.json")
    )
    config.max_seq_len = max_seq_len

    print("Building Gemma4_31BWithHidden on meta device:")
    with torch.device("meta"):
        model = Gemma4_31BWithHidden(config, layer_ids=layer_ids)

    safetensors_path = os.path.join(prequantized_dir, "model.safetensors")
    print(f"Loading quantized checkpoint from {safetensors_path}:")
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
    """Exports the DFlash target model through torch.export and the MLX backend."""
    import executorch.backends.mlx.custom_kernel_ops.gguf.patterns
    import executorch.extension.llm.export.gguf  # noqa: F401

    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.examples.models.gemma4_31b.dflash_export import (
        dflash_mlx_source_transformations,
    )
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim, export

    # Upper bound for the drafted block verified in one forward pass.
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

    print(f"Exporting DFlash target (T in [1, {max_verify_len}]): ")
    with torch.no_grad():
        exported = export(
            model,
            example_args,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )

    del model
    gc.collect()

    print("Lowering to ExecuTorch with MLX backend: ")
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
    print(f"Saving to {pte_path}: ")
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
        help="Directory with quantized checkpoint.",
    )
    p.add_argument(
        "--dflash-layers",
        required=True,
        help=(
            "Comma separated 0-indexed layer IDs matching the draft checkpoint, e.g. 1,12,23,35,46,57."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the exported .pte and .ptd files.",
    )
    p.add_argument("--max-seq-len", type=int, default=4096)
    args = p.parse_args()

    layer_ids = [int(x) for x in args.dflash_layers.split(",")]

    model, config = load_prequantized_dflash_target(
        args.prequantized, layer_ids, max_seq_len=args.max_seq_len
    )
    export_dflash_target_mlx(model, config, args.output_dir)


if __name__ == "__main__":
    main()
