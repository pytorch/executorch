"""
Export Qwen 3.5 MoE to ExecuTorch .pte format (CUDA only).

Usage:
  python export.py --model-dir /path/to/Qwen3.5-MoE-A3B
  python export.py --model-dir /path/to/model --qlinear 4w --qlinear-packing-format tile_packed_to_4d
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.export import Dim, export
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass

from executorch.examples.models.qwen3_5_moe.model import Qwen35MoE


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5 MoE to ExecuTorch (CUDA)")
    parser.add_argument("--model-dir", required=True, help="HuggingFace model directory")
    parser.add_argument("--output-dir", default="./qwen35_moe_exports", help="Output directory")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="KV cache length")
    parser.add_argument("--qlinear", default=None, choices=["4w", "8w", "8da4w", "8da8w"],
                        help="Quantize linear layers.")
    parser.add_argument("--qlinear-group-size", type=int, default=32,
                        help="Group size for linear quantization.")
    parser.add_argument("--qlinear-packing-format", default=None,
                        choices=["tile_packed_to_4d"],
                        help="Packing format for 4w quantization (CUDA: tile_packed_to_4d).")
    parser.add_argument("--qembedding", default=None, choices=["8w"],
                        help="Quantize embedding layers.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Register FLA Triton kernel
    import executorch.backends.cuda.triton.kernels  # noqa: F401

    print("Loading model...")
    model, config = Qwen35MoE.from_hf_checkpoint(args.model_dir, max_seq_len=args.max_seq_len)
    model.eval()
    print(f"Model: {config.num_hidden_layers} layers, {config.hidden_size}d, "
          f"{config.num_experts} experts top-{config.num_experts_per_tok}")

    # Quantize layer-by-layer to avoid loading the full bf16 model onto GPU.
    # Each layer is moved to CUDA, quantized (shrinking ~4x for 4w), then
    # stays on CUDA. Peak GPU memory is ~1 layer in bf16 + quantized layers.
    if args.qlinear or args.qembedding:
        from executorch.extension.llm.export.quantize import quantize_model_

        # Untie lm_head/embedding
        if model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr():
            model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())

        for i, layer in enumerate(model.layers):
            layer.to(device="cuda", dtype=torch.bfloat16)
            if args.qlinear:
                quantize_model_(
                    layer,
                    qlinear_config=args.qlinear,
                    qlinear_group_size=args.qlinear_group_size,
                    qlinear_packing_format=args.qlinear_packing_format,
                )
            layer.to(device="cpu")
            torch.cuda.empty_cache()
            print(f"  Quantized layer {i + 1}/{config.num_hidden_layers}", end="\r")
        print()

        # Quantize non-layer modules
        model.embed_tokens.to(device="cuda", dtype=torch.bfloat16)
        model.norm.to(device="cuda", dtype=torch.bfloat16)
        model.lm_head.to(device="cuda", dtype=torch.bfloat16)

        if args.qlinear:
            print("Quantizing lm_head...")
            # Wrap in a container so quantize_() can find the nn.Linear child
            wrapper = nn.ModuleDict({"lm_head": model.lm_head})
            quantize_model_(
                wrapper,
                qlinear_config=args.qlinear,
                qlinear_group_size=args.qlinear_group_size,
                qlinear_packing_format=args.qlinear_packing_format,
            )
            model.lm_head = wrapper.lm_head

        if args.qembedding:
            print(f"Quantizing embeddings ({args.qembedding})...")
            quantize_model_(model, qembedding_config=args.qembedding)

        if args.qlinear:
            print(f"Quantized linear layers ({args.qlinear})")

        # Move entire quantized model to CUDA (~17 GB for 4w)
        model.to(device="cuda")
    else:
        model.to(device="cuda", dtype=torch.bfloat16)

    # Dynamic shapes
    example_tokens = torch.tensor([[0, 1]], dtype=torch.long, device="cuda")
    example_input_pos = torch.tensor([0, 1], dtype=torch.long, device="cuda")
    seq_dim = Dim("seq_len", min=1, max=config.max_seq_len - 1)
    dynamic_shapes = ({1: seq_dim}, {0: seq_dim})

    print("Exporting with torch.export...")
    with torch.no_grad():
        exported = export(
            model,
            (example_tokens, example_input_pos),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
    print("Export successful!")

    # conv1d → conv2d decomposition (required for CUDA backend)
    from torch._inductor.decomposition import conv1d_to_conv2d
    exported = exported.run_decompositions(
        {torch.ops.aten.conv1d.default: conv1d_to_conv2d}
    )

    # Lower with CUDA backend
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

    print("Lowering to ExecuTorch with CUDA...")
    compile_specs = [CudaBackend.generate_method_name_compile_spec("forward")]
    metadata = {
        "get_max_seq_len": config.max_seq_len,
        "get_vocab_size": config.vocab_size,
        "get_n_layers": config.num_hidden_layers,
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": False,
        "enable_dynamic_shape": True,
    }
    et_prog = to_edge_transform_and_lower(
        exported,
        partitioner=[CudaPartitioner(compile_specs)],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True,
        ),
        constant_methods=metadata,
    )
    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )

    # Save .pte
    pte_path = os.path.join(args.output_dir, "model.pte")
    print(f"Saving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)
    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"Saved {size_mb:.1f} MB")

    # Save .ptd tensor data (CUDA backend)
    if et_program._tensor_data:
        et_program.write_tensor_data_to_file(args.output_dir)
        print(f"Saved tensor data to {args.output_dir}/")

    print("Done!")


if __name__ == "__main__":
    main()
