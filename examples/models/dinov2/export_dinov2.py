# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export DINOv2 image classification model for ExecuTorch with CUDA backend.

Usage:
    python examples/models/dinov2/export_dinov2.py \
        --backend cuda --output-dir ./dinov2_exports

    # With fp32 precision:
    python examples/models/dinov2/export_dinov2.py \
        --backend cuda --dtype fp32 --output-dir ./dinov2_exports

    # For Windows CUDA:
    python examples/models/dinov2/export_dinov2.py \
        --backend cuda-windows --output-dir ./dinov2_exports
"""

import argparse
import os

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes import MemoryPlanningPass
from transformers import Dinov2Config, Dinov2ForImageClassification


def get_model(
    model_name: str = "facebook/dinov2-small-imagenet1k-1-layer",
    random_weights: bool = False,
):
    """Load and return the DINOv2 model in eval mode."""
    if random_weights:
        config = Dinov2Config.from_pretrained(model_name)
        model = Dinov2ForImageClassification(config)
    else:
        model = Dinov2ForImageClassification.from_pretrained(model_name)
    return model.eval()


def export_model(model, sample_input, dtype=None):
    """Export the model using torch.export."""
    if dtype == torch.bfloat16:
        model = model.to(dtype=torch.bfloat16)
        sample_input = (sample_input[0].to(dtype=torch.bfloat16),)

    exported = torch.export.export(model, sample_input, strict=False)
    return exported


def lower_to_executorch(exported_program, backend="cuda", metadata=None):
    """Lower the exported program to ExecuTorch format with CUDA backend."""
    from torch._inductor.decomposition import conv1d_to_conv2d

    exported_program = exported_program.run_decompositions(
        {torch.ops.aten.conv1d.default: conv1d_to_conv2d}
    )

    compile_specs = [CudaBackend.generate_method_name_compile_spec("forward")]
    if backend == "cuda-windows":
        compile_specs.append(
            CompileSpec("platform", "windows".encode("utf-8"))
        )
    partitioner = [CudaPartitioner(compile_specs)]

    constant_methods = {}
    if metadata:
        for key, value in metadata.items():
            constant_methods[key] = value

    programs = {"forward": exported_program}
    partitioner_dict = {"forward": partitioner}

    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner=partitioner_dict,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods if constant_methods else None,
    )

    return et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            do_quant_fusion_and_const_prop=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export DINOv2 model for ExecuTorch CUDA backend"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov2-small-imagenet1k-1-layer",
        help="HuggingFace model name for DINOv2",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Data type for export (default: bf16, required for CUDA Triton SDPA)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dinov2_exports",
        help="Output directory for exported artifacts",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size (default: 224)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cuda",
        choices=["cuda", "cuda-windows"],
        help="Backend to export for (default: cuda)",
    )
    parser.add_argument(
        "--random-weights",
        action="store_true",
        help="Use random weights instead of pretrained (for pipeline testing)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine dtype
    dtype = None
    if args.dtype == "bf16":
        dtype = torch.bfloat16

    print(f"Loading DINOv2 model: {args.model_name}")
    model = get_model(args.model_name, random_weights=args.random_weights)

    # Create sample input
    sample_input = (torch.randn(1, 3, args.img_size, args.img_size),)
    if dtype == torch.bfloat16:
        sample_input = (sample_input[0].to(dtype=torch.bfloat16),)

    print(f"Exporting model with torch.export (dtype={args.dtype or 'fp32'})...")
    exported = export_model(model, sample_input, dtype=dtype)

    # Metadata to embed in the .pte file
    metadata = {
        "get_img_size": args.img_size,
        "get_num_classes": 1000,
    }

    print(f"Lowering to ExecuTorch with {args.backend} backend...")
    et = lower_to_executorch(
        exported, backend=args.backend, metadata=metadata
    )

    # Save the .pte file
    pte_path = os.path.join(args.output_dir, "model.pte")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    print(f"Saved model to {pte_path}")

    # Save tensor data (.ptd)
    if et._tensor_data:
        et.write_tensor_data_to_file(args.output_dir)
        print(f"Saved tensor data to {args.output_dir}/")

    print("Export complete!")


if __name__ == "__main__":
    main()
