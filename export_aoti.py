#!/usr/bin/env python3
"""
Unified export script for AOTI backend.
Usage:
  python export_aoti.py <model_name>              # Uses export_model_to_et_aoti
  python export_aoti.py <model_name> --aoti_only  # Uses export_model_to_pure_aoti

Supported models:
- mv2: MobileNetV2 model
- linear: Simple linear layer model
- conv2d: Single Conv2d layer model
- add: Simple tensor addition model
"""

import argparse
import copy
import os

import shutil

import sys
from subprocess import check_call
from typing import Any, Dict, Tuple

import torch
from executorch.backends.aoti.aoti_partitioner import AotiPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower, to_edge
from torch import nn
from torch.export import export
from torchvision import models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.resnet import ResNet18_Weights


# Model classes
class MV2(torch.nn.Module):
    def __init__(self):
        super(MV2, self).__init__()
        self.mv2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights)

    def forward(self, x: torch.Tensor):
        return self.mv2(x)


class ResNet18(torch.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    def forward(self, x: torch.Tensor):
        return self.resnet18(x)


class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear = nn.Linear(7, 101)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


class SingleConv2d(nn.Module):
    def __init__(self):
        super(SingleConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + y


class DepthwiseConv(nn.Module):
    def __init__(self):
        super().__init__()
        # 32 input channels, 32 output channels, groups=32 for depthwise
        self.conv = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=32,
            bias=False,
        )

    def forward(self, x):
        return self.conv(x)


class BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=16)

    def forward(self, x):
        return self.bn(x)


# Model registry mapping model names to their configurations
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mv2": {
        "model_class": MV2,
        "input_shapes": [(1, 3, 224, 224)],
        "device": "cuda",
        "description": "MobileNetV2 model",
    },
    "resnet18": {
        "model_class": ResNet18,
        "input_shapes": [(1, 3, 224, 224)],
        "device": "cpu",
        "description": "ResNet18 model",
    },
    "linear": {
        "model_class": Linear,
        "input_shapes": [(127, 7)],
        "device": "cuda",
        "description": "Simple linear layer model",
    },
    "conv2d": {
        "model_class": SingleConv2d,
        "input_shapes": [(4, 3, 8, 8)],
        "device": "cuda",
        "description": "Single Conv2d layer model",
    },
    "depthwise_conv": {
        "model_class": DepthwiseConv,
        "input_shapes": [(1, 32, 112, 112)],
        "device": "cuda",
        "description": "Single Depthwise Conv2d layer model",
    },
    "add": {
        "model_class": Add,
        "input_shapes": [(10,), (10,)],
        "device": "cuda",
        "description": "Simple tensor addition model",
    },
    "batchnorm": {
        "model_class": BatchNorm,
        "input_shapes": [(1, 16, 32, 32)],
        "device": "cuda",
        "description": "Single BatchNorm2d layer model",
    },
}


def get_model_and_inputs(
    model_name: str,
) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    """Get model and example inputs based on model name."""

    if model_name not in MODEL_REGISTRY:
        available_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model: {model_name}. Available models: {available_models}"
        )

    model_config = MODEL_REGISTRY[model_name]
    model_class = model_config["model_class"]
    input_shapes = model_config["input_shapes"]
    device = model_config["device"]

    # Create model instance
    model = model_class().to(device).eval()

    # Create example inputs (support multiple inputs)
    example_inputs = tuple(torch.randn(*shape, device=device) for shape in input_shapes)

    return model, example_inputs


def export_model_to_et_aoti(model, example_inputs, output_filename="aoti_model.pte"):
    """Export model through the AOTI pipeline."""
    all_one_input = tuple(
        torch.ones_like(example_input) for example_input in example_inputs
    )

    print("label", model(*all_one_input))

    print(f"Starting export process...")

    # 1. torch.export: Defines the program with the ATen operator set.
    print("Step 1: Converting to ATen dialect...")
    aten_dialect = export(model, example_inputs)

    # print(aten_dialect)
    # exit(0)

    # 2. to_edge: Make optimizations for Edge devices
    # aoti part should be decomposed by the internal torch._inductor.aot_compile
    # we should preserve the lowerable part and waiting for aoti backend handle that
    # Q: maybe need to turn on fallback_random?

    edge_program = to_edge_transform_and_lower(
        aten_dialect, partitioner=[AotiPartitioner([])]
    )

    # edge_program = to_edge(aten_dialect)

    print(edge_program.exported_program())

    # 3. to_executorch: Convert the graph to an ExecuTorch program
    print("Step 4: Converting to ExecuTorch program...")
    executorch_program = edge_program.to_executorch()
    print("To executorch done.")

    # 4. Save the compiled .pte program
    print(f"Step 5: Saving to {output_filename}...")
    with open(output_filename, "wb") as file:
        file.write(executorch_program.buffer)

    print(f"Export completed successfully! Output saved to {output_filename}")


def export_model_to_pure_aoti(model, example_inputs):
    """Export model through the AOTI pipeline."""
    all_one_input = tuple(
        torch.ones_like(example_input) for example_input in example_inputs
    )

    print("label", model(*all_one_input))

    print(f"Starting export process...")

    # 1. torch.export: Defines the program with the ATen operator set.
    print("Step 1: Converting to ATen dialect...")
    aten_dialect = export(model, example_inputs)

    # 2. torch._inductor.aot_compile to aoti delegate
    aten_dialect_module = aten_dialect.module()

    output_path = os.path.join(os.getcwd(), "aoti.so")

    options: dict[str, Any] = {
        "aot_inductor.package_constants_in_so": True,
        "aot_inductor.output_path": output_path,
        "aot_inductor.debug_compile": True,
        "aot_inductor.repro_level": 3,
        "aot_inductor.debug_intermediate_value_printer": "3",
        "max_autotune": True,
        "max_autotune_gemm_backends": "TRITON",
        "max_autotune_conv_backends": "TRITON",
    }

    so_path = torch._inductor.aot_compile(aten_dialect_module, example_inputs, options=options)  # type: ignore[arg-type]

    assert so_path == output_path, f"Expected {output_path} but got {so_path}"

    check_call(
        f"patchelf --remove-needed libtorch.so --remove-needed libc10.so --remove-needed libtorch_cuda.so --remove-needed libc10_cuda.so --remove-needed libtorch_cpu.so --add-needed libcudart.so {output_path}",
        shell=True,
    )


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Unified export script for AOTI backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add model name as positional argument
    parser.add_argument(
        "model_name",
        help="Name of the model to export",
        choices=list(MODEL_REGISTRY.keys()),
        metavar="model_name",
    )

    # Add the --aoti_only flag
    parser.add_argument(
        "--aoti_only",
        action="store_true",
        help="Use export_model_to_pure_aoti instead of export_model_to_et_aoti",
    )

    # Parse arguments
    args = parser.parse_args()

    # Show available models and descriptions in help
    if len(sys.argv) == 1:
        parser.print_help()
        print(f"\nAvailable models: {', '.join(MODEL_REGISTRY.keys())}")
        print("\nModel descriptions:")
        for name, config in MODEL_REGISTRY.items():
            print(f"  {name}: {config['description']}")
        sys.exit(1)

    try:
        model, example_inputs = get_model_and_inputs(args.model_name)

        # Choose export function based on --aoti_only flag
        if args.aoti_only:
            print("Using export_model_to_pure_aoti...")
            export_model_to_pure_aoti(model, example_inputs)
        else:
            print("Using export_model_to_et_aoti...")
            export_model_to_et_aoti(model, example_inputs)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
