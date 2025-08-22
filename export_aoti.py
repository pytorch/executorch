#!/usr/bin/env python3
"""
Unified export script for AOTI backend.
Usage: python export_aoti.py <model_name>

Supported models:
- mv2: MobileNetV2 model
- linear: Simple linear layer model
- conv2d: Single Conv2d layer model
- add: Simple tensor addition model
"""

import copy
import os

import shutil

import sys
from subprocess import check_call
from typing import Any, Dict, Tuple

import torch
from executorch.backends.aoti.aoti_partitioner import AotiPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch import nn
from torch.export import export
from torchvision import models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


# Model classes
class MV2(torch.nn.Module):
    def __init__(self):
        super(MV2, self).__init__()
        self.mv2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights)

    def forward(self, x: torch.Tensor):
        return self.mv2(x)


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


# Model registry mapping model names to their configurations
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mv2": {
        "model_class": MV2,
        "input_shapes": [(1, 3, 224, 224)],
        "device": "cuda",
        "description": "MobileNetV2 model",
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
    "add": {
        "model_class": Add,
        "input_shapes": [(10,), (10,)],
        "device": "cuda",
        "description": "Simple tensor addition model",
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


def export_model(model, example_inputs, output_filename="aoti_model.pte"):
    """Export model through the AOTI pipeline."""
    all_one_input = tuple(
        torch.ones_like(example_input) for example_input in example_inputs
    )

    print("label", model(*all_one_input))

    print(f"Starting export process...")

    # 1. torch.export: Defines the program with the ATen operator set.
    print("Step 1: Converting to ATen dialect...")
    aten_dialect = export(model, example_inputs)

    # 2. to_edge: Make optimizations for Edge devices
    # print("Step 2: Converting to Edge program...")
    # edge_program = to_edge(aten_dialect)
    # print(edge_program.exported_program().graph.print_tabular())

    # print("Step 3: Converting to backend...")
    # edge_program = edge_program.to_backend(AotiPartitioner([]))
    # print("To backend done.")

    # aoti part should be decomposed by the internal torch._inductor.aot_compile
    # we should preserve the lowerable part and waiting for aoti backend handle that
    # Q: maybe need to turn on fallback_random?
    edge_program = to_edge_transform_and_lower(
        aten_dialect, partitioner=[AotiPartitioner([])]
    )

    # 3. to_executorch: Convert the graph to an ExecuTorch program
    print("Step 4: Converting to ExecuTorch program...")
    executorch_program = edge_program.to_executorch()
    print("To executorch done.")

    # 4. Save the compiled .pte program
    print(f"Step 5: Saving to {output_filename}...")
    with open(output_filename, "wb") as file:
        file.write(executorch_program.buffer)

    print(f"Export completed successfully! Output saved to {output_filename}")


def main():
    if len(sys.argv) != 2:
        available_models = ", ".join(MODEL_REGISTRY.keys())
        print("Usage: python export_aoti.py <model_name>")
        print(f"Available models: {available_models}")
        print("\nModel descriptions:")
        for name, config in MODEL_REGISTRY.items():
            print(f"  {name}: {config['description']}")
        sys.exit(1)

    model_name = sys.argv[1]

    try:
        model, example_inputs = get_model_and_inputs(model_name)
        export_model(model, example_inputs)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
