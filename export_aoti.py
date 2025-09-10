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

# from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge, to_edge_transform_and_lower
from torch import nn
from torch.export import export
from torch.nn.attention import SDPBackend
from torchvision import models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.resnet import ResNet18_Weights
from transformers import AutoModelForCausalLM, WhisperModel


# for maintaing precision of 32-bit float as much as possible
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.conv.fp32_precision = "fp32"


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


class SingleResNetBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection - identity mapping if same channels, 1x1 conv if different
        self.skip_connection = None
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.skip_connection is not None:
            identity = self.skip_connection(x)

        out += identity
        out = self.relu(out)

        return out


class Llama31(torch.nn.Module):
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B", use_cache=False):
        super(Llama31, self).__init__()
        # Load Llama 3.1 model from HF
        self.use_cache = use_cache
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cuda",
            use_cache=self.use_cache,  # Turn off KV cache
        )
        self.model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # Disable KV cache for inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=self.use_cache,  # Explicitly turn off KV cache
            )
        return outputs.logits


class Whisper(torch.nn.Module):
    def __init__(self, model_name="openai/whisper-tiny"):
        super(Whisper, self).__init__()
        # 1. Load pre-trained Whisper model (tiny version is lightweight)
        self.model = WhisperModel.from_pretrained(model_name)
        self.model.eval()

    def forward(self, input_features: torch.Tensor):
        outputs = self.model.encoder(input_features=input_features)

        # Return both encoder and decoder hidden states for compatibility
        return outputs.last_hidden_state


class MockConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=80,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
        )

    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention block with residual connection
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward block with residual connection
        ff_output = self.ffn(x)
        x = self.norm2(x + ff_output)

        return x


# Model registry mapping model names to their configurations
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mv2": {
        "model_class": MV2,
        "input_shapes": [(1, 3, 224, 224)],
        "description": "MobileNetV2 model",
    },
    "resnet18": {
        "model_class": ResNet18,
        "input_shapes": [(1, 3, 224, 224)],
        "description": "ResNet18 model",
    },
    "linear": {
        "model_class": Linear,
        "input_shapes": [(127, 7)],
        "description": "Simple linear layer model",
    },
    "conv2d": {
        "model_class": SingleConv2d,
        "input_shapes": [(4, 3, 8, 8)],
        "description": "Single Conv2d layer model",
    },
    "depthwise_conv": {
        "model_class": DepthwiseConv,
        "input_shapes": [(1, 32, 112, 112)],
        "description": "Single Depthwise Conv2d layer model",
    },
    "add": {
        "model_class": Add,
        "input_shapes": [(10,), (10,)],
        "description": "Simple tensor addition model",
    },
    "batchnorm": {
        "model_class": BatchNorm,
        "input_shapes": [(1, 16, 32, 32)],
        "description": "Single BatchNorm2d layer model",
    },
    "single_resnet_block": {
        "model_class": SingleResNetBlock,
        "input_shapes": [(1, 64, 8, 8)],
        "description": "Single ResNet block with skip connection",
    },
    "llama31": {
        "model_class": Llama31,
        "input_shapes": [(1, 32)],  # batch_size=1, sequence_length=128
        "description": "Llama 3.1 model with KV cache disabled",
    },
    "whisper": {
        "model_class": Whisper,
        "input_shapes": [(1, 80, 3000)],
        "description": "OpenAI Whisper ASR model. now is encoder only",
    },
    "conv1d": {
        "model_class": MockConv1d,
        "input_shapes": [(1, 80, 3000)],
        "description": "Conv1d layer with 80 input channels, 384 output channels",
    },
    "transformer_block": {
        "model_class": TransformerBlock,
        "input_shapes": [(4, 32, 256)],  # batch_size=4, seq_len=32, embed_dim=256
        "description": "Single transformer block with multi-head attention and feed-forward network",
    },
}


def get_model_and_inputs(
    model_name: str,
) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]:
    """Get model and example inputs based on model name."""
    #
    if model_name not in MODEL_REGISTRY:
        available_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model: {model_name}. Available models: {available_models}"
        )

    model_config = MODEL_REGISTRY[model_name]
    model_class = model_config["model_class"]
    input_shapes = model_config["input_shapes"]
    device = "cpu"

    # Create model instance
    model = model_class().to(device).eval()

    # Create example inputs (support multiple inputs)
    example_inputs = tuple(
        (
            torch.randint(0, 10000, size=shape, device=device)
            if model_name == "llama31"
            else torch.randn(*shape, device=device)
        )
        for shape in input_shapes
    )

    return model, example_inputs


def export_model_to_et_aoti(
    model, example_inputs, output_pte_path="aoti_model.pte", output_data_dir=None
):
    """Export model through the AOTI pipeline."""
    all_one_input = tuple(
        torch.ones_like(example_input) for example_input in example_inputs
    )

    label_output = model(*all_one_input)
    print("label", label_output)

    # Create directory if it doesn't exist
    os.makedirs("aoti_debug_data", exist_ok=True)

    # Dump label to file
    with open("aoti_debug_data/label_output.txt", "w") as f:
        if isinstance(label_output, tuple):
            # Multiple outputs
            all_elements = []
            for tensor in label_output:
                if tensor.numel() > 0:
                    all_elements.extend(tensor.flatten().tolist())
            f.write(",".join(map(str, all_elements)))
        else:
            # Single output
            if label_output.numel() > 0:
                f.write(",".join(map(str, label_output.flatten().tolist())))

    print(f"Starting export process...")

    print("Step 1: Converting to ATen dialect...")
    with torch.nn.attention.sdpa_kernel(
        [SDPBackend.MATH]  # pyre-fixme[16]
    ), torch.no_grad():
        # 1. torch.export: Defines the program with the ATen operator set.
        aten_dialect = export(model, example_inputs, strict=False)

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
    if output_data_dir is None:
        output_data_dir = os.getcwd()

    print(f"Step 5: Saving pte to {output_pte_path} and ptd to {output_data_dir}")
    with open(output_pte_path, "wb") as file:
        file.write(executorch_program.buffer)

    print(f"size of Named Data: {len(executorch_program._tensor_data)}")

    executorch_program.write_tensor_data_to_file(output_data_dir)

    print(
        f"Export completed successfully! PTE saved to {output_pte_path} and ptd saved to {output_data_dir}"
    )


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
        "aot_inductor.debug_intermediate_value_printer": "2",
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
