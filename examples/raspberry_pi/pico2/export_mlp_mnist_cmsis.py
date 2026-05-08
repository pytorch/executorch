#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export the TinyMLP MNIST model with INT8 quantization for CMSIS-NN acceleration.

Uses the CortexMQuantizer to produce INT8 quantized ops that map to CMSIS-NN
kernels on Cortex-M33 (RP2350/Pico2). The model I/O stays float — quantize and
dequantize nodes are inserted inside the graph.

Usage:
    python export_mlp_mnist_cmsis.py
    python export_mlp_mnist_cmsis.py --output my_model.pte
    python export_mlp_mnist_cmsis.py --num-calibration 200
"""

import argparse
import logging
import os

import torch

from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.extension.export_util.utils import save_pte_program

from export_mlp_mnist import create_balanced_model, IMAGE_SIZE, test_comprehensive
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def get_calibration_data(num_samples: int = 100):
    """
    Generate calibration data for quantization.
    Mixes structured digit-like patterns and random noise so the observer
    sees a representative activation range.
    """
    calibration_data = []

    # Structured patterns that look like the digits the model will see
    for _ in range(num_samples // 2):
        x = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        # Random vertical / horizontal strokes
        col = torch.randint(5, 23, (1,)).item()
        row = torch.randint(5, 23, (1,)).item()
        x[0, 2:26, col - 1 : col + 2] = 1.0  # vertical stroke
        x[0, row - 1 : row + 2, 5:23] = 1.0  # horizontal stroke
        calibration_data.append(x)

    # Random pixel patterns
    for _ in range(num_samples - num_samples // 2):
        x = (torch.rand(1, IMAGE_SIZE, IMAGE_SIZE) > 0.7).float()
        calibration_data.append(x)

    return calibration_data


def quantize_model(model, calibration_data):
    quantizer = CortexMQuantizer()
    example_input = calibration_data[0]

    exported = torch.export.export(model, (example_input,))
    graph_module = exported.module()

    prepared = prepare_pt2e(graph_module, quantizer)

    logger.info(f"Calibrating with {len(calibration_data)} samples...")
    with torch.no_grad():
        for i, data in enumerate(calibration_data):
            prepared(data)
            if (i + 1) % 25 == 0:
                logger.info(f"  Calibrated {i + 1}/{len(calibration_data)} samples")

    quantized = convert_pt2e(prepared)
    return quantized, example_input


def export_to_pte(quantized_model, example_input, output_path: str):
    exported_program = torch.export.export(quantized_model, (example_input,))

    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        preserve_ops=[torch.ops.aten.linear.default],
    )
    edge_program = to_edge(exported_program, compile_config=edge_config)
    logger.info("Edge program created")

    logger.info("Applying Cortex-M optimization passes...")
    pass_manager = CortexMPassManager(edge_program.exported_program())
    transformed_ep = pass_manager.transform()

    edge_program = to_edge(transformed_ep, compile_config=edge_config)

    logger.info("Converting to ExecuTorch format...")
    exec_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    save_pte_program(exec_program, output_path)
    file_size = os.path.getsize(output_path)
    logger.info(f"Model saved to {output_path} ({file_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export TinyMLP MNIST for Cortex-M with CMSIS-NN (INT8)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="balanced_tiny_mlp_mnist_cmsis.pte",
        help="Output .pte file path",
    )
    parser.add_argument(
        "--num-calibration",
        type=int,
        default=100,
        help="Number of calibration samples for quantization",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Creating balanced MLP MNIST model...")
    model = create_balanced_model()
    model.eval()

    logger.info("Testing FP32 model before quantization:")
    test_comprehensive(model)

    calibration_data = get_calibration_data(args.num_calibration)
    quantized_model, example_input = quantize_model(model, calibration_data)

    logger.info("Testing quantized model:")
    with torch.no_grad():
        test_input = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        test_input[0, 2:26, 13:16] = 1.0  # digit-1-like pattern
        output = quantized_model(test_input)
        pred = output.argmax(dim=1).item()
        logger.info(f"  Digit-1 pattern -> predicted: {pred}")

    export_to_pte(quantized_model, example_input, args.output)
    logger.info("Export complete!")
    logger.info(f"Input shape: (1, {IMAGE_SIZE}, {IMAGE_SIZE})")
    logger.info("Input format: Float [0.0, 1.0] (same as FP32 variant)")


if __name__ == "__main__":
    main()
