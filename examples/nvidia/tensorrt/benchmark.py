#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Benchmark script for TensorRT backend.

Exports supported models with TensorRT delegate and prepares them for benchmarking.
Use the C++ benchmark runner for actual inference timing.

Usage:
    # Export models for benchmarking:
    python -m executorch.examples.nvidia.tensorrt.benchmark -m mv2 mv3

    # Run benchmark with C++ runner (after building with cmake):
    ./cmake-out/backends/nvidia/tensorrt/benchmark_runner_tensorrt \
        --model_path=/tmp/benchmark/mv3_tensorrt.pte --num_executions=100
"""

import argparse
import logging
import os
from typing import Any, Dict, List, Tuple

import torch
from torch.export import export

from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.exir import to_edge_transform_and_lower

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

# Models supported by TensorRT backend
TENSORRT_SUPPORTED_MODELS = [
    "add",
    "mul",
    "linear",
    "add_mul",
    "softmax",
    "conv1d",
    "mv2",
    "mv3",
    "resnet18",
    "resnet50",
    "w2l",
    "ic3",
    "ic4",
    "dl3",
    "edsr",
    "emformer_join",
    "sdpa",
    "mobilebert",
    "efficient_sam",
]

# Default number of inference iterations
DEFAULT_NUM_ITERATIONS = 100

# Seed for reproducible random input generation
BENCHMARK_SEED = 2025


def get_model_and_inputs(model_name: str) -> Tuple[torch.nn.Module, Tuple[Any, ...]]:
    """Create model and example inputs from the model factory."""
    if model_name not in MODEL_NAME_TO_MODEL:
        raise ValueError(f"Model {model_name} not in MODEL_NAME_TO_MODEL")

    model, example_inputs, _, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[model_name]
    )
    model.eval()
    return model, example_inputs


def export_tensorrt(
    model: torch.nn.Module, example_inputs: Tuple[Any, ...], output_path: str
) -> bool:
    """Export model with TensorRT delegate and save to file."""
    try:
        from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner

        exported = export(model, example_inputs)
        edge_program = to_edge_transform_and_lower(
            exported,
            partitioner=[TensorRTPartitioner()],
        )
        exec_prog = edge_program.to_executorch()

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(exec_prog.buffer)
        return True
    except Exception as e:
        logger.warning(f"TensorRT export failed: {e}")
        return False


def benchmark_model(
    model_name: str,
    output_dir: str,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
) -> Dict[str, Any]:
    """Export a model with TensorRT for benchmarking."""
    logger.info(f"Exporting {model_name} for benchmarking...")

    result = {
        "model": model_name,
        "num_iterations": num_iterations,
        "exported": False,
        "pte_path": None,
    }

    try:
        model, example_inputs = get_model_and_inputs(model_name)

        # TensorRT export
        trt_path = os.path.join(output_dir, f"{model_name}_tensorrt.pte")
        if export_tensorrt(model, example_inputs, trt_path):
            result["exported"] = True
            result["pte_path"] = trt_path
            logger.info(f"  Exported to {trt_path}")

    except Exception as e:
        logger.error(f"Error exporting {model_name}: {e}")

    return result


def print_results(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Print export results and C++ runner commands."""
    print("\n" + "=" * 80)
    print("TENSORRT BENCHMARK EXPORT RESULTS")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Iterations: {results[0]['num_iterations'] if results else 'N/A'}")
    print("-" * 80)
    print(f"{'Model':<15} {'Status':<20} {'Path':<45}")
    print("-" * 80)

    for r in results:
        model = r["model"]
        status = "Exported" if r["exported"] else "Failed"
        path = r["pte_path"] or "N/A"
        print(f"{model:<15} {status:<20} {path:<45}")

    print("-" * 80)
    print("\nTo run benchmarks, use the C++ runner (after building with cmake):")
    print("=" * 80)
    for r in results:
        if r["exported"]:
            num_iter = r["num_iterations"]
            pte_path = r["pte_path"]
            print(
                f"# {r['model']}:\n"
                f"./cmake-out/backends/nvidia/tensorrt/benchmark_runner_tensorrt "
                f"--model_path={pte_path} --num_executions={num_iter}\n"
            )
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TensorRT backend")
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=TENSORRT_SUPPORTED_MODELS,
        help=f"Models to benchmark. Default: {TENSORRT_SUPPORTED_MODELS}",
    )
    parser.add_argument(
        "-n",
        "--num_iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help=f"Number of inference iterations per model. Default: {DEFAULT_NUM_ITERATIONS}",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="/tmp/benchmark",
        help="Output directory for exported models. Default: /tmp/benchmark",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for model_name in args.models:
        if model_name not in TENSORRT_SUPPORTED_MODELS:
            logger.warning(
                f"Model {model_name} not in supported models, skipping. "
                f"Supported: {TENSORRT_SUPPORTED_MODELS}"
            )
            continue
        result = benchmark_model(model_name, args.output_dir, args.num_iterations)
        results.append(result)

    print_results(results, args.output_dir)


if __name__ == "__main__":
    with torch.no_grad():
        main()
