#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script to compare performance between XNNPack and CPU backends
for the all-MiniLM-L6-v2 sentence-transformer model exported to ExecuTorch.

This script:
1. Exports the all-MiniLM-L6-v2 model with both XNNPack and CPU backends
2. Runs inference with different sentence lengths
3. Compares latency, throughput, and memory usage
4. Generates a performance comparison report

Example usage:
    # Run full benchmark with default settings
    python benchmark_backends.py

    # Run with more iterations for stable results
    python benchmark_backends.py --iterations 1000
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_models(model_name: str, max_seq_length: int, output_base_dir: Path):
    """
    Export the model with both XNNPack and CPU backends.

    Args:
        model_name: HuggingFace model name
        max_seq_length: Maximum sequence length
        output_base_dir: Base directory for exported models

    Returns:
        Tuple of (xnnpack_model_path, cpu_model_path)
    """
    from export_sentence_transformer import (
        export_cpu,
        export_with_xnnpack,
        SentenceTransformerModel,
    )

    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformerModel(model_name)
    model.eval()

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create example inputs
    text = "This is an example sentence for generating embeddings."
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
        return_tensors="pt",
    )
    example_inputs = (encoded["input_ids"], encoded["attention_mask"])

    # Export with XNNPack
    xnnpack_dir = output_base_dir / "xnnpack"
    xnnpack_dir.mkdir(parents=True, exist_ok=True)
    xnnpack_path = xnnpack_dir / "model.pte"

    logger.info("Exporting with XNNPack backend...")
    export_with_xnnpack(model, example_inputs, str(xnnpack_path))

    # Export with CPU
    cpu_dir = output_base_dir / "cpu"
    cpu_dir.mkdir(parents=True, exist_ok=True)
    cpu_path = cpu_dir / "model.pte"

    logger.info("Exporting with CPU backend...")
    export_cpu(model, example_inputs, str(cpu_path))

    return xnnpack_path, cpu_path


class ExecuTorchInferenceRunner:
    """Runner for ExecuTorch model inference."""

    def __init__(self, model_path: str, backend_name: str):
        """
        Initialize the inference runner.

        Args:
            model_path: Path to the .pte model file
            backend_name: Name of the backend (for logging)
        """
        self.model_path = model_path
        self.backend_name = backend_name

        # Import ExecuTorch runtime
        try:
            from executorch.extension.pybindings.portable_lib import (
                _load_for_executorch,
            )

            self._load_for_executorch = _load_for_executorch
        except ImportError:
            logger.error("Failed to import ExecuTorch runtime")
            raise

        # Load the model
        logger.info(f"Loading {backend_name} model from {model_path}")
        self.model = self._load_for_executorch(str(model_path))

    def run_inference(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Run inference on the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Sentence embeddings [batch_size, hidden_size]
        """
        outputs = self.model.forward((input_ids, attention_mask))
        return outputs[0]


def benchmark_model(
    runner: ExecuTorchInferenceRunner,
    tokenizer,
    test_sentences: List[str],
    max_seq_length: int,
    warmup_iterations: int,
    benchmark_iterations: int,
) -> Dict[str, float]:
    """
    Benchmark a model with given test sentences.

    Args:
        runner: ExecuTorch inference runner
        tokenizer: Tokenizer for the model
        test_sentences: List of test sentences
        max_seq_length: Maximum sequence length
        warmup_iterations: Number of warmup iterations
        benchmark_iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking {runner.backend_name} backend...")

    # Prepare inputs
    encoded = tokenizer(
        test_sentences,
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    batch_size = len(test_sentences)

    # Warmup
    logger.info(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        _ = runner.run_inference(input_ids, attention_mask)

    # Benchmark
    logger.info(f"Running {benchmark_iterations} benchmark iterations...")
    latencies = []

    for _ in range(benchmark_iterations):
        start_time = time.perf_counter()
        embeddings = runner.run_inference(input_ids, attention_mask)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    # Calculate statistics
    latencies_tensor = torch.tensor(latencies)
    avg_latency = latencies_tensor.mean().item()
    std_latency = latencies_tensor.std().item()
    min_latency = latencies_tensor.min().item()
    max_latency = latencies_tensor.max().item()
    p50_latency = latencies_tensor.median().item()
    p95_latency = torch.quantile(latencies_tensor, 0.95).item()
    p99_latency = torch.quantile(latencies_tensor, 0.99).item()

    throughput = (batch_size * 1000) / avg_latency  # sentences per second

    results = {
        "avg_latency_ms": avg_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "throughput_sps": throughput,
        "batch_size": batch_size,
    }

    logger.info(f"{runner.backend_name} results:")
    logger.info(f"  Average latency: {avg_latency:.2f} ms (± {std_latency:.2f} ms)")
    logger.info(
        f"  P50/P95/P99: {p50_latency:.2f} / {p95_latency:.2f} / {p99_latency:.2f} ms"
    )
    logger.info(f"  Throughput: {throughput:.2f} sentences/sec")

    return results


def generate_comparison_report(
    xnnpack_results: Dict[str, Dict[str, float]],
    cpu_results: Dict[str, Dict[str, float]],
    output_file: Path,
):
    """
    Generate a comparison report between XNNPack and CPU backends.

    Args:
        xnnpack_results: Results for XNNPack backend
        cpu_results: Results for CPU backend
        output_file: Path to save the report
    """
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE COMPARISON REPORT")
    logger.info("=" * 80)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PERFORMANCE COMPARISON: XNNPack vs CPU Backend")
    report_lines.append("=" * 80)
    report_lines.append("")

    for test_case in xnnpack_results.keys():
        xnn = xnnpack_results[test_case]
        cpu = cpu_results[test_case]

        report_lines.append(f"\nTest Case: {test_case}")
        report_lines.append("-" * 80)

        # Latency comparison
        speedup = cpu["avg_latency_ms"] / xnn["avg_latency_ms"]
        report_lines.append(f"\nLatency (Average):")
        report_lines.append(
            f"  XNNPack: {xnn['avg_latency_ms']:.2f} ms (± {xnn['std_latency_ms']:.2f})"
        )
        report_lines.append(
            f"  CPU:     {cpu['avg_latency_ms']:.2f} ms (± {cpu['std_latency_ms']:.2f})"
        )
        report_lines.append(f"  Speedup: {speedup:.2f}x {'✓' if speedup > 1 else ''}")

        # P95 latency
        report_lines.append(f"\nLatency (P95):")
        report_lines.append(f"  XNNPack: {xnn['p95_latency_ms']:.2f} ms")
        report_lines.append(f"  CPU:     {cpu['p95_latency_ms']:.2f} ms")

        # Throughput comparison
        throughput_improvement = (
            (xnn["throughput_sps"] - cpu["throughput_sps"])
            / cpu["throughput_sps"]
            * 100
        )
        report_lines.append(f"\nThroughput:")
        report_lines.append(f"  XNNPack: {xnn['throughput_sps']:.2f} sentences/sec")
        report_lines.append(f"  CPU:     {cpu['throughput_sps']:.2f} sentences/sec")
        report_lines.append(f"  Improvement: {throughput_improvement:+.1f}%")

        report_lines.append("")

    # Summary
    report_lines.append("\n" + "=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)

    # Calculate average speedup across all test cases
    speedups = [
        cpu_results[tc]["avg_latency_ms"] / xnnpack_results[tc]["avg_latency_ms"]
        for tc in xnnpack_results.keys()
    ]
    avg_speedup = sum(speedups) / len(speedups)

    report_lines.append(f"\nAverage speedup (XNNPack vs CPU): {avg_speedup:.2f}x")

    if avg_speedup > 1.2:
        report_lines.append("\n✓ XNNPack shows significant performance improvements!")
        report_lines.append("  Recommendation: Use XNNPack backend for production")
    elif avg_speedup > 1.0:
        report_lines.append("\n~ XNNPack shows moderate performance improvements")
        report_lines.append("  Recommendation: Use XNNPack for better performance")
    else:
        report_lines.append("\n✗ CPU backend is faster in this configuration")
        report_lines.append("  Note: This is unexpected - verify the setup")

    report_lines.append("")

    # Print and save report
    report_text = "\n".join(report_lines)
    print(report_text)

    with open(output_file, "w") as f:
        f.write(report_text)

    logger.info(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark XNNPack vs CPU backends for all-MiniLM-L6-v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )

    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for models and results (default: ./benchmark_results)",
    )

    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip model export (use existing models)",
    )

    args = parser.parse_args()

    # Hardcode the model we're benchmarking
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export models
    if not args.skip_export:
        xnnpack_path, cpu_path = export_models(
            model_name, args.max_seq_length, output_dir
        )
    else:
        xnnpack_path = output_dir / "xnnpack" / "model.pte"
        cpu_path = output_dir / "cpu" / "model.pte"

        if not xnnpack_path.exists() or not cpu_path.exists():
            logger.error("Models not found. Run without --skip-export first.")
            return

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create inference runners
    xnnpack_runner = ExecuTorchInferenceRunner(str(xnnpack_path), "XNNPack")
    cpu_runner = ExecuTorchInferenceRunner(str(cpu_path), "CPU")

    # Define test cases with batch size 1
    # Note: ExecuTorch models have static shapes, so we test with the exported shape (batch=1)
    test_cases = {
        "Single sentence (batch=1) - Short": ["This is a test sentence."],
        "Single sentence (batch=1) - Medium": [
            "Performance benchmarking helps us optimize models for production deployment."
        ],
        "Single sentence (batch=1) - Long": [
            "ExecuTorch enables on-device inference with optimized backends like XNNPack, making it possible to run transformer models efficiently on mobile devices and embedded systems."
        ],
    }

    # Run benchmarks
    xnnpack_results = {}
    cpu_results = {}

    for test_name, test_sentences in test_cases.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'='*60}")

        # Benchmark XNNPack
        xnnpack_results[test_name] = benchmark_model(
            xnnpack_runner,
            tokenizer,
            test_sentences,
            args.max_seq_length,
            args.warmup_iterations,
            args.iterations,
        )

        # Benchmark CPU
        cpu_results[test_name] = benchmark_model(
            cpu_runner,
            tokenizer,
            test_sentences,
            args.max_seq_length,
            args.warmup_iterations,
            args.iterations,
        )

    # Generate comparison report
    report_file = output_dir / "performance_report.txt"
    generate_comparison_report(xnnpack_results, cpu_results, report_file)

    logger.info("\n" + "=" * 60)
    logger.info("Benchmark complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
