#!/usr/bin/env python3
"""
Benchmark script for Whisper ASR runner.
Runs the whisper_runner command multiple times and collects throughput metrics.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class RunMetrics:
    """Metrics from a single run."""

    generated_tokens: int
    tokens_per_sec: float
    model_load_time_ms: float
    inference_time_ms: float
    prompt_eval_to_end_ms: float
    first_token_latency_ms: float

    def __repr__(self):
        return (
            f"Tokens: {self.generated_tokens}, "
            f"Throughput: {self.tokens_per_sec:.2f} t/s, "
            f"Model load: {self.model_load_time_ms:.0f}ms, "
            f"Inference: {self.inference_time_ms:.0f}ms, "
            f"First token: {self.first_token_latency_ms:.0f}ms"
        )


def parse_pytorch_observer_log(log_line: str) -> Optional[RunMetrics]:
    """Parse PyTorchObserver JSON output and compute metrics."""
    try:
        # Find the JSON part in the log line
        if "PyTorchObserver" not in log_line:
            return None

        json_str = log_line.split("PyTorchObserver")[1].strip()
        data = json.loads(json_str)

        # Extract values
        generated_tokens = data.get("generated_tokens", 0)
        inference_end_ms = data.get("inference_end_ms", 0)
        prompt_eval_end_ms = data.get("prompt_eval_end_ms", 0)
        first_token_ms = data.get("first_token_ms", 0)
        model_load_start_ms = data.get("model_load_start_ms", 0)
        model_load_end_ms = data.get("model_load_end_ms", 0)

        # Compute metrics
        prompt_eval_to_end_ms = inference_end_ms - prompt_eval_end_ms
        tokens_per_sec = (
            (generated_tokens / prompt_eval_to_end_ms * 1000)
            if prompt_eval_to_end_ms > 0
            else 0
        )
        model_load_time_ms = model_load_end_ms - model_load_start_ms
        inference_time_ms = inference_end_ms - prompt_eval_end_ms
        first_token_latency_ms = first_token_ms - prompt_eval_end_ms

        return RunMetrics(
            generated_tokens=generated_tokens,
            tokens_per_sec=tokens_per_sec,
            model_load_time_ms=model_load_time_ms,
            inference_time_ms=inference_time_ms,
            prompt_eval_to_end_ms=prompt_eval_to_end_ms,
            first_token_latency_ms=first_token_latency_ms,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing PyTorchObserver log: {e}", file=sys.stderr)
        return None


def run_whisper_benchmark(
    command: str, num_runs: int = 5, verbose: bool = False
) -> List[RunMetrics]:
    """
    Run the whisper_runner command multiple times and collect metrics.

    Args:
        command: Full command to run
        num_runs: Number of times to run the command
        verbose: Print detailed output

    Returns:
        List of RunMetrics from each run
    """
    results = []

    for run_num in range(1, num_runs + 1):
        print(f"\n[Run {run_num}/{num_runs}] Executing: {command}")

        try:
            # Run command and capture output
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                print(
                    f"Error: Command failed with return code {result.returncode}",
                    file=sys.stderr,
                )
                if result.stderr:
                    print(f"stderr: {result.stderr}", file=sys.stderr)
                continue

            # Search for PyTorchObserver line in output
            observer_line = None
            for line in result.stdout.split("\n"):
                if "PyTorchObserver" in line:
                    observer_line = line
                    break

            if observer_line is None:
                print(
                    f"Warning: No PyTorchObserver output found in run {run_num}",
                    file=sys.stderr,
                )
                if verbose:
                    print(f"stdout:\n{result.stdout}", file=sys.stderr)
                continue

            # Parse metrics
            metrics = parse_pytorch_observer_log(observer_line)
            if metrics is None:
                print(
                    f"Warning: Failed to parse metrics from run {run_num}",
                    file=sys.stderr,
                )
                continue

            results.append(metrics)
            print(f"✓ {metrics}")

        except subprocess.TimeoutExpired:
            print(f"Error: Command timed out on run {run_num}", file=sys.stderr)
        except Exception as e:
            print(f"Error on run {run_num}: {e}", file=sys.stderr)

    return results


def print_summary(results: List[RunMetrics]) -> None:
    """Print summary statistics."""
    if not results:
        print("No valid results to summarize.")
        return

    tokens_per_sec_list = [r.tokens_per_sec for r in results]
    model_load_times = [r.model_load_time_ms for r in results]
    inference_times = [r.inference_time_ms for r in results]
    first_token_latencies = [r.first_token_latency_ms for r in results]

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total runs: {len(results)}")
    print(f"Generated tokens per run: {results[0].generated_tokens}")
    print()

    print("THROUGHPUT (tokens/sec):")
    print(f"  Min:    {min(tokens_per_sec_list):.2f} t/s")
    print(f"  Max:    {max(tokens_per_sec_list):.2f} t/s")
    print(f"  Mean:   {statistics.mean(tokens_per_sec_list):.2f} t/s")
    if len(tokens_per_sec_list) > 1:
        print(f"  Stdev:  {statistics.stdev(tokens_per_sec_list):.2f} t/s")
    print()

    print("MODEL LOAD TIME (ms):")
    print(f"  Min:    {min(model_load_times):.0f} ms")
    print(f"  Max:    {max(model_load_times):.0f} ms")
    print(f"  Mean:   {statistics.mean(model_load_times):.0f} ms")
    if len(model_load_times) > 1:
        print(f"  Stdev:  {statistics.stdev(model_load_times):.0f} ms")
    print()

    print("INFERENCE TIME (ms, prompt_eval_end to inference_end):")
    print(f"  Min:    {min(inference_times):.0f} ms")
    print(f"  Max:    {max(inference_times):.0f} ms")
    print(f"  Mean:   {statistics.mean(inference_times):.0f} ms")
    if len(inference_times) > 1:
        print(f"  Stdev:  {statistics.stdev(inference_times):.0f} ms")
    print()

    print("FIRST TOKEN LATENCY (ms):")
    print(f"  Min:    {min(first_token_latencies):.0f} ms")
    print(f"  Max:    {max(first_token_latencies):.0f} ms")
    print(f"  Mean:   {statistics.mean(first_token_latencies):.0f} ms")
    if len(first_token_latencies) > 1:
        print(f"  Stdev:  {statistics.stdev(first_token_latencies):.0f} ms")
    print("=" * 70)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper ASR runner and collect throughput metrics"
    )
    parser.add_argument(
        "num_runs",
        type=int,
        nargs="?",
        default=50,
        help="Number of benchmark runs (default: 5)",
    )
    parser.add_argument(
        "--model_dir_name",
        type=str,
        default="decomposed",
        help="Path to the directory that has model .pte and .ptd files",
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default="~/kernel-gen/whisper-large-v3-turbo/decomposed/whisper_preprocessor.pte",
        help="Path to the preprocessor/processor .pte file",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    base_path = "~/kernel-gen/whisper-large-v3-turbo/"
    model_dir_path = os.path.join(base_path, args.model_dir_name)

    # Expand user paths
    model_path = os.path.expanduser(model_dir_path + "/model.pte")
    data_path = os.path.expanduser(model_dir_path + "/aoti_cuda_blob.ptd")
    tokenizer_path = os.path.expanduser(
        "~/kernel-gen/whisper-large-v3-turbo/decomposed"
    )
    audio_path = os.path.expanduser(
        "~/kernel-gen/whisper-large-v3-turbo/decomposed/output.wav"
    )
    processor_path = os.path.expanduser(args.processor_path)

    # Build command
    command = (
        "cmake-out/examples/models/whisper/whisper_runner "
        f"--model_path {model_path} "
        f"--data_path {data_path} "
        f"--tokenizer_path {tokenizer_path} "
        f"--audio_path {audio_path} "
        f"--processor_path {processor_path} "
        "--model_name whisper_large_v3 "
        "--temperature 0 "
    )

    print(f"Running Whisper benchmark {args.num_runs} times...")
    print(f"Command: {command}\n")

    # Run benchmark
    results = run_whisper_benchmark(
        command, num_runs=args.num_runs, verbose=args.verbose
    )

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
