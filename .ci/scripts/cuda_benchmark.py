#!/usr/bin/env python3
"""
Benchmark script for CUDA model runners.
Runs model runner commands multiple times and collects performance metrics.
Supports whisper, voxtral, gemma3, and other CUDA models.
"""
import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


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


def get_gpu_clocks() -> Optional[Tuple[str, str]]:
    """Get current GPU and memory clock frequencies."""
    try:
        # Get GPU clock
        result_gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.gr",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Get memory clock
        result_mem = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.mem",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result_gpu.returncode == 0 and result_mem.returncode == 0:
            gpu_clock = result_gpu.stdout.strip().split("\n")[0]
            mem_clock = result_mem.stdout.strip().split("\n")[0]
            return gpu_clock, mem_clock
    except Exception as e:
        print(f"Warning: Failed to get GPU clocks: {e}", file=sys.stderr)
    return None


def set_gpu_clocks(gpu_clock: Optional[int] = None) -> bool:
    """
    Set GPU clock frequency to a fixed value.

    Args:
        gpu_clock: Target GPU clock frequency in MHz.
                   If None, will use max available.

    Returns:
        True if successful, False otherwise
    """
    try:
        print("\n[GPU Clock Setup] Fixing GPU clock frequency...")

        # Enable persistence mode
        result = subprocess.run(
            ["sudo", "nvidia-smi", "-pm", "1"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            print(
                f"Warning: Failed to enable persistence mode: {result.stderr}",
                file=sys.stderr,
            )
            return False
        print("✓ Enabled persistence mode")

        # Lock GPU clocks
        if gpu_clock is None:
            # Get max GPU clock
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.max.gr",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                gpu_clock = int(result.stdout.strip().split("\n")[0])
                print(f"✓ Detected max GPU clock: {gpu_clock} MHz")

        # Lock GPU clock to the target frequency
        result = subprocess.run(
            ["sudo", "nvidia-smi", "-lgc", f"{gpu_clock},{gpu_clock}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            print(
                f"Warning: Failed to lock GPU clock: {result.stderr}",
                file=sys.stderr,
            )
            return False

        print(f"✓ Locked GPU clock to {gpu_clock} MHz")
        return True

    except Exception as e:
        print(f"Error: Failed to set GPU clocks: {e}", file=sys.stderr)
        return False


def reset_gpu_clocks() -> bool:
    """Reset GPU clock frequencies to default."""
    try:
        print("\n[GPU Clock Cleanup] Resetting GPU clock frequency...")

        # Reset GPU clocks
        result = subprocess.run(
            ["sudo", "nvidia-smi", "-rgc"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            print(
                f"Warning: Failed to reset GPU clock: {result.stderr}",
                file=sys.stderr,
            )
            return False
        print("✓ Reset GPU clock to default")

        # Disable persistence mode
        result = subprocess.run(
            ["sudo", "nvidia-smi", "-pm", "0"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            print(
                "Warning: Failed to disable persistence mode: " f"{result.stderr}",
                file=sys.stderr,
            )
            return False
        print("✓ Disabled persistence mode")

        return True

    except Exception as e:
        print(f"Error: Failed to reset GPU clocks: {e}", file=sys.stderr)
        return False


def run_whisper_benchmark(
    command: str,
    num_runs: int = 5,
    warmup_runs: int = 0,
    verbose: bool = False,
) -> List[RunMetrics]:
    """
    Run the whisper_runner command multiple times and collect metrics.

    For trimmed mean calculation, this function runs extra iterations
    to ensure we can trim outliers. Based on num_runs, we calculate
    trim_count = num_runs * 0.1, then run num_runs + 2*trim_count total
    iterations. The top and bottom trim_count results will be discarded.

    Args:
        command: Full command to run
        num_runs: Number of benchmark runs requested by user (after trim)
        warmup_runs: Number of warmup runs (results will be discarded)
        verbose: Print detailed output

    Returns:
        List of RunMetrics from benchmark runs (excluding warmup).
    """
    results = []
    # Calculate trim count based on requested num_runs
    trim_count = int(num_runs * 0.1)
    # Run extra iterations to account for trimming
    actual_benchmark_runs = num_runs + 2 * trim_count
    total_runs = warmup_runs + actual_benchmark_runs

    # Execute warmup runs
    if warmup_runs > 0:
        print(f"\n{'='*70}")
        print(f"WARMUP PHASE: Running {warmup_runs} warmup iterations...")
        print(f"{'='*70}")

    # Inform about trimmed mean strategy
    print(f"\n{'='*70}")
    print(f"BENCHMARK PHASE: Running {actual_benchmark_runs} iterations")
    print(f"Will trim top and bottom {trim_count} results " f"(10% of {num_runs})")
    print(f"Final statistics will be based on middle " f"{num_runs} results")
    print(f"{'='*70}")

    for run_num in range(1, total_runs + 1):
        is_warmup = run_num <= warmup_runs
        phase = "Warmup" if is_warmup else "Benchmark"
        benchmark_run_num = run_num - warmup_runs if not is_warmup else run_num

        if is_warmup:
            print(f"\n[{phase} {run_num}/{warmup_runs}] " f"Executing: {command}")
        else:
            print(
                f"\n[{phase} {benchmark_run_num}/{actual_benchmark_runs}] "
                f"Executing: {command}"
            )

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
                    "Error: Command failed with return code " f"{result.returncode}",
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
                    "Warning: No PyTorchObserver output found in " f"run {run_num}",
                    file=sys.stderr,
                )
                if verbose:
                    print(f"stdout:\n{result.stdout}", file=sys.stderr)
                continue

            # Parse metrics
            metrics = parse_pytorch_observer_log(observer_line)
            if metrics is None:
                print(
                    f"Warning: Failed to parse metrics from " f"run {run_num}",
                    file=sys.stderr,
                )
                continue

            # Only collect results from benchmark runs (not warmup)
            if not is_warmup:
                results.append(metrics)
            print(f"✓ {metrics}")

        except subprocess.TimeoutExpired:
            print(
                f"Error: Command timed out on run {run_num}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"Error on run {run_num}: {e}", file=sys.stderr)

    return results


def calculate_trimmed_stats(
    values: List[float], trim_count: int
) -> Tuple[List[float], float, float, float, float]:
    """
    Calculate statistics on trimmed data.

    Args:
        values: List of numeric values
        trim_count: Number of values to trim from each end

    Returns:
        Tuple of (trimmed_values, min, max, mean, stdev)
    """
    if not values:
        return [], 0.0, 0.0, 0.0, 0.0

    # Sort values
    sorted_values = sorted(values)
    n = len(sorted_values)

    # Trim if we have enough data and trim_count > 0
    if trim_count > 0 and n > 2 * trim_count:
        trimmed_values = sorted_values[trim_count : n - trim_count]
    else:
        trimmed_values = sorted_values

    # Calculate stats on trimmed data
    min_val = min(trimmed_values)
    max_val = max(trimmed_values)
    mean_val = statistics.mean(trimmed_values)
    stdev_val = statistics.stdev(trimmed_values) if len(trimmed_values) > 1 else 0.0

    return trimmed_values, min_val, max_val, mean_val, stdev_val


@dataclass
class BenchmarkResults:
    """Summary of benchmark results with trimmed statistics."""

    model_name: str
    total_runs: int
    trimmed_runs: int
    discarded_runs: int
    generated_tokens: int
    throughput_mean: float
    throughput_min: float
    throughput_max: float
    throughput_stdev: float
    model_load_time_mean: float
    model_load_time_min: float
    model_load_time_max: float
    model_load_time_stdev: float
    inference_time_mean: float
    inference_time_min: float
    inference_time_max: float
    inference_time_stdev: float
    first_token_latency_mean: float
    first_token_latency_min: float
    first_token_latency_max: float
    first_token_latency_stdev: float

    def to_dict(self) -> dict:
        """Convert results to dictionary for JSON serialization."""
        return asdict(self)

    def save_json(self, output_path: str) -> None:
        """Save results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")


def compute_summary(
    model_name: str, results: List[RunMetrics], requested_runs: int
) -> BenchmarkResults:
    """
    Compute summary statistics using trimmed data.

    All statistics (min, max, mean, stdev) are calculated based on
    the trimmed dataset after removing outliers.

    Args:
        model_name: Name of the model being benchmarked
        results: List of all collected run metrics
        requested_runs: Number of runs originally requested by user

    Returns:
        BenchmarkResults object with all computed statistics
    """
    if not results:
        raise ValueError("No valid results to summarize.")

    # Calculate trim count based on requested runs (not actual runs)
    trim_count = int(requested_runs * 0.1)

    tokens_per_sec_list = [r.tokens_per_sec for r in results]
    model_load_times = [r.model_load_time_ms for r in results]
    inference_times = [r.inference_time_ms for r in results]
    first_token_latencies = [r.first_token_latency_ms for r in results]

    # Calculate trimmed statistics for each metric
    (
        trimmed_throughput,
        throughput_min,
        throughput_max,
        throughput_mean,
        throughput_stdev,
    ) = calculate_trimmed_stats(tokens_per_sec_list, trim_count)

    (
        _,
        load_min,
        load_max,
        load_mean,
        load_stdev,
    ) = calculate_trimmed_stats(model_load_times, trim_count)

    (
        _,
        inference_min,
        inference_max,
        inference_mean,
        inference_stdev,
    ) = calculate_trimmed_stats(inference_times, trim_count)

    (
        _,
        latency_min,
        latency_max,
        latency_mean,
        latency_stdev,
    ) = calculate_trimmed_stats(first_token_latencies, trim_count)

    return BenchmarkResults(
        model_name=model_name,
        total_runs=len(results),
        trimmed_runs=len(trimmed_throughput),
        discarded_runs=trim_count * 2,
        generated_tokens=results[0].generated_tokens,
        throughput_mean=throughput_mean,
        throughput_min=throughput_min,
        throughput_max=throughput_max,
        throughput_stdev=throughput_stdev,
        model_load_time_mean=load_mean,
        model_load_time_min=load_min,
        model_load_time_max=load_max,
        model_load_time_stdev=load_stdev,
        inference_time_mean=inference_mean,
        inference_time_min=inference_min,
        inference_time_max=inference_max,
        inference_time_stdev=inference_stdev,
        first_token_latency_mean=latency_mean,
        first_token_latency_min=latency_min,
        first_token_latency_max=latency_max,
        first_token_latency_stdev=latency_stdev,
    )


def print_summary(summary: BenchmarkResults) -> None:
    """Print formatted summary of benchmark results."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY for model: {summary.model_name}")
    print("=" * 70)
    print(f"Total runs collected: {summary.total_runs}")
    print(f"Trimmed to: {summary.trimmed_runs} runs")
    print(
        f"(Discarded {summary.discarded_runs // 2} highest and "
        f"{summary.discarded_runs // 2} lowest results)"
    )
    print(f"Generated tokens per run: {summary.generated_tokens}")
    print()

    print("THROUGHPUT (tokens/sec):")
    print(f"  Min:    {summary.throughput_min:.2f} t/s")
    print(f"  Max:    {summary.throughput_max:.2f} t/s")
    print(f"  Mean:   {summary.throughput_mean:.2f} t/s")
    print(f"  Stdev:  {summary.throughput_stdev:.2f} t/s")
    print()

    print("MODEL LOAD TIME (ms):")
    print(f"  Min:    {summary.model_load_time_min:.0f} ms")
    print(f"  Max:    {summary.model_load_time_max:.0f} ms")
    print(f"  Mean:   {summary.model_load_time_mean:.0f} ms")
    print(f"  Stdev:  {summary.model_load_time_stdev:.0f} ms")
    print()

    print("INFERENCE TIME (ms, prompt_eval_end to inference_end):")
    print(f"  Min:    {summary.inference_time_min:.0f} ms")
    print(f"  Max:    {summary.inference_time_max:.0f} ms")
    print(f"  Mean:   {summary.inference_time_mean:.0f} ms")
    print(f"  Stdev:  {summary.inference_time_stdev:.0f} ms")
    print()

    print("FIRST TOKEN LATENCY (ms):")
    print(f"  Min:    {summary.first_token_latency_min:.0f} ms")
    print(f"  Max:    {summary.first_token_latency_max:.0f} ms")
    print(f"  Mean:   {summary.first_token_latency_mean:.0f} ms")
    print(f"  Stdev:  {summary.first_token_latency_stdev:.0f} ms")
    print("=" * 70)
    print()
    print("=" * 70)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA model runners and collect performance metrics"
    )
    parser.add_argument(
        "--runner_command",
        type=str,
        required=True,
        help="Full command to run the model runner",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model being benchmarked",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=50,
        help="Number of benchmark runs (default: 50)",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=0,
        help="Number of warmup runs before benchmark (default: 0.1 * num_runs)",
    )
    parser.add_argument(
        "--fix_gpu_clock",
        action="store_true",
        help="Fix GPU clock frequency to maximum before benchmarking",
    )
    parser.add_argument(
        "--gpu_clock",
        type=int,
        default=None,
        help="Target GPU clock frequency in MHz (requires "
        "--fix_gpu_clock). If not specified, uses max available.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save JSON results",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    warmup_runs = (
        int(0.1 * args.num_runs) if args.warmup_runs == 0 else args.warmup_runs
    )

    print(f"Running benchmark for model: {args.model_name}")
    print(f"Number of runs: {args.num_runs}")
    if warmup_runs > 0:
        print(f"Warmup runs: {warmup_runs}")
    if args.fix_gpu_clock:
        clock_str = f"{args.gpu_clock}" if args.gpu_clock else "max available"
        print(f"GPU clock will be fixed to: {clock_str} MHz")
    print(f"Command: {args.runner_command}\n")

    # Fix GPU clocks if requested
    gpu_clock_fixed = False
    if args.fix_gpu_clock:
        # Get current clocks before fixing
        initial_clocks = get_gpu_clocks()
        if initial_clocks:
            print(
                f"Current GPU clocks - GPU: {initial_clocks[0]} MHz, "
                f"Memory: {initial_clocks[1]} MHz"
            )

        gpu_clock_fixed = set_gpu_clocks(args.gpu_clock)
        if not gpu_clock_fixed:
            print(
                "Warning: Failed to fix GPU clocks. "
                "Continuing without fixed clocks...",
                file=sys.stderr,
            )

    try:
        # Run benchmark
        results = run_whisper_benchmark(
            command=args.runner_command,
            num_runs=args.num_runs,
            warmup_runs=warmup_runs,
            verbose=args.verbose,
        )

        # Compute and print summary
        summary = compute_summary(args.model_name, results, args.num_runs)
        print_summary(summary)

        # Save JSON results if requested
        if args.output_json:
            summary.save_json(args.output_json)
            print(f"✓ Results saved to: {args.output_json}")

    finally:
        # Reset GPU clocks if they were fixed
        if gpu_clock_fixed:
            reset_gpu_clocks()


if __name__ == "__main__":
    main()
