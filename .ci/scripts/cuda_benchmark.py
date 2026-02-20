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
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RunMetrics:
    """Metrics from a single run."""

    generated_tokens: int
    tokens_per_sec: float
    model_load_time_ms: float
    total_inference_time_ms: float
    encoder_time_ms: float
    generation_time_ms: float
    first_token_latency_ms: float

    def __repr__(self):
        return (
            f"Tokens: {self.generated_tokens}, "
            f"Throughput: {self.tokens_per_sec:.2f} t/s, "
            f"Model load: {self.model_load_time_ms:.0f}ms, "
            f"Total inference: {self.total_inference_time_ms:.0f}ms, "
            f"Encoder: {self.encoder_time_ms:.0f}ms, "
            f"Generation: {self.generation_time_ms:.0f}ms, "
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
        inference_start_ms = data.get("inference_start_ms", 0)
        inference_end_ms = data.get("inference_end_ms", 0)
        prompt_eval_end_ms = data.get("prompt_eval_end_ms", 0)
        first_token_ms = data.get("first_token_ms", 0)
        model_load_start_ms = data.get("model_load_start_ms", 0)
        model_load_end_ms = data.get("model_load_end_ms", 0)

        # Compute metrics
        # Total inference time: from inference start to inference end
        total_inference_time_ms = inference_end_ms - inference_start_ms

        # Encoder time: from inference start to prompt evaluation end
        encoder_time_ms = prompt_eval_end_ms - inference_start_ms

        # Generation time: from prompt evaluation end to inference end
        generation_time_ms = inference_end_ms - prompt_eval_end_ms

        # Calculate throughput based on generation time
        tokens_per_sec = (
            (generated_tokens / generation_time_ms * 1000)
            if generation_time_ms > 0
            else 0
        )
        model_load_time_ms = model_load_end_ms - model_load_start_ms

        # First token latency (TTFT): time from inference start to first token
        # For multimodal models (e.g., whisper, voxtral, gemma3), this includes:
        # 1. Encoding time (image/audio processing)
        # 2. Prefill time (decoder processing encoder outputs)
        # This represents the end-to-end user experience of waiting for the first token
        first_token_latency_ms = first_token_ms - inference_start_ms

        return RunMetrics(
            generated_tokens=generated_tokens,
            tokens_per_sec=tokens_per_sec,
            model_load_time_ms=model_load_time_ms,
            total_inference_time_ms=total_inference_time_ms,
            encoder_time_ms=encoder_time_ms,
            generation_time_ms=generation_time_ms,
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


def _print_warmup_info(warmup_runs: int) -> None:
    """Print warmup phase information."""
    if warmup_runs > 0:
        print(f"\n{'='*70}")
        print(f"WARMUP PHASE: Running {warmup_runs} warmup iterations...")
        print(f"{'='*70}")


def _print_benchmark_info(
    actual_benchmark_runs: int, trim_count: int, num_runs: int
) -> None:
    """Print benchmark phase information."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK PHASE: Running {actual_benchmark_runs} iterations")
    print(f"Will trim top and bottom {trim_count} results (10% of {num_runs})")
    print(f"Final statistics will be based on middle {num_runs} results")
    print(f"{'='*70}")


def _run_single_iteration(
    command: str, run_num: int, verbose: bool
) -> Optional[RunMetrics]:
    """
    Run a single benchmark iteration and return metrics.

    Args:
        command: Command to execute
        run_num: Current run number
        verbose: Print verbose output

    Returns:
        RunMetrics if successful, None otherwise
    """
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
            return None

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
            return None

        # Parse and return metrics
        metrics = parse_pytorch_observer_log(observer_line)
        if metrics is None:
            print(
                f"Warning: Failed to parse metrics from run {run_num}",
                file=sys.stderr,
            )
            return None

        print(f"✓ {metrics}")
        return metrics

    except subprocess.TimeoutExpired:
        print(f"Error: Command timed out on run {run_num}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error on run {run_num}: {e}", file=sys.stderr)
        return None


def run_model_benchmark(
    command: str,
    num_runs: int = 5,
    warmup_runs: int = 0,
    verbose: bool = False,
) -> List[RunMetrics]:
    """
    Run the model runner command multiple times and collect metrics.

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
    # Calculate trim count and total runs
    trim_count = int(num_runs * 0.1)
    actual_benchmark_runs = num_runs + 2 * trim_count
    total_runs = warmup_runs + actual_benchmark_runs

    # Print phase information
    _print_warmup_info(warmup_runs)
    _print_benchmark_info(actual_benchmark_runs, trim_count, num_runs)

    # Execute all runs
    results = []
    for run_num in range(1, total_runs + 1):
        is_warmup = run_num <= warmup_runs
        phase = "Warmup" if is_warmup else "Benchmark"
        benchmark_run_num = run_num - warmup_runs if not is_warmup else run_num

        # Print run header
        if is_warmup:
            print(f"\n[{phase} {run_num}/{warmup_runs}] Executing: {command}")
        else:
            print(
                f"\n[{phase} {benchmark_run_num}/{actual_benchmark_runs}] "
                f"Executing: {command}"
            )

        # Run iteration and collect metrics
        metrics = _run_single_iteration(command, run_num, verbose)
        if metrics is not None and not is_warmup:
            results.append(metrics)

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
class MetricStats:
    """Statistics for a single metric with operations."""

    name: str
    mean: float
    min_val: float
    max_val: float
    stdev: float
    unit: str = ""
    extra_info: dict | None = None

    def create_v3_record(
        self,
        model_name: str,
        backend: str,
        runner_name: str,
        runner_type: str,
        base_extra_info: dict,
    ) -> dict:
        """
        Create a v3 format record for this metric.

        Args:
            model_name: Model name with quantization
            backend: Backend name (e.g., "cuda-aoti")
            runner_name: GPU device name
            runner_type: CUDA driver version
            base_extra_info: Base extra_info dict to copy

        Returns:
            Complete v3 format metric record
        """
        extra_stats = {
            "min": self.min_val,
            "max": self.max_val,
            "stdev": self.stdev,
        }
        if self.extra_info:
            extra_stats.update(self.extra_info)

        return {
            "benchmark": {
                "name": "ExecuTorch",
                "mode": "inference",
                "extra_info": base_extra_info.copy(),
            },
            "model": {
                "name": model_name,
                "type": "OSS model",
                "backend": backend,
            },
            "metric": {
                "name": self.name,
                "benchmark_values": [self.mean],
                "target_value": 0,
                "extra_info": extra_stats,
            },
            "runners": [{"name": runner_name, "type": runner_type}],
        }

    def print_stats(self) -> None:
        """Print formatted statistics for this metric."""
        # Determine precision based on metric type
        is_throughput = "tokens" in self.name.lower()
        precision = 2 if is_throughput else 0

        # Format metric name for display
        display_name = self.name.replace("_", " ").upper()
        if self.unit:
            display_name = f"{display_name} ({self.unit})"

        print(f"{display_name}:")
        print(f"  Min:    {self.min_val:.{precision}f} {self.unit}")
        print(f"  Max:    {self.max_val:.{precision}f} {self.unit}")
        print(f"  Mean:   {self.mean:.{precision}f} {self.unit}")
        print(f"  Stdev:  {self.stdev:.{precision}f} {self.unit}")
        print()


@dataclass
class BenchmarkResults:
    """Summary of benchmark results."""

    model_name: str
    total_runs: int
    trimmed_runs: int
    discarded_runs: int
    generated_tokens: int

    # Metrics
    throughput: MetricStats
    model_load_time: MetricStats
    total_inference_time: MetricStats
    encoder_time: MetricStats
    generation_time: MetricStats
    first_token_latency: MetricStats

    def save_json(self, output_path: str) -> None:
        """Save results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")

    def to_dict(self) -> dict:
        """Convert results to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "total_runs": self.total_runs,
            "trimmed_runs": self.trimmed_runs,
            "discarded_runs": self.discarded_runs,
            "generated_tokens": self.generated_tokens,
            "throughput_mean": self.throughput.mean,
            "throughput_min": self.throughput.min_val,
            "throughput_max": self.throughput.max_val,
            "throughput_stdev": self.throughput.stdev,
            "model_load_time_mean": self.model_load_time.mean,
            "model_load_time_min": self.model_load_time.min_val,
            "model_load_time_max": self.model_load_time.max_val,
            "model_load_time_stdev": self.model_load_time.stdev,
            "total_inference_time_mean": self.total_inference_time.mean,
            "total_inference_time_min": self.total_inference_time.min_val,
            "total_inference_time_max": self.total_inference_time.max_val,
            "total_inference_time_stdev": self.total_inference_time.stdev,
            "encoder_time_mean": self.encoder_time.mean,
            "encoder_time_min": self.encoder_time.min_val,
            "encoder_time_max": self.encoder_time.max_val,
            "encoder_time_stdev": self.encoder_time.stdev,
            "generation_time_mean": self.generation_time.mean,
            "generation_time_min": self.generation_time.min_val,
            "generation_time_max": self.generation_time.max_val,
            "generation_time_stdev": self.generation_time.stdev,
            "first_token_latency_mean": self.first_token_latency.mean,
            "first_token_latency_min": self.first_token_latency.min_val,
            "first_token_latency_max": self.first_token_latency.max_val,
            "first_token_latency_stdev": self.first_token_latency.stdev,
        }

    def to_v3_format(
        self,
        model: str,
        quantization: str,
        git_sha: str,
        workflow_run_id: str,
        workflow_run_url: str = "",
        gpu_name: str = "CUDA",
        cuda_driver_version: str = "cuda",
    ) -> List[dict]:
        """
        Transform benchmark results to PyTorch benchmark database v3 format.

        Args:
            model: Model name (e.g., "openai/whisper-small")
            quantization: Quantization type (e.g., "non-quantized")
            git_sha: Git commit SHA
            workflow_run_id: GitHub workflow run ID
            workflow_run_url: GitHub workflow run URL
            gpu_name: GPU device name (e.g., "Tesla V100", "A100")
            cuda_driver_version: CUDA driver version (e.g., "12.6", "535.104.05")

        Returns:
            List of benchmark records in v3 format
        """
        # Shared configuration
        model_name_with_quant = f"{model}_{quantization}"
        backend = "cuda-aoti"
        runner_name = gpu_name
        runner_type = cuda_driver_version

        # Create base extra_info
        base_extra_info = {
            "backend": "cuda",
            "quantization": quantization,
            "git_sha": git_sha,
            "workflow_run_id": workflow_run_id,
        }
        if workflow_run_url:
            base_extra_info["workflow_run_url"] = workflow_run_url

        # Create v3 records for all metrics
        return [
            self.throughput.create_v3_record(
                model_name_with_quant,
                backend,
                runner_name,
                runner_type,
                base_extra_info,
            ),
            self.model_load_time.create_v3_record(
                model_name_with_quant,
                backend,
                runner_name,
                runner_type,
                base_extra_info,
            ),
            self.total_inference_time.create_v3_record(
                model_name_with_quant,
                backend,
                runner_name,
                runner_type,
                base_extra_info,
            ),
            self.encoder_time.create_v3_record(
                model_name_with_quant,
                backend,
                runner_name,
                runner_type,
                base_extra_info,
            ),
            self.generation_time.create_v3_record(
                model_name_with_quant,
                backend,
                runner_name,
                runner_type,
                base_extra_info,
            ),
            self.first_token_latency.create_v3_record(
                model_name_with_quant,
                backend,
                runner_name,
                runner_type,
                base_extra_info,
            ),
        ]


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

    # Helper to create MetricStats from values
    def create_metric_stats(
        name: str, values: List[float], unit: str = "", extra_info: dict | None = None
    ) -> MetricStats:
        _, min_val, max_val, mean_val, stdev_val = calculate_trimmed_stats(
            values, trim_count
        )
        return MetricStats(
            name=name,
            mean=mean_val,
            min_val=min_val,
            max_val=max_val,
            stdev=stdev_val,
            unit=unit,
            extra_info=extra_info,
        )

    # Get the first trimmed result to get trimmed_runs count
    trimmed_throughput, _, _, _, _ = calculate_trimmed_stats(
        [r.tokens_per_sec for r in results], trim_count
    )

    return BenchmarkResults(
        model_name=model_name,
        total_runs=len(results),
        trimmed_runs=len(trimmed_throughput),
        discarded_runs=trim_count * 2,
        generated_tokens=results[0].generated_tokens,
        throughput=create_metric_stats(
            "throughput(tokens/sec)",
            [r.tokens_per_sec for r in results],
            "t/s",
            {"trimmed_runs": len(trimmed_throughput)},
        ),
        model_load_time=create_metric_stats(
            "model_load_time(ms)",
            [r.model_load_time_ms for r in results],
            "ms",
        ),
        total_inference_time=create_metric_stats(
            "total_inference_time(ms)",
            [r.total_inference_time_ms for r in results],
            "ms",
        ),
        encoder_time=create_metric_stats(
            "encoder_time(ms)",
            [r.encoder_time_ms for r in results],
            "ms",
        ),
        generation_time=create_metric_stats(
            "generation_time(ms)",
            [r.generation_time_ms for r in results],
            "ms",
        ),
        first_token_latency=create_metric_stats(
            "first_token_latency(ms)",
            [r.first_token_latency_ms for r in results],
            "ms",
        ),
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

    # Print all metrics using their print_stats method
    summary.throughput.print_stats()
    summary.model_load_time.print_stats()
    summary.total_inference_time.print_stats()
    summary.encoder_time.print_stats()
    summary.generation_time.print_stats()
    summary.first_token_latency.print_stats()

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
        type=bool,
        default=True,
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
    parser.add_argument(
        "--output_v3",
        type=str,
        default=None,
        help="Path to save v3 format JSON results for dashboard",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (e.g., 'openai/whisper-small') - required for v3 format",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization type (e.g., 'non-quantized') - required for v3 format",
    )
    parser.add_argument(
        "--git_sha",
        type=str,
        default=None,
        help="Git commit SHA - required for v3 format",
    )
    parser.add_argument(
        "--workflow_run_id",
        type=str,
        default=None,
        help="GitHub workflow run ID - required for v3 format",
    )
    parser.add_argument(
        "--workflow_run_url",
        type=str,
        default="",
        help="GitHub workflow run URL - optional for v3 format",
    )
    parser.add_argument(
        "--gpu_name",
        type=str,
        default=None,
        help="GPU device name (e.g., 'Tesla V100', 'A100') - optional for v3 format",
    )
    parser.add_argument(
        "--cuda_driver_version",
        type=str,
        default=None,
        help="CUDA driver version (e.g., '12.6', '535.104.05') - optional for v3 format",
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
        results = run_model_benchmark(
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

        # Save v3 format if requested
        if args.output_v3:
            # Validate required parameters for v3 format
            if not all(
                [args.model, args.quantization, args.git_sha, args.workflow_run_id]
            ):
                print(
                    "Error: --output_v3 requires --model, --quantization, "
                    "--git_sha, and --workflow_run_id",
                    file=sys.stderr,
                )
                sys.exit(1)

            v3_records = summary.to_v3_format(
                model=args.model,
                quantization=args.quantization,
                git_sha=args.git_sha,
                workflow_run_id=args.workflow_run_id,
                workflow_run_url=args.workflow_run_url,
                gpu_name=args.gpu_name if args.gpu_name else "UNKNOWN GPU",
                cuda_driver_version=(
                    args.cuda_driver_version if args.cuda_driver_version else "cuda"
                ),
            )

            with open(args.output_v3, "w") as f:
                json.dump(v3_records, f, indent=2)

            print(f"✓ v3 format results saved to: {args.output_v3}")
            print(f"✓ Generated {len(v3_records)} v3 records for dashboard upload")

    finally:
        # Reset GPU clocks if they were fixed
        if gpu_clock_fixed:
            reset_gpu_clocks()


if __name__ == "__main__":
    main()
