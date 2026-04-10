#!/usr/bin/env python3
"""
Compare ResNet18 output between optimized and generic builds.

Parses the final [1, 1000] output tensor from executor_runner log files
and reports absolute-difference metrics. Absolute difference is the correct
metric here because outputs are dequantized int8 values — ULP (units in
last place) measures floating-point representation precision, which is not
meaningful for quantized data.

Metrics
-------
Quant step (dequantization scale):
    The smallest nonzero absolute difference between any two output values.
    In a uniformly quantized tensor, every value is an integer multiple of
    this step. All difference metrics below are expressed both in raw float
    units and in multiples of this step ("levels").

    Background — quantization levels:
    The model's final linear layer produces int8 logits which are then
    dequantized: float_value = (int8_value - zero_point) * scale.
    Because int8 can only represent 256 discrete values, the float output
    is a discrete grid with spacing = scale (the "quant step"). Any
    difference between generic and optimized outputs must therefore be
    an integer multiple of this step. We call each step a "level":
      - 0 levels = exact match (same int8 value)
      - 1 level  = adjacent int8 values (off by 1 in the integer domain)
      - N levels = off by N in the integer domain
    For ResNet18 with the current quantization config, the step is
    ~0.0724 (i.e. scale ≈ 0.0724), so 1 level ≈ 0.07 float units.

Exact match (0 levels):
    Number of output elements where |generic - optimized| < 0.5 levels.
    These values produced the identical int8 result in both paths.

Off-by-1 level:
    Elements differing by >= 0.5 and < 1.5 quantization levels.
    A 1-level difference is the minimum possible nonzero error for int8
    quantized outputs. It is expected due to rounding differences in the
    fixed-point PACK pipeline (accumShift + outputScale + outputShift)
    versus the generic path's float-based round-and-clamp. In practice,
    even running the same int8 model on different hardware (x86 vs ARM)
    can produce 1-level differences from rounding alone.

Off-by-2 levels:
    Elements differing by >= 1.5 and < 2.5 quantization levels.
    These arise from cumulative rounding across the bias-correction,
    24-bit clamping, and post-kernel residual steps. Still acceptable
    for production — a 2-level error is ~0.14 float units, negligible
    relative to the full output range (typically tens of units).

>2 levels:
    Elements differing by more than 2.5 quantization levels.
    A small count (< ~5% of outputs) with max ~5 levels is acceptable.
    If this count is high or max levels are large (> 10), investigate
    potential accumulator overflow, double-clamp issues, or 24-bit
    wrapping in deep layers.

Max absolute diff:
    Largest element-wise |generic - optimized| across all 1000 outputs,
    reported in both float units and quantization levels.

Mean absolute diff:
    Average element-wise |generic - optimized|. A value near or below
    1.0 level indicates the optimized kernel is numerically faithful.

Top-10 largest differences:
    The 10 output indices with the greatest absolute difference, useful
    for identifying systematic bias in specific class logits.

Top-K classification:
    Compares the predicted class rankings (Top-1 and Top-5) between
    generic and optimized. The 1000 output values are ImageNet class
    logits — the model's confidence score for each of 1000 categories.

    - Top-1 match: Both builds predict the same highest-scoring class.
      This is the primary correctness check. If Top-1 disagrees, the
      optimized kernel has changed the model's prediction.

    - Top-5 overlap: How many of the 5 highest-scoring classes appear
      in both outputs (out of 5). Classes ranked 3rd–5th often have
      very similar scores, so a single quantization level of difference
      can swap their order. 4/5 or 5/5 overlap is expected and good.
      3/5 or below warrants investigation.

    Why this matters more than raw numerical diff:
    A max diff of 5 levels sounds large, but if the top class scores
    9.6 vs 9.9 while the runner-up scores 6.8, the ranking is stable.
    Classification accuracy is what end users care about — small
    numerical noise that doesn't change the predicted class is harmless.

Performance (PERF_LOG cycles):
    Total conv2d execution cycles summed across all 20 ResNet18 conv
    layers, extracted from the PERF_LOG lines in each log file.
    Speedup = generic_cycles / optimized_cycles. This is only accurate
    when both logs were captured without debug instrumentation.

Usage:
    python compare_resnet18_output.py <generic_log> <optimized_log>

Example:
    python compare_resnet18_output.py \
        output/resnet18_quantized_conv_generic_output_turbo.log \
        output/resnet18_quantized_conv_opt_output_turbo.log
"""

import re
import sys
import os
from collections import Counter


def extract_output_tensor(filepath: str) -> list[float]:
    """Extract the final [1, 1000] output tensor from an executor_runner log.

    The tensor is printed as:
        Output 0: tensor(sizes=[1, 1000], [
          v0, v1, v2, ...
        ])
    """
    with open(filepath) as f:
        content = f.read()

    pattern = r"Output\s+0:\s+tensor\(sizes=\[1,\s*1000\],\s*\[(.*?)\]\)"
    matches = list(re.finditer(pattern, content, re.DOTALL))
    if not matches:
        print(f"ERROR: no [1,1000] output tensor found in {filepath}")
        return []

    # Use the last match (final model output)
    val_str = matches[-1].group(1)
    nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", val_str)
    return [float(v) for v in nums]


def detect_quant_step(values: list[float]) -> float:
    """Detect the quantization step size from a set of dequantized values.

    In a uniformly quantized tensor, the smallest nonzero absolute difference
    between any two values equals the dequantization scale.
    """
    sample = values[: min(100, len(values))]
    diffs = set()
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            d = abs(sample[i] - sample[j])
            if d > 1e-8:
                diffs.add(round(d, 8))
    return min(diffs) if diffs else 0.0724269


def compare_outputs(
    gen_vals: list[float],
    opt_vals: list[float],
    quant_step: float,
) -> dict:
    """Compute comparison metrics between generic and optimized outputs."""
    n = min(len(gen_vals), len(opt_vals))
    abs_diffs = [abs(gen_vals[i] - opt_vals[i]) for i in range(n)]

    # Bucket by quantization-level distance
    buckets: Counter = Counter()
    for d in abs_diffs:
        levels = d / quant_step
        if levels < 0.5:
            buckets["exact"] += 1
        elif levels < 1.5:
            buckets["off_1"] += 1
        elif levels < 2.5:
            buckets["off_2"] += 1
        else:
            buckets["worse"] += 1

    max_diff = max(abs_diffs)
    max_idx = abs_diffs.index(max_diff)
    mean_diff = sum(abs_diffs) / n if n else 0.0

    # Top-10 worst
    top10 = sorted(enumerate(abs_diffs), key=lambda x: -x[1])[:10]

    return {
        "n": n,
        "quant_step": quant_step,
        "buckets": buckets,
        "max_diff": max_diff,
        "max_idx": max_idx,
        "mean_diff": mean_diff,
        "abs_diffs": abs_diffs,
        "top10": top10,
    }


def top_k_accuracy(gen: list[float], opt: list[float], k: int = 5):
    """Compare top-k predicted classes between generic and optimized."""
    gen_topk = sorted(range(len(gen)), key=lambda i: -gen[i])[:k]
    opt_topk = sorted(range(len(opt)), key=lambda i: -opt[i])[:k]
    overlap = len(set(gen_topk) & set(opt_topk))
    return gen_topk, opt_topk, overlap


def extract_perf_cycles(filepath: str) -> list[dict]:
    """Extract PERF_LOG entries from a log file."""
    entries = []
    with open(filepath) as f:
        for line in f:
            m = re.search(
                r"PERF_LOG\s*:\s*(\S+)\s*:\s*(\d+)\s*:\s*\S+.*?:\s*(\d+)\s*:\s*cycles",
                line,
            )
            if m:
                entries.append(
                    {"op": m.group(1), "elements": int(m.group(2)), "cycles": int(m.group(3))}
                )
    return entries


def print_report(
    gen_file: str,
    opt_file: str,
    gen_vals: list[float],
    opt_vals: list[float],
    metrics: dict,
):
    """Print a formatted comparison report."""
    sep = "=" * 60
    print(sep)
    print("  ResNet18 Output Comparison: Generic vs Optimized")
    print(sep)
    print(f"  Generic log : {os.path.basename(gen_file)}")
    print(f"  Optimized log: {os.path.basename(opt_file)}")
    print(f"  Output size  : {metrics['n']} values")
    print(f"  Quant step   : {metrics['quant_step']:.7f}")
    print(sep)

    # --- Absolute-difference summary ---
    n = metrics["n"]
    b = metrics["buckets"]
    print(f"\n{'Category':<22} {'Count':>8} {'Percent':>9}")
    print("-" * 42)
    for label, key in [
        ("Exact match", "exact"),
        ("Off-by-1 level", "off_1"),
        ("Off-by-2 levels", "off_2"),
        (">2 levels", "worse"),
    ]:
        cnt = b.get(key, 0)
        print(f"{label:<22} {cnt:>8} {100 * cnt / n:>8.1f}%")
    print(f"{'Total':<22} {n:>8}")

    print(f"\n  Max absolute diff : {metrics['max_diff']:.7f}  "
          f"({metrics['max_diff'] / metrics['quant_step']:.1f} levels)  "
          f"at index {metrics['max_idx']}")
    print(f"  Mean absolute diff: {metrics['mean_diff']:.7f}  "
          f"({metrics['mean_diff'] / metrics['quant_step']:.2f} levels)")

    # --- Top-10 worst ---
    print(f"\n  Top 10 largest differences:")
    for idx, d in metrics["top10"]:
        lvls = d / metrics["quant_step"]
        print(
            f"    [{idx:>4}]  gen={gen_vals[idx]:>10.6f}  "
            f"opt={opt_vals[idx]:>10.6f}  diff={d:.6f} ({lvls:.1f} levels)"
        )

    # --- Top-K classification ---
    gen1k = gen_vals[:1000]
    opt1k = opt_vals[:1000]
    gen_top5, opt_top5, overlap = top_k_accuracy(gen1k, opt1k, k=5)
    print(f"\n{sep}")
    print("  Top-K Classification Comparison")
    print(sep)
    print(f"  Generic  Top-1: class {gen_top5[0]:>4}  (score {gen1k[gen_top5[0]]:.4f})")
    print(f"  Optimized Top-1: class {opt_top5[0]:>4}  (score {opt1k[opt_top5[0]]:.4f})")
    print(f"  Top-1 match: {'YES' if gen_top5[0] == opt_top5[0] else 'NO'}")
    print(f"  Generic  Top-5: {gen_top5}")
    print(f"  Optimized Top-5: {opt_top5}")
    print(f"  Top-5 overlap : {overlap}/5")

    # --- Performance (if available) ---
    gen_perf = extract_perf_cycles(gen_file)
    opt_perf = extract_perf_cycles(opt_file)
    if gen_perf and opt_perf:
        print(f"\n{sep}")
        print("  Performance (PERF_LOG cycles)")
        print(sep)
        gen_conv = [e for e in gen_perf if "conv2d" in e["op"]]
        opt_conv = [e for e in opt_perf if "conv2d" in e["op"]]
        gen_total = sum(e["cycles"] for e in gen_conv)
        opt_total = sum(e["cycles"] for e in opt_conv)
        if opt_total > 0:
            speedup = gen_total / opt_total
            print(f"  Generic  conv2d total: {gen_total:>12,} cycles  ({len(gen_conv)} layers)")
            print(f"  Optimized conv2d total: {opt_total:>12,} cycles  ({len(opt_conv)} layers)")
            print(f"  Speedup: {speedup:.1f}x")

    print(f"\n{sep}")


def main():
    if len(sys.argv) < 3:
        # Default paths relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gen_file = os.path.join(script_dir, "output", "resnet18_quantized_conv_generic_output_turbo.log")
        opt_file = os.path.join(script_dir, "output", "resnet18_quantized_conv_opt_output_turbo.log")
        if not os.path.exists(gen_file) or not os.path.exists(opt_file):
            print(__doc__)
            sys.exit(1)
        print(f"Using default log files (pass paths as arguments to override).\n")
    else:
        gen_file = sys.argv[1]
        opt_file = sys.argv[2]

    gen_vals = extract_output_tensor(gen_file)
    opt_vals = extract_output_tensor(opt_file)
    if not gen_vals or not opt_vals:
        sys.exit(1)

    quant_step = detect_quant_step(gen_vals)
    metrics = compare_outputs(gen_vals, opt_vals, quant_step)
    print_report(gen_file, opt_file, gen_vals, opt_vals, metrics)


if __name__ == "__main__":
    main()
