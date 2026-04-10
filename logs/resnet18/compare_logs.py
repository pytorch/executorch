#!/usr/bin/env python3
"""
Compare ResNet18 generic vs optimized (vision) builds.

Layer-by-layer performance comparison + output correctness.
Based on compare_resnet18_output.py from cad_rlc.

Usage:
    python compare_logs.py <generic_log> <optimized_log>
    python compare_logs.py   # uses default log filenames in same directory
"""

import re
import sys
import os
from collections import Counter


# ── Operator name normalization ──────────────────────────────────────────
# Generic and vision builds use different names for the same logical op.
# Normalize to a common name for layer-by-layer matching.
OP_NORMALIZE = {
    # generic names → canonical
    "quantize_per_tensor": "quantize",
    "quantized_conv2d_nchw": "conv2d",
    "quantized_relu_per_tensor": "relu",
    "dequantize_per_tensor": "dequantize",
    "max_pool2d": "maxpool",
    "add": "add",
    "mean": "mean",
    "quantized_linear_per_tensor": "linear",
    # vision names → canonical
    "quantize_asym8s": "quantize",
    "conv2d": "conv2d",
    "quantized_relu": "relu",
    "dequantize_asym8s": "dequantize",
    "maxpool": "maxpool",
    "add_float": "add",
    "add_generic": "add",
    "mean_simd_optimized": "mean",
    "mean_portable_fallback": "mean",
    "quantized_linear": "linear",
}


def normalize_op(name):
    return OP_NORMALIZE.get(name, name)


# ── PERF_LOG extraction ─────────────────────────────────────────────────
def extract_perf_entries(filepath):
    """Extract all PERF_LOG lines into structured dicts.

    Format: PERF_LOG : <op> : <elements> : <annotation> : <cycles> : cycles ...
    The annotations vary (e.g. "elements", "elements (DMA ping-pong)", "floats").
    """
    entries = []
    with open(filepath) as f:
        for line in f:
            # Match: PERF_LOG : <op_name> : <number> : <anything> : <number> : cycles
            m = re.match(
                r"PERF_LOG\s*:\s*(\S+)\s*:\s*(\d+)\s*:.*?:\s*(\d+)\s*:\s*cycles",
                line.strip(),
            )
            if m:
                entries.append({
                    "op_raw": m.group(1),
                    "op": normalize_op(m.group(1)),
                    "elements": int(m.group(2)),
                    "cycles": int(m.group(3)),
                })
    return entries


# ── Output tensor extraction ─────────────────────────────────────────────
def extract_output_tensor(filepath):
    with open(filepath) as f:
        content = f.read()
    pattern = r"Output\s+0:\s+tensor\(sizes=\[1,\s*1000\],\s*\[(.*?)\]\)"
    matches = list(re.finditer(pattern, content, re.DOTALL))
    if not matches:
        return []
    val_str = matches[-1].group(1)
    nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", val_str)
    return [float(v) for v in nums]


def detect_quant_step(values):
    sample = values[:min(100, len(values))]
    diffs = set()
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            d = abs(sample[i] - sample[j])
            if d > 1e-8:
                diffs.add(round(d, 8))
    return min(diffs) if diffs else 0.0724269


# ── Layer-by-layer performance comparison ────────────────────────────────
def print_layer_comparison(gen_entries, opt_entries):
    sep = "=" * 100
    print(f"\n{sep}")
    print("  Layer-by-Layer Performance Comparison (Generic vs Optimized)")
    print(sep)

    # Group by op type for summary
    gen_by_op = {}
    opt_by_op = {}
    for e in gen_entries:
        gen_by_op.setdefault(e["op"], []).append(e)
    for e in opt_entries:
        opt_by_op.setdefault(e["op"], []).append(e)

    # Print sequential layer-by-layer comparison
    max_layers = max(len(gen_entries), len(opt_entries))

    print(f"\n{'#':>3}  {'Op (Generic)':<30} {'Elem':>8} {'Cycles':>14} {'Op (Optimized)':<30} {'Elem':>8} {'Cycles':>14} {'Speedup':>9}")
    print("-" * 130)

    gi, oi = 0, 0
    layer_num = 0
    total_gen_cycles = 0
    total_opt_cycles = 0

    while gi < len(gen_entries) or oi < len(opt_entries):
        layer_num += 1
        gen_str = ""
        opt_str = ""
        gen_cyc = 0
        opt_cyc = 0

        if gi < len(gen_entries):
            ge = gen_entries[gi]
            gen_str = f"{ge['op_raw']:<30} {ge['elements']:>8} {ge['cycles']:>14,}"
            gen_cyc = ge['cycles']
            total_gen_cycles += gen_cyc
            gi += 1
        else:
            gen_str = f"{'—':<30} {'—':>8} {'—':>14}"

        if oi < len(opt_entries):
            oe = opt_entries[oi]
            opt_str = f"{oe['op_raw']:<30} {oe['elements']:>8} {oe['cycles']:>14,}"
            opt_cyc = oe['cycles']
            total_opt_cycles += opt_cyc
            oi += 1
        else:
            opt_str = f"{'—':<30} {'—':>8} {'—':>14}"

        if gen_cyc > 0 and opt_cyc > 0:
            speedup = gen_cyc / opt_cyc
            speedup_str = f"{speedup:>8.1f}x"
        else:
            speedup_str = f"{'—':>9}"

        print(f"{layer_num:>3}  {gen_str} {opt_str} {speedup_str}")

    print("-" * 130)
    print(f"{'':>3}  {'TOTAL':<30} {'':>8} {total_gen_cycles:>14,} {'TOTAL':<30} {'':>8} {total_opt_cycles:>14,}", end="")
    if total_opt_cycles > 0:
        print(f" {total_gen_cycles/total_opt_cycles:>8.1f}x")
    else:
        print()

    # ── Per-operator-type summary ────────────────────────────────────────
    all_ops = sorted(set(list(gen_by_op.keys()) + list(opt_by_op.keys())))
    print(f"\n{sep}")
    print("  Per-Operator Summary")
    print(sep)
    print(f"  {'Operator':<20} {'Generic':>14} {'# Calls':>8} {'Optimized':>14} {'# Calls':>8} {'Speedup':>9}")
    print(f"  {'-'*20} {'-'*14} {'-'*8} {'-'*14} {'-'*8} {'-'*9}")

    grand_gen = 0
    grand_opt = 0
    for op in all_ops:
        gen_list = gen_by_op.get(op, [])
        opt_list = opt_by_op.get(op, [])
        gc = sum(e["cycles"] for e in gen_list)
        oc = sum(e["cycles"] for e in opt_list)
        grand_gen += gc
        grand_opt += oc
        gcnt = len(gen_list)
        ocnt = len(opt_list)
        if gc > 0 and oc > 0:
            sp = f"{gc/oc:.1f}x"
        elif gc > 0:
            sp = "N/A"
        elif oc > 0:
            sp = "new"
        else:
            sp = "—"
        print(f"  {op:<20} {gc:>14,} {gcnt:>8} {oc:>14,} {ocnt:>8} {sp:>9}")

    print(f"  {'-'*20} {'-'*14} {'-'*8} {'-'*14} {'-'*8} {'-'*9}")
    if grand_opt > 0:
        print(f"  {'TOTAL':<20} {grand_gen:>14,} {'':>8} {grand_opt:>14,} {'':>8} {grand_gen/grand_opt:>8.1f}x")
    else:
        print(f"  {'TOTAL':<20} {grand_gen:>14,}")


# ── Correctness comparison ───────────────────────────────────────────────
def print_correctness(gen_file, opt_file, gen_vals, opt_vals):
    sep = "=" * 100
    print(f"\n{sep}")
    print("  Output Correctness Comparison (Generic vs Optimized)")
    print(sep)

    if not gen_vals:
        print("  ERROR: No output tensor found in generic log")
        return
    if not opt_vals:
        print("  ERROR: No output tensor found in optimized log")
        return

    n = min(len(gen_vals), len(opt_vals))
    print(f"  Output size  : {n} values")

    quant_step = detect_quant_step(gen_vals)
    print(f"  Quant step   : {quant_step:.7f}")

    abs_diffs = [abs(gen_vals[i] - opt_vals[i]) for i in range(n)]
    buckets = Counter()
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
    mean_diff = sum(abs_diffs) / n

    print(f"\n  {'Category':<22} {'Count':>8} {'Percent':>9}")
    print(f"  {'-'*42}")
    for label, key in [
        ("Exact match", "exact"),
        ("Off-by-1 level", "off_1"),
        ("Off-by-2 levels", "off_2"),
        (">2 levels", "worse"),
    ]:
        cnt = buckets.get(key, 0)
        print(f"  {label:<22} {cnt:>8} {100*cnt/n:>8.1f}%")
    print(f"  {'Total':<22} {n:>8}")

    print(f"\n  Max absolute diff : {max_diff:.7f}  "
          f"({max_diff/quant_step:.1f} levels)  at index {max_idx}")
    print(f"  Mean absolute diff: {mean_diff:.7f}  "
          f"({mean_diff/quant_step:.2f} levels)")

    # Top-10 worst
    top10 = sorted(enumerate(abs_diffs), key=lambda x: -x[1])[:10]
    print(f"\n  Top 10 largest differences:")
    for idx, d in top10:
        lvls = d / quant_step
        print(f"    [{idx:>4}]  gen={gen_vals[idx]:>10.6f}  "
              f"opt={opt_vals[idx]:>10.6f}  diff={d:.6f} ({lvls:.1f} levels)")

    # Top-K classification
    gen1k = gen_vals[:1000]
    opt1k = opt_vals[:1000]
    gen_top5 = sorted(range(len(gen1k)), key=lambda i: -gen1k[i])[:5]
    opt_top5 = sorted(range(len(opt1k)), key=lambda i: -opt1k[i])[:5]
    overlap = len(set(gen_top5) & set(opt_top5))

    print(f"\n  {'-'*60}")
    print(f"  Top-K Classification Comparison")
    print(f"  {'-'*60}")
    print(f"  Generic   Top-1: class {gen_top5[0]:>4}  (score {gen1k[gen_top5[0]]:.4f})")
    print(f"  Optimized Top-1: class {opt_top5[0]:>4}  (score {opt1k[opt_top5[0]]:.4f})")
    print(f"  Top-1 match: {'YES' if gen_top5[0] == opt_top5[0] else 'NO'}")
    print(f"  Generic   Top-5: {gen_top5}")
    print(f"  Optimized Top-5: {opt_top5}")
    print(f"  Top-5 overlap : {overlap}/5")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) >= 3:
        gen_file = sys.argv[1]
        opt_file = sys.argv[2]
    else:
        gen_file = os.path.join(script_dir, "resnet18_cache_generic.log")
        opt_file = os.path.join(script_dir, "resnet18_cache_quant.log")

    for f in [gen_file, opt_file]:
        if not os.path.exists(f):
            print(f"ERROR: File not found: {f}")
            sys.exit(1)

    print(f"Generic log  : {os.path.basename(gen_file)}")
    print(f"Optimized log: {os.path.basename(opt_file)}")

    # Performance comparison
    gen_perf = extract_perf_entries(gen_file)
    opt_perf = extract_perf_entries(opt_file)
    if gen_perf and opt_perf:
        print_layer_comparison(gen_perf, opt_perf)

    # Correctness comparison
    gen_vals = extract_output_tensor(gen_file)
    opt_vals = extract_output_tensor(opt_file)
    print_correctness(gen_file, opt_file, gen_vals, opt_vals)

    print()


if __name__ == "__main__":
    main()
