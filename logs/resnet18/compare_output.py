#!/usr/bin/env python3
"""
ResNet-18 Final Output Comparison: Generic vs Optimized.

Compares the 1000-class output logits element-by-element.
Reports:
  - Per-level mismatch histogram (quantized int8)
  - Top-5 class comparison
  - Overall accuracy metrics
"""

import re
import sys
import os
import math


def extract_output_tensor(logfile):
    """Extract the final 'Output 0: tensor(sizes=[1, 1000], [...]' from log."""
    with open(logfile, 'r') as f:
        content = f.read()

    # Find "Output 0: tensor(sizes=[1, 1000], [" and extract all values until closing ]
    m = re.search(r'Output 0:\s*tensor\(sizes=\[1,\s*1000\],\s*\[(.*?)\]\)', content, re.DOTALL)
    if not m:
        return None
    raw = m.group(1)
    # Parse all float values
    vals = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', raw)]
    return vals


def extract_quantized_linear(logfile):
    """Extract the quantized linear output (int8, 1000 elements) from LAYER_DUMP."""
    with open(logfile, 'r') as f:
        for line in f:
            if 'LAYER_DUMP' in line and ('quantized_linear' in line or 'linear' in line) and '1000' in line:
                m = re.search(r'first=\[([^\]]*)\]', line)
                if m:
                    return [int(float(x)) for x in m.group(1).split(',') if x.strip()]
    return None


def top_k_indices(vals, k=5):
    """Return indices of top-k values."""
    indexed = sorted(enumerate(vals), key=lambda x: x[1], reverse=True)
    return [idx for idx, val in indexed[:k]]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) >= 3:
        gen_log, opt_log = sys.argv[1], sys.argv[2]
    else:
        gen_log = os.path.join(script_dir, 'resnet18_cache_generic.log')
        opt_log = os.path.join(script_dir, 'resnet18_cache_quant2.log')

    for f in (gen_log, opt_log):
        if not os.path.exists(f):
            print(f"ERROR: {f} not found"); sys.exit(1)

    # ── Extract float output (dequantized logits) ────────────────────
    g_vals = extract_output_tensor(gen_log)
    o_vals = extract_output_tensor(opt_log)

    if not g_vals or not o_vals:
        print("ERROR: Could not extract Output 0 tensor")
        sys.exit(1)

    n = min(len(g_vals), len(o_vals))
    print(f"Extracted {len(g_vals)} generic values, {len(o_vals)} optimized values")
    print()

    # ── Float-level comparison ───────────────────────────────────────
    print("=" * 80)
    print("  Float Output Comparison (dequantized logits, 1000 classes)")
    print("=" * 80)

    diffs = [abs(o_vals[i] - g_vals[i]) for i in range(n)]
    max_diff = max(diffs)
    avg_diff = sum(diffs) / n
    exact_match = sum(1 for d in diffs if d == 0.0)
    within_001 = sum(1 for d in diffs if d <= 0.01)
    within_01 = sum(1 for d in diffs if d <= 0.1)
    within_05 = sum(1 for d in diffs if d <= 0.5)
    within_1 = sum(1 for d in diffs if d <= 1.0)

    print(f"  Exact match (diff=0):    {exact_match:>5} / {n}")
    print(f"  Within 0.01:             {within_001:>5} / {n}")
    print(f"  Within 0.1:              {within_01:>5} / {n}")
    print(f"  Within 0.5:              {within_05:>5} / {n}")
    print(f"  Within 1.0:              {within_1:>5} / {n}")
    print(f"  Max absolute diff:       {max_diff:.6f}")
    print(f"  Avg absolute diff:       {avg_diff:.6f}")
    print()

    # ── Quantized level mismatch histogram ───────────────────────────
    # Estimate quant scale from LAYER_DUMP or compute from dequant scale
    # The linear output is int8 with some scale+zp. Use the float output
    # and the quantized linear output to infer scale.
    # Alternative: bucket by the dequantized step size
    
    # Try to get the quant parameters from the log
    # Look for the dequantize after linear to get scale
    with open(gen_log, 'r') as f:
        gen_content = f.read()
    with open(opt_log, 'r') as f:
        opt_content = f.read()

    # Extract quant scale from the pattern of values
    # Use the LAYER_DUMP for quantized_linear to get int8 values
    g_quant_m = re.search(r'LAYER_DUMP\s*:\s*quantized_linear_per_tensor\s*:\s*1000.*?first=\[([^\]]*)\].*?sum=([^\s:]+)', gen_content)
    o_quant_dq = re.search(r'LAYER_DUMP\s*:\s*dequantize_per_tensor\s*:\s*1000.*?first=\[([^\]]*)\]', opt_content)

    # Infer scale: if we have both int8 and float for generic, scale = float_val / (int8_val - zero_point)
    # For now, compute quant-level difference using the dequant scale
    # Estimate scale from first non-zero pair
    gen_scale = None
    if g_quant_m:
        g_int_first = [int(float(x)) for x in g_quant_m.group(1).split(',') if x.strip()]
        # First float value / first int value (if non-zero) gives approximate scale
        for j in range(min(len(g_int_first), len(g_vals))):
            if g_int_first[j] != 0 and abs(g_vals[j]) > 0.001:
                # float = scale * (int - zp), but we don't know zp
                # Use multiple points to estimate
                pass
        # Simpler: use the quant step = diff between consecutive unique float values
        unique_sorted = sorted(set(g_vals))
        steps = [unique_sorted[i+1] - unique_sorted[i] for i in range(len(unique_sorted)-1) if unique_sorted[i+1] - unique_sorted[i] > 1e-6]
        if steps:
            gen_scale = min(steps)

    if gen_scale and gen_scale > 0:
        print("=" * 80)
        print(f"  Quantized Level Mismatch (estimated quant step = {gen_scale:.6f})")
        print("=" * 80)

        level_diffs = [round(abs(o_vals[i] - g_vals[i]) / gen_scale) for i in range(n)]
        max_level = max(level_diffs) if level_diffs else 0

        # Histogram
        hist = {}
        for d in level_diffs:
            hist[d] = hist.get(d, 0) + 1

        print(f"  {'Level Diff':>12} {'Count':>8} {'Percentage':>12} {'Cumulative':>12}")
        print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*12}")
        
        cumulative = 0
        for level in sorted(hist.keys()):
            count = hist[level]
            pct = 100.0 * count / n
            cumulative += count
            cum_pct = 100.0 * cumulative / n
            print(f"  {level:>12} {count:>8} {pct:>11.1f}% {cum_pct:>11.1f}%")
            if level > 10 and cum_pct > 99.5:
                remaining = n - cumulative
                if remaining > 0:
                    print(f"  {'>' + str(level):>12} {remaining:>8} {100.0*remaining/n:>11.1f}%")
                break
        
        print(f"\n  Max level difference: {max_level}")
        print(f"  Exact match (0 levels): {hist.get(0, 0)} / {n} ({100.0*hist.get(0,0)/n:.1f}%)")
        within_1_level = hist.get(0, 0) + hist.get(1, 0)
        print(f"  Within 1 level: {within_1_level} / {n} ({100.0*within_1_level/n:.1f}%)")
        within_2_level = within_1_level + hist.get(2, 0)
        print(f"  Within 2 levels: {within_2_level} / {n} ({100.0*within_2_level/n:.1f}%)")
    print()

    # ── Top-5 class comparison ───────────────────────────────────────
    print("=" * 80)
    print("  Top-5 Classification Comparison")
    print("=" * 80)

    g_top5 = top_k_indices(g_vals[:n], 5)
    o_top5 = top_k_indices(o_vals[:n], 5)
    g_top10 = top_k_indices(g_vals[:n], 10)
    o_top10 = top_k_indices(o_vals[:n], 10)

    print(f"\n  {'Rank':>5} │ {'Generic':>30} │ {'Optimized':>30} │ {'Match'}")
    print(f"  {'':>5} │ {'Class':>8} {'Logit':>10} │ {'Class':>8} {'Logit':>10} │")
    print(f"  {'-'*5} │ {'-'*30} │ {'-'*30} │ {'-'*5}")

    top5_match = 0
    for rank in range(5):
        gi = g_top5[rank]
        oi = o_top5[rank]
        match = "✓" if gi == oi else ""
        if gi == oi:
            top5_match += 1
        print(f"  {rank+1:>5} │ {gi:>8} {g_vals[gi]:>10.4f} │ {oi:>8} {o_vals[oi]:>10.4f} │ {match}")

    # Check overlap (order-independent)
    g_top5_set = set(g_top5)
    o_top5_set = set(o_top5)
    top5_overlap = len(g_top5_set & o_top5_set)

    g_top10_set = set(g_top10)
    o_top10_set = set(o_top10)
    top10_overlap = len(g_top10_set & o_top10_set)

    print(f"\n  Top-1 match:                {'YES' if g_top5[0] == o_top5[0] else 'NO'} (generic={g_top5[0]}, optimized={o_top5[0]})")
    print(f"  Top-5 rank-exact match:     {top5_match}/5")
    print(f"  Top-5 overlap (any order):  {top5_overlap}/5 — classes {g_top5_set & o_top5_set}")
    print(f"  Top-10 overlap (any order): {top10_overlap}/10 — classes {g_top10_set & o_top10_set}")

    # Is generic top-1 in optimized top-5?
    g_top1 = g_top5[0]
    g_top1_in_o_top5 = g_top1 in o_top5_set
    g_top1_in_o_top10 = g_top1 in o_top10_set
    print(f"  Generic top-1 (class {g_top1}) in optimized top-5:  {'YES' if g_top1_in_o_top5 else 'NO'}")
    print(f"  Generic top-1 (class {g_top1}) in optimized top-10: {'YES' if g_top1_in_o_top10 else 'NO'}")

    # Logit diff for top-1
    print(f"\n  Top-1 logit diff: {abs(o_vals[g_top1] - g_vals[g_top1]):.6f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
