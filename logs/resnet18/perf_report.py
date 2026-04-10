#!/usr/bin/env python3
"""
ResNet-18 Performance Report: Generic vs Optimized (Vision) builds.

Generates a full performance summary with per-layer timing, per-operator
aggregate, and overall speedup.

Usage:
    python perf_report.py [generic_log] [optimized_log]
"""

import re
import sys
import os
from collections import defaultdict

# ── Op name normalization ────────────────────────────────────────────
OP_NORMALIZE = {
    "quantize_per_tensor": "quantize",
    "quantized_conv2d_nchw": "conv2d",
    "quantized_relu_per_tensor": "relu",
    "dequantize_per_tensor": "dequantize",
    "max_pool2d": "maxpool",
    "add": "add",
    "mean": "mean",
    "quantized_linear_per_tensor": "linear",
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


def parse_perf_lines(logfile):
    """Parse PERF_LOG lines. Format:
    PERF_LOG : <op> : <numel> : <unit> : <cycles> : cycles : <elem/cyc> : <unit>/cycle : <cyc/elem> : cycles/<unit>
    """
    pattern = re.compile(
        r'PERF_LOG\s*:\s*(\S+)\s*:\s*(\d+)\s*:\s*[^:]+\s*:\s*(\d+)\s*:\s*cycles'
    )
    layers = []
    with open(logfile, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                op_raw = m.group(1)
                numel = int(m.group(2))
                cycles = int(m.group(3))
                layers.append({
                    'op_raw': op_raw,
                    'op': normalize_op(op_raw),
                    'numel': numel,
                    'cycles': cycles,
                })
    return layers


def parse_dump_lines(logfile):
    """Parse LAYER_DUMP for correctness summary."""
    pattern = re.compile(
        r'LAYER_DUMP\s*:\s*(\S+)\s*:\s*(\d+)\s*:\s*dtype=(\d+)\s*:'
        r'\s*first=\[([^\]]*)\]\s*:\s*sum=([^\s:]+)\s*:\s*min=([^\s:]+)\s*:\s*max=([^\s:]+)'
    )
    layers = []
    with open(logfile, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                layers.append({
                    'op': normalize_op(m.group(1)),
                    'numel': int(m.group(2)),
                    'dtype': int(m.group(3)),
                    'first': [float(x) for x in m.group(4).split(',') if x.strip()],
                    'sum': float(m.group(5)),
                    'min': float(m.group(6)),
                    'max': float(m.group(7)),
                })
    return layers


def fmt_cycles(c):
    if c >= 1_000_000:
        return f"{c/1_000_000:.2f}M"
    elif c >= 1_000:
        return f"{c/1_000:.1f}K"
    return str(c)


def report(generic_log, optimized_log):
    g_perf = parse_perf_lines(generic_log)
    o_perf = parse_perf_lines(optimized_log)
    g_dump = parse_dump_lines(generic_log)
    o_dump = parse_dump_lines(optimized_log)

    n_perf = min(len(g_perf), len(o_perf))
    n_dump = min(len(g_dump), len(o_dump))

    # ── Header ───────────────────────────────────────────────────────
    print("=" * 100)
    print("  ResNet-18 Performance Report: Generic vs Vision-Optimized")
    print("=" * 100)
    print(f"  Generic log:   {os.path.basename(generic_log)}")
    print(f"  Optimized log: {os.path.basename(optimized_log)}")
    print(f"  Layers (perf): {len(g_perf)} generic, {len(o_perf)} optimized")
    print()

    # ── Per-layer table ──────────────────────────────────────────────
    print("─" * 100)
    print(f"{'#':>3} {'Op':>12} {'Numel':>7} │ {'Generic':>12} {'Optimized':>12} {'Speedup':>8} │ {'ΔSum':>12} {'Status':>10}")
    print("─" * 100)

    # Aggregate accumulators
    op_generic = defaultdict(lambda: {'cycles': 0, 'count': 0, 'numel': 0})
    op_optim = defaultdict(lambda: {'cycles': 0, 'count': 0, 'numel': 0})
    total_g = 0
    total_o = 0

    correctness_match = 0
    correctness_minor = 0
    correctness_diverge = 0

    for i in range(n_perf):
        g = g_perf[i]
        o = o_perf[i]
        op = g['op']

        gc = g['cycles']
        oc = o['cycles']
        speedup = gc / oc if oc > 0 else float('inf')

        total_g += gc
        total_o += oc
        op_generic[op]['cycles'] += gc
        op_generic[op]['count'] += 1
        op_generic[op]['numel'] += g['numel']
        op_optim[op]['cycles'] += oc
        op_optim[op]['count'] += 1
        op_optim[op]['numel'] += o['numel']

        # Correctness from dumps
        delta_sum = ""
        status = ""
        if i < n_dump:
            gd = g_dump[i]
            od = o_dump[i]
            ds = od['sum'] - gd['sum']
            delta_sum = f"{ds:>12.1f}"

            is_int = gd['dtype'] in (0, 1)
            nfirst = min(len(gd['first']), len(od['first']))
            max_elem_diff = max((abs(od['first'][j] - gd['first'][j]) for j in range(nfirst)), default=0)

            if is_int:
                if abs(ds) == 0 and max_elem_diff == 0:
                    status = "MATCH"
                    correctness_match += 1
                elif max_elem_diff <= 2 and abs(ds) < gd['numel'] * 0.05:
                    status = "~ok"
                    correctness_minor += 1
                else:
                    status = "~drift"
                    correctness_diverge += 1
            else:
                range_g = gd['max'] - gd['min'] if gd['max'] != gd['min'] else max(abs(gd['max']), 1.0)
                rel_sum = abs(ds) / max(abs(gd['sum']), 1e-9)
                if rel_sum < 1e-4:
                    status = "MATCH"
                    correctness_match += 1
                elif rel_sum < 0.01:
                    status = "~ok"
                    correctness_minor += 1
                else:
                    status = "~drift"
                    correctness_diverge += 1

        spd_str = f"{speedup:.1f}x"
        print(f"{i:3d} {op:>12} {g['numel']:>7} │ {fmt_cycles(gc):>12} {fmt_cycles(oc):>12} {spd_str:>8} │ {delta_sum:>12} {status:>10}")

    print("─" * 100)

    # ── Total ────────────────────────────────────────────────────────
    total_speedup = total_g / total_o if total_o > 0 else 0
    print(f"{'':>3} {'TOTAL':>12} {'':>7} │ {fmt_cycles(total_g):>12} {fmt_cycles(total_o):>12} {total_speedup:.1f}x │")
    print()

    # ── Per-operator aggregate ───────────────────────────────────────
    print("=" * 100)
    print("  Per-Operator Aggregate")
    print("=" * 100)
    print(f"{'Operator':>14} {'Count':>6} │ {'Generic Total':>14} {'Optimized Total':>16} {'Speedup':>8} │ {'Gen cyc/elem':>13} {'Opt cyc/elem':>13}")
    print("─" * 100)

    all_ops = sorted(set(list(op_generic.keys()) + list(op_optim.keys())),
                     key=lambda x: op_generic[x]['cycles'], reverse=True)

    for op in all_ops:
        g = op_generic[op]
        o = op_optim[op]
        spd = g['cycles'] / o['cycles'] if o['cycles'] > 0 else float('inf')
        g_cpe = g['cycles'] / g['numel'] if g['numel'] > 0 else 0
        o_cpe = o['cycles'] / o['numel'] if o['numel'] > 0 else 0
        print(f"{op:>14} {g['count']:>6} │ {fmt_cycles(g['cycles']):>14} {fmt_cycles(o['cycles']):>16} {spd:.1f}x │ {g_cpe:>13.2f} {o_cpe:>13.2f}")

    print("─" * 100)
    print(f"{'TOTAL':>14} {n_perf:>6} │ {fmt_cycles(total_g):>14} {fmt_cycles(total_o):>16} {total_speedup:.1f}x │")
    print()

    # ── Correctness summary ──────────────────────────────────────────
    print("=" * 100)
    print("  Correctness Summary")
    print("=" * 100)
    print(f"  MATCH (identical or <0.01% relative diff):  {correctness_match}")
    print(f"  ~ok   (minor rounding, <1% relative diff):  {correctness_minor}")
    print(f"  ~drift (accumulating quantization noise):   {correctness_diverge}")
    total_c = correctness_match + correctness_minor + correctness_diverge
    if total_c > 0:
        pct_ok = 100.0 * (correctness_match + correctness_minor) / total_c
        print(f"  Layers with acceptable accuracy:            {correctness_match + correctness_minor}/{total_c} ({pct_ok:.1f}%)")
    print()

    # ── Key findings ─────────────────────────────────────────────────
    print("=" * 100)
    print("  Key Findings")
    print("=" * 100)

    # Find biggest speedup op
    best_op = max(all_ops, key=lambda x: op_generic[x]['cycles'] / max(op_optim[x]['cycles'], 1))
    best_spd = op_generic[best_op]['cycles'] / max(op_optim[best_op]['cycles'], 1)
    print(f"  Overall speedup:          {total_speedup:.1f}x ({fmt_cycles(total_g)} → {fmt_cycles(total_o)} cycles)")
    print(f"  Best per-op speedup:      {best_op} at {best_spd:.1f}x")

    # Dominant op by generic cycles
    dom_op = max(all_ops, key=lambda x: op_generic[x]['cycles'])
    dom_pct = 100.0 * op_generic[dom_op]['cycles'] / total_g
    print(f"  Dominant op (generic):    {dom_op} = {dom_pct:.1f}% of total cycles")

    dom_op_o = max(all_ops, key=lambda x: op_optim[x]['cycles'])
    dom_pct_o = 100.0 * op_optim[dom_op_o]['cycles'] / total_o
    print(f"  Dominant op (optimized):  {dom_op_o} = {dom_pct_o:.1f}% of total cycles")

    # Clock estimate (assume 1GHz for cycle→time)
    print(f"\n  Estimated inference time @ 1GHz:")
    print(f"    Generic:   {total_g / 1e6:.2f} ms")
    print(f"    Optimized: {total_o / 1e6:.2f} ms")
    print("=" * 100)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) >= 3:
        gen_log, opt_log = sys.argv[1], sys.argv[2]
    else:
        gen_log = os.path.join(script_dir, 'resnet18_cache_generic.log')
        opt_log = os.path.join(script_dir, 'resnet18_cache_quant2.log')

    if not os.path.exists(gen_log):
        print(f"ERROR: {gen_log} not found"); sys.exit(1)
    if not os.path.exists(opt_log):
        print(f"ERROR: {opt_log} not found"); sys.exit(1)

    report(gen_log, opt_log)
