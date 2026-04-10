#!/usr/bin/env python3
"""
Compare LAYER_DUMP output between generic and optimized (vision) builds.
Identifies the first layer where output diverges.

Usage:
    python compare_dumps.py [generic_log] [optimized_log]
"""

import re
import sys
import os
import math

# ── Op name normalization (same mapping as compare_logs.py) ──────────
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


def parse_dump_lines(logfile):
    """Parse LAYER_DUMP lines from a log file."""
    # LAYER_DUMP : <op> : <numel> : dtype=<d> : first=[...] : sum=<s> : min=<lo> : max=<hi>
    pattern = re.compile(
        r'LAYER_DUMP\s*:\s*(\S+)\s*:\s*(\d+)\s*:\s*dtype=(\d+)\s*:'
        r'\s*first=\[([^\]]*)\]\s*:\s*sum=([^\s:]+)\s*:\s*min=([^\s:]+)\s*:\s*max=([^\s:]+)'
    )
    layers = []
    with open(logfile, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                op_raw = m.group(1)
                numel = int(m.group(2))
                dtype = int(m.group(3))
                first_vals = [float(x) for x in m.group(4).split(',') if x.strip()]
                sum_val = float(m.group(5))
                min_val = float(m.group(6))
                max_val = float(m.group(7))
                layers.append({
                    'op_raw': op_raw,
                    'op': normalize_op(op_raw),
                    'numel': numel,
                    'dtype': dtype,
                    'first': first_vals,
                    'sum': sum_val,
                    'min': min_val,
                    'max': max_val,
                })
    return layers


def fmt_vals(vals, n=8):
    """Format first n values for display."""
    show = vals[:n]
    if all(v == int(v) for v in show):
        return ','.join(str(int(v)) for v in show)
    return ','.join(f'{v:.4f}' for v in show)


def compare(generic_log, optimized_log):
    g_layers = parse_dump_lines(generic_log)
    o_layers = parse_dump_lines(optimized_log)

    print(f"Generic layers:   {len(g_layers)}")
    print(f"Optimized layers: {len(o_layers)}")
    print()

    n = min(len(g_layers), len(o_layers))

    # Track first major divergence
    first_diverge_idx = None
    diverge_threshold_int = 1       # for int8/uint8: > 1 quant level
    diverge_threshold_float = 0.01  # for float: relative threshold

    # Header
    hdr = f"{'#':>3} {'Op':>12} {'Numel':>8} {'Dtype':>5} | {'Sum-G':>14} {'Sum-O':>14} {'ΔSum':>12} | {'Min-G':>8} {'Min-O':>8} {'Max-G':>8} {'Max-O':>8} | {'Status'}"
    print(hdr)
    print('-' * len(hdr))

    summary_match = 0
    summary_minor = 0
    summary_major = 0

    for i in range(n):
        g = g_layers[i]
        o = o_layers[i]

        op_match = g['op'] == o['op']
        numel_match = g['numel'] == o['numel']

        delta_sum = o['sum'] - g['sum']

        # Compare first values element-wise
        nfirst = min(len(g['first']), len(o['first']))
        max_elem_diff = 0
        for j in range(nfirst):
            max_elem_diff = max(max_elem_diff, abs(o['first'][j] - g['first'][j]))

        # Determine status
        is_int = g['dtype'] in (0, 1)  # Byte(0), Char/int8(1)

        if is_int:
            if max_elem_diff == 0 and abs(delta_sum) == 0:
                status = "MATCH"
                summary_match += 1
            elif max_elem_diff <= diverge_threshold_int and abs(delta_sum) < g['numel'] * 0.01:
                status = "~minor"
                summary_minor += 1
            else:
                status = "**DIVERGE**"
                summary_major += 1
                if first_diverge_idx is None:
                    first_diverge_idx = i
        else:
            range_g = g['max'] - g['min'] if g['max'] != g['min'] else max(abs(g['max']), 1.0)
            rel_sum = abs(delta_sum) / max(abs(g['sum']), 1e-9)
            rel_elem = max_elem_diff / max(range_g, 1e-9)
            if rel_sum < 1e-6 and rel_elem < 1e-6:
                status = "MATCH"
                summary_match += 1
            elif rel_sum < diverge_threshold_float and rel_elem < diverge_threshold_float:
                status = "~minor"
                summary_minor += 1
            else:
                status = "**DIVERGE**"
                summary_major += 1
                if first_diverge_idx is None:
                    first_diverge_idx = i

        op_label = g['op'] if op_match else f"{g['op']}/{o['op']}"
        numel_label = str(g['numel']) if numel_match else f"{g['numel']}/{o['numel']}"

        print(f"{i:3d} {op_label:>12} {numel_label:>8} {'int' if is_int else 'flt':>5}"
              f" | {g['sum']:>14.4f} {o['sum']:>14.4f} {delta_sum:>12.4f}"
              f" | {g['min']:>8.2f} {o['min']:>8.2f} {g['max']:>8.2f} {o['max']:>8.2f}"
              f" | {status}")

    print()
    print(f"Summary: {summary_match} MATCH, {summary_minor} minor, {summary_major} DIVERGE")

    if first_diverge_idx is not None:
        g = g_layers[first_diverge_idx]
        o = o_layers[first_diverge_idx]
        print(f"\n{'='*70}")
        print(f"FIRST DIVERGENCE at layer #{first_diverge_idx}: {g['op']} (numel={g['numel']})")
        print(f"{'='*70}")
        print(f"  Generic op:   {g['op_raw']}")
        print(f"  Optimized op: {o['op_raw']}")
        print(f"  Generic   first: [{fmt_vals(g['first'], 16)}]")
        print(f"  Optimized first: [{fmt_vals(o['first'], 16)}]")

        # Element diffs
        nfirst = min(len(g['first']), len(o['first']))
        diffs = [o['first'][j] - g['first'][j] for j in range(nfirst)]
        print(f"  Elem diffs:      [{fmt_vals(diffs, 16)}]")
        print(f"  Sum:    generic={g['sum']:.4f}  optimized={o['sum']:.4f}  delta={o['sum']-g['sum']:.4f}")
        print(f"  Min:    generic={g['min']:.4f}  optimized={o['min']:.4f}")
        print(f"  Max:    generic={g['max']:.4f}  optimized={o['max']:.4f}")

        # Check the layer BEFORE for clues
        if first_diverge_idx > 0:
            pg = g_layers[first_diverge_idx - 1]
            po = o_layers[first_diverge_idx - 1]
            print(f"\n  Previous layer #{first_diverge_idx-1}: {pg['op']} (numel={pg['numel']})")
            print(f"    Sum delta: {po['sum'] - pg['sum']:.4f}")
            nfp = min(len(pg['first']), len(po['first']))
            prev_max_diff = max(abs(po['first'][j] - pg['first'][j]) for j in range(nfp))
            print(f"    Max elem diff (first {nfp}): {prev_max_diff:.6f}")
    else:
        print("\nAll layers MATCH or have minor differences.")

    # Check if layer counts differ
    if len(g_layers) != len(o_layers):
        print(f"\nWARNING: Layer count mismatch (generic={len(g_layers)}, optimized={len(o_layers)})")
        if len(g_layers) > n:
            print(f"  Extra generic layers: {[l['op'] for l in g_layers[n:]]}")
        if len(o_layers) > n:
            print(f"  Extra optimized layers: {[l['op'] for l in o_layers[n:]]}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) >= 3:
        gen_log, opt_log = sys.argv[1], sys.argv[2]
    else:
        gen_log = os.path.join(script_dir, 'resnet18_cache_generic.log')
        opt_log = os.path.join(script_dir, 'resnet18_cache_quant1.log')

    if not os.path.exists(gen_log):
        print(f"ERROR: {gen_log} not found"); sys.exit(1)
    if not os.path.exists(opt_log):
        print(f"ERROR: {opt_log} not found"); sys.exit(1)

    compare(gen_log, opt_log)
