#!/usr/bin/env python3
"""Compare generic vs optimized conv2d output values for bit-exactness."""
import re
import sys
import os

LOGDIR = os.path.dirname(os.path.abspath(__file__)) + "/logs"

def extract_output_values(filepath):
    """Extract dequantized float output values from executor_runner log."""
    with open(filepath) as f:
        content = f.read()
    
    # Find the output tensor section: "Output 0: tensor(sizes=..." or "OutputX 0: tensor(sizes=..."
    m = re.search(r'Output[X]?\s+0:\s+tensor\(sizes=\[.*?\],\s*\[(.*?)\]\)', content, re.DOTALL)
    if not m:
        print(f"  ERROR: Could not find output tensor in {filepath}")
        return []
    
    val_str = m.group(1)
    # Extract all floating point numbers
    values = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', val_str)
    return [float(v) for v in values]

layers = [
    (0, "conv1"),
    (1, "conv2_1"),
    (2, "conv4b_1"),
    (3, "conv4b_2"),
    (4, "conv4a_1"),
    (5, "conv6b_1"),
    (6, "conv6b_2"),
    (7, "conv6a_1"),
    (8, "conv8b_1"),
    (9, "conv8b_2"),
    (10, "conv8a_1"),
]

print(f"{'Layer':<22} {'Total':>7} {'Exact':>7} {'Off-by-1':>8} {'Worse':>7} {'MaxDiff':>10} {'Status'}")
print("-" * 78)

all_exact = True
for idx, name in layers:
    gen_file = f"{LOGDIR}/conv2d_layer{idx}_{name}_generic.log"
    opt_file = f"{LOGDIR}/conv2d_layer{idx}_{name}_opt.log"
    
    if not os.path.exists(gen_file) or not os.path.exists(opt_file):
        print(f"L{idx:<2} {name:<18} MISSING LOG FILES")
        continue
    
    gen_vals = extract_output_values(gen_file)
    opt_vals = extract_output_values(opt_file)
    
    if not gen_vals or not opt_vals:
        continue
    
    total = min(len(gen_vals), len(opt_vals))
    
    # Also extract output_scale from the opt log to determine 1-quant-step size
    with open(opt_file) as f:
        opt_content = f.read()
    m_scale = re.search(r'output_scale=([\d.]+)', opt_content)
    # This is the pytorch output_scale (dequant scale), from the first line
    m_scale2 = re.search(r'output_scale=(0\.\d+)', opt_content)
    quant_step = float(m_scale2.group(1)) if m_scale2 else 0.02  # fallback
    
    exact_match = 0
    off_by_1 = 0  # within 1 quantization step
    worse = 0
    max_diff = 0.0
    worst_examples = []
    
    for i in range(total):
        diff = abs(gen_vals[i] - opt_vals[i])
        if diff < 1e-6:
            exact_match += 1
        elif diff <= quant_step * 1.01:
            off_by_1 += 1
        else:
            worse += 1
        
        if diff > max_diff:
            max_diff = diff
        
        if diff > quant_step * 1.01 and len(worst_examples) < 5:
            worst_examples.append((i, gen_vals[i], opt_vals[i], diff))
    
    status = "BIT-EXACT" if exact_match == total else \
             "~1-STEP" if worse == 0 else \
             f"MISMATCH"
    
    if exact_match != total:
        all_exact = False
    
    print(f"L{idx:<2} {name:<18} {total:>7} {exact_match:>7} {off_by_1:>8} {worse:>7} {max_diff:>10.6f}  {status}")
    
    if worst_examples:
        for pos, gv, ov, d in worst_examples[:3]:
            print(f"    idx={pos}: generic={gv:.6f} opt={ov:.6f} diff={d:.6f} ({d/quant_step:.1f} steps)")

print()
if all_exact:
    print("ALL LAYERS BIT-EXACT")
else:
    print("Some layers have differences. Off-by-1 = within 1 quantization step (rounding).")
