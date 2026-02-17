#!/usr/bin/env python3
"""Compare ResNet18 full-model generic vs optimized final output."""
import re
import sys

def extract_final_output(filepath):
    """Extract final dequantized output tensor [1, 1000] from log."""
    with open(filepath) as f:
        content = f.read()
    
    # Find the LAST output tensor (final model output)
    matches = list(re.finditer(r'Output[X]?\s+0:\s+tensor\(sizes=\[1,\s*1000\],\s*\[(.*?)\]\)', content, re.DOTALL))
    if not matches:
        print(f"ERROR: No [1,1000] output tensor found in {filepath}")
        return []
    
    val_str = matches[-1].group(1)
    values = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', val_str)
    return [float(v) for v in values]

def extract_layer_info(filepath):
    """Extract per-layer info from opt log."""
    with open(filepath) as f:
        lines = f.readlines()
    
    layers = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'quantized_conv2d_nchw: n=(\d+), c=(\d+), h=(\d+), w=(\d+), oc=(\d+), wc=(\d+), wh=(\d+), ww=(\d+), oh=(\d+), ow=(\d+)', line)
        if m:
            info = {k: int(v) for k, v in zip(['n','c','h','w','oc','wc','wh','ww','oh','ow'], m.groups())}
            # Look ahead for accum_shift info
            for j in range(i+1, min(i+20, len(lines))):
                m2 = re.search(r'accum_shift=(\d+).*num_products=(\d+)', lines[j])
                if m2:
                    info['accum_shift'] = int(m2.group(1))
                    info['num_products'] = int(m2.group(2))
                    break
                m3 = re.search(r'output_scale=(\d+), output_shift=(\d+), accum_shift=(\d+)', lines[j])
                if m3:
                    info['xai_output_scale'] = int(m3.group(1))
                    info['xai_output_shift'] = int(m3.group(2))
                    info['accum_shift'] = int(m3.group(3))
                # Get kernel name
                m4 = re.search(r'Kernel Name: (\S+)', lines[j])
                if m4:
                    info['kernel'] = m4.group(1)
                m5 = re.search(r'Layer Name: (\S+)', lines[j])
                if m5:
                    info['layer_name'] = m5.group(1)
            layers.append(info)
        i += 1
    return layers

# ===== Compare final outputs =====
gen_file = "/home/sraut/ext_main/cad_rlc/executorch/resnet18_generic.log"
opt_file = "/home/sraut/ext_main/cad_rlc/executorch/resnet18.log"

gen_vals = extract_final_output(gen_file)
opt_vals = extract_final_output(opt_file)

print(f"Generic output: {len(gen_vals)} values")
print(f"Optimized output: {len(opt_vals)} values")
print()

if not gen_vals or not opt_vals:
    sys.exit(1)

total = min(len(gen_vals), len(opt_vals))

# The output scale for the final dequantize_per_tensor 
# Find it from the generic log - last output_scale or we can compute from step size
# The final output is dequantized with a single scale
# Let's detect the quant step from the values themselves
gen_steps = sorted(set(abs(gen_vals[i] - gen_vals[j]) for i in range(min(50, total)) for j in range(i+1, min(50, total)) if abs(gen_vals[i] - gen_vals[j]) > 0.001))
quant_step = gen_steps[0] if gen_steps else 0.0724269  # smallest nonzero diff

print(f"Detected quantization step: {quant_step:.6f}")
print()

# Count mismatches
exact = 0
off1 = 0
off2 = 0
worse = 0
max_diff = 0.0
max_diff_idx = 0
diffs = []

for i in range(total):
    d = abs(gen_vals[i] - opt_vals[i])
    diffs.append(d)
    if d < 1e-6:
        exact += 1
    elif d <= quant_step * 1.01:
        off1 += 1
    elif d <= quant_step * 2.01:
        off2 += 1
    else:
        worse += 1
    if d > max_diff:
        max_diff = d
        max_diff_idx = i

print(f"{'Metric':<25} {'Count':>8} {'Pct':>8}")
print("-" * 45)
print(f"{'Exact match':<25} {exact:>8} {100*exact/total:>7.1f}%")
print(f"{'Off-by-1 step':<25} {off1:>8} {100*off1/total:>7.1f}%")
print(f"{'Off-by-2 steps':<25} {off2:>8} {100*off2/total:>7.1f}%")
print(f"{'Worse (>2 steps)':<25} {worse:>8} {100*worse/total:>7.1f}%")
print(f"{'Total':<25} {total:>8}")
print(f"\nMax difference: {max_diff:.6f} at index {max_diff_idx} ({max_diff/quant_step:.1f} steps)")
print(f"  Generic[{max_diff_idx}] = {gen_vals[max_diff_idx]:.6f}")
print(f"  Optimized[{max_diff_idx}] = {opt_vals[max_diff_idx]:.6f}")

# Show top-10 worst mismatches
print(f"\nTop 10 largest differences:")
sorted_diffs = sorted(enumerate(diffs), key=lambda x: -x[1])
for idx, d in sorted_diffs[:10]:
    steps = d / quant_step
    print(f"  [{idx:>4}] gen={gen_vals[idx]:>10.6f}  opt={opt_vals[idx]:>10.6f}  diff={d:.6f} ({steps:.1f} steps)")

# Top-K accuracy comparison
gen_arr = gen_vals[:1000]
opt_arr = opt_vals[:1000]

gen_top5 = sorted(range(1000), key=lambda i: -gen_arr[i])[:5]
opt_top5 = sorted(range(1000), key=lambda i: -opt_arr[i])[:5]
gen_top1 = gen_top5[0]
opt_top1 = opt_top5[0]

print(f"\n--- Top-K Classification ---")
print(f"Generic  Top-1: class {gen_top1} (score {gen_arr[gen_top1]:.4f})")
print(f"Optimized Top-1: class {opt_top1} (score {opt_arr[opt_top1]:.4f})")
print(f"Top-1 match: {'YES' if gen_top1 == opt_top1 else 'NO'}")
print(f"\nGeneric  Top-5: {gen_top5}")
print(f"Optimized Top-5: {opt_top5}")
overlap = len(set(gen_top5) & set(opt_top5))
print(f"Top-5 overlap: {overlap}/5")

# ===== Per-layer info =====
print(f"\n--- Per-Layer Optimized Kernel Info ---")
layers = extract_layer_info(opt_file)
print(f"{'#':<3} {'Name':<12} {'Kernel':<12} {'ic':>4} {'kh':>3} {'kw':>3} {'nProd':>6} {'accShift':>8}")
print("-" * 58)
for i, l in enumerate(layers):
    name = l.get('layer_name', '?')
    kernel = l.get('kernel', '?')
    ic = l.get('c', 0)
    kh = l.get('wh', 0)
    kw = l.get('ww', 0)
    np_ = l.get('num_products', ic*kh*kw)
    ashift = l.get('accum_shift', '?')
    print(f"{i:<3} {name:<12} {kernel:<12} {ic:>4} {kh:>3} {kw:>3} {np_:>6} {str(ashift):>8}")
