import re, os
from collections import Counter

logdir = "/home/sraut/ext_main/cad_rlc/executorch/operator_and_model_testing/conv2d/logs"

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

def parse_output_values(filepath):
    """Extract dequantized output values from log file."""
    with open(filepath) as f:
        content = f.read()
    m = re.search(r'Output[X]? 0: tensor\(sizes=\[.*?\], \[\s*\n(.*?)\n\]\)', content, re.DOTALL)
    if not m:
        return []
    data = m.group(1)
    # Match all floating point numbers including bare "0."
    vals = re.findall(r'-?\d+\.\d*(?:e[+-]?\d+)?', data)
    result = [float(v) for v in vals]
    return result

def parse_params(filepath):
    """Extract params from opt log."""
    with open(filepath) as f:
        content = f.read()
    params = {}
    m = re.search(r'effective_scale=([\d.]+),\s*output_scale=(\d+),\s*output_shift=(\d+),\s*accum_shift=(\d+)', content)
    if m:
        params['effective_scale'] = float(m.group(1))
        params['output_scale'] = int(m.group(2))
        params['output_shift'] = int(m.group(3))
        params['accum_shift'] = int(m.group(4))

    m2 = re.search(r'n=(\d+), c=(\d+), h=(\d+), w=(\d+), oc=(\d+), wc=(\d+), wh=(\d+), ww=(\d+)', content)
    if m2:
        params['ic'] = int(m2.group(2))
        params['kh'] = int(m2.group(7))
        params['kw'] = int(m2.group(8))
        params['num_products'] = int(m2.group(2)) * int(m2.group(7)) * int(m2.group(8))

    return params

def check_clamping(values, threshold=0.15):
    """Check if values show clamping: many values at exactly the same abs magnitude."""
    if len(values) < 50:
        return False, 0, 0.0
    abs_vals = [round(abs(v), 6) for v in values if abs(v) > 1e-9]
    if not abs_vals:
        return False, 0, 0.0
    c = Counter(abs_vals)
    most_common_val, most_common_count = c.most_common(1)[0]
    ratio = most_common_count / len(abs_vals)
    return ratio > threshold, most_common_count, most_common_val

print("=" * 150)
print(f"{'Layer':<25} {'accumShift':>10} {'outScale':>10} {'outShift':>9} {'ic*kh*kw':>9} {'Match?':>12} {'MaxDiff':>10} {'Clamped?':>12} {'ClampVal':>10}")
print("=" * 150)

results = []
for idx, name in layers:
    gen_file = os.path.join(logdir, f"conv2d_layer{idx}_{name}_generic.log")
    opt_file = os.path.join(logdir, f"conv2d_layer{idx}_{name}_opt.log")

    params = parse_params(opt_file)
    gen_vals = parse_output_values(gen_file)
    opt_vals = parse_output_values(opt_file)

    # Get quant step from output_scale
    with open(opt_file) as f:
        content = f.read()
    m_os = re.search(r'output_scale=(0\.\d+)', content)
    quant_step = float(m_os.group(1)) if m_os else 0.02

    # Compare first 20 values
    n_compare = min(20, len(gen_vals), len(opt_vals))
    gen_first = gen_vals[:n_compare]
    opt_first = opt_vals[:n_compare]

    max_diff = 0.0
    mismatches = 0
    for g, o in zip(gen_first, opt_first):
        diff = abs(g - o)
        max_diff = max(max_diff, diff)
        if diff > quant_step * 1.5:
            mismatches += 1

    match_status = "MATCH" if mismatches == 0 else f"MISMATCH({mismatches})"

    # Check clamping in opt output (all values, not just first 20)
    is_clamped, clamp_count, clamp_val = check_clamping(opt_vals)
    
    # Also check generic for comparison
    gen_clamped, gen_clamp_count, gen_clamp_val = check_clamping(gen_vals)
    
    clamp_str = f"YES({clamp_count})" if is_clamped else "no"
    clamp_val_str = f"{clamp_val:.6f}" if is_clamped else "-"

    num_products = params.get('num_products', '-')

    print(f"L{idx:>2} {name:<20} {params.get('accum_shift','?'):>10} {params.get('output_scale','?'):>10} {params.get('output_shift','?'):>9} {str(num_products):>9} {match_status:>12} {max_diff:>10.6f} {clamp_str:>12} {clamp_val_str:>10}")

    results.append({
        'layer': idx, 'name': name, 'params': params,
        'gen_first10': gen_first[:10], 'opt_first10': opt_first[:10],
        'match': mismatches == 0, 'max_diff': max_diff,
        'is_clamped': is_clamped, 'clamp_val': clamp_val if is_clamped else None,
        'clamp_count': clamp_count if is_clamped else 0,
        'mismatches': mismatches,
        'quant_step': quant_step,
        'total_gen': len(gen_vals), 'total_opt': len(opt_vals),
    })

print("=" * 150)

# Count totals
total_match = sum(1 for r in results if r['match'])
total_clamped = sum(1 for r in results if r['is_clamped'])
print(f"\nSummary: {total_match}/11 layers match, {total_clamped}/11 layers show clamping in opt")
print()

# Print first 10 values for each layer
for r in results:
    print(f"\n--- Layer {r['layer']} ({r['name']}) | quant_step={r['quant_step']:.6f} | total_vals: gen={r['total_gen']}, opt={r['total_opt']} ---")
    g10 = r['gen_first10']
    o10 = r['opt_first10']
    print(f"  Generic: {', '.join(['%9.6f' % v for v in g10])}")
    print(f"  Opt:     {', '.join(['%9.6f' % v for v in o10])}")
    diffs = [abs(g - o) for g, o in zip(g10, o10)]
    print(f"  Diff:    {', '.join(['%9.6f' % d for d in diffs])}")
    if r['is_clamped']:
        print(f"  >>> CLAMPING DETECTED: {r['clamp_count']} opt values saturated to +/-{r['clamp_val']:.6f} <<<")
    if not r['match']:
        print(f"  >>> MISMATCH: {r['mismatches']}/20 values differ by > 1.5x quant_step ({r['quant_step']*1.5:.6f}) <<<")
