import re, os, sys
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

def read_file_safe(filepath):
    try:
        with open(filepath) as f:
            return f.read()
    except:
        return ""

def parse_output_values(content):
    """Extract dequantized output values from log content."""
    if not content:
        return []
    # Match both OutputX 0: and Output 0:
    m = re.search(r'Output[X]? 0: tensor\(sizes=\[.*?\], \[\s*\n(.*?)\n\]\)', content, re.DOTALL)
    if not m:
        return []
    data = m.group(1)
    vals = re.findall(r'-?\d+\.\d*(?:e[+-]?\d+)?', data)
    return [float(v) for v in vals]

def parse_params(content):
    """Extract params from log content."""
    params = {}
    if not content:
        return params
    
    # New format: effective_scale=..., output_scale=..., output_shift=..., accum_shift=... (ic=... kh=... kw=... num_products=...)
    m = re.search(
        r'effective_scale=([\d.]+),\s*output_scale=(\d+),\s*output_shift=(\d+),\s*accum_shift=(\d+)'
        r'(?:\s*\(ic=(\d+)\s+kh=(\d+)\s+kw=(\d+)\s+num_products=(\d+)\))?',
        content
    )
    if m:
        params['effective_scale'] = float(m.group(1))
        params['output_scale'] = int(m.group(2))
        params['output_shift'] = int(m.group(3))
        params['accum_shift'] = int(m.group(4))
        if m.group(5):
            params['ic'] = int(m.group(5))
            params['kh'] = int(m.group(6))
            params['kw'] = int(m.group(7))
            params['num_products'] = int(m.group(8))

    # Get layer dims from n=... line
    m2 = re.search(r'n=(\d+), c=(\d+), h=(\d+), w=(\d+), oc=(\d+), wc=(\d+), wh=(\d+), ww=(\d+)', content)
    if m2:
        if 'ic' not in params:
            params['ic'] = int(m2.group(2))
        if 'kh' not in params:
            params['kh'] = int(m2.group(7))
        if 'kw' not in params:
            params['kw'] = int(m2.group(8))
        if 'num_products' not in params:
            params['num_products'] = int(m2.group(2)) * int(m2.group(7)) * int(m2.group(8))

    # Get float output_scale
    m3 = re.search(r'output_scale=(0\.\d+)', content)
    if m3:
        params['output_scale_float'] = float(m3.group(1))

    return params

def check_clamping(values, threshold=0.15):
    """Check if values show clamping pattern."""
    if len(values) < 50:
        return False, 0, 0.0
    abs_vals = [round(abs(v), 6) for v in values if abs(v) > 1e-9]
    if not abs_vals:
        return False, 0, 0.0
    c = Counter(abs_vals)
    most_common_val, most_common_count = c.most_common(1)[0]
    ratio = most_common_count / len(abs_vals)
    return ratio > threshold, most_common_count, most_common_val

# Collect all data
results = []
for idx, name in layers:
    gen_file = os.path.join(logdir, f"conv2d_layer{idx}_{name}_generic.log")
    opt_file = os.path.join(logdir, f"conv2d_layer{idx}_{name}_opt.log")

    gen_content = read_file_safe(gen_file)
    opt_content = read_file_safe(opt_file)

    gen_params = parse_params(gen_content)
    opt_params = parse_params(opt_content)
    gen_vals = parse_output_values(gen_content)
    opt_vals = parse_output_values(opt_content)

    quant_step = opt_params.get('output_scale_float', gen_params.get('output_scale_float', 0.02))

    # File info
    gen_size = os.path.getsize(gen_file) if os.path.exists(gen_file) else 0
    opt_size = os.path.getsize(opt_file) if os.path.exists(opt_file) else 0
    gen_mtime = os.path.getmtime(gen_file) if os.path.exists(gen_file) else 0
    opt_mtime = os.path.getmtime(opt_file) if os.path.exists(opt_file) else 0

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

    if n_compare == 0:
        match_status = "NO_DATA"
    elif mismatches == 0:
        match_status = "MATCH"
    else:
        match_status = f"MISMATCH({mismatches})"

    # Check clamping
    opt_clamped, opt_clamp_count, opt_clamp_val = check_clamping(opt_vals)
    gen_clamped, gen_clamp_count, gen_clamp_val = check_clamping(gen_vals)

    is_fixed = opt_params.get('accum_shift', -1) > 0
    
    results.append({
        'layer': idx, 'name': name,
        'gen_params': gen_params, 'opt_params': opt_params,
        'gen_first10': gen_vals[:10], 'opt_first10': opt_vals[:10],
        'gen_count': len(gen_vals), 'opt_count': len(opt_vals),
        'match_status': match_status, 'max_diff': max_diff,
        'mismatches': mismatches,
        'opt_clamped': opt_clamped, 'opt_clamp_count': opt_clamp_count, 'opt_clamp_val': opt_clamp_val,
        'gen_clamped': gen_clamped,
        'quant_step': quant_step,
        'opt_size': opt_size,
        'is_fixed': is_fixed,
    })

# Print summary table
print("=" * 170)
print(f"{'Layer':<25} {'accumShift':>10} {'outScale':>10} {'outShift':>9} {'num_prod':>9} {'Status':>12} {'MaxDiff':>10} {'OptClamped':>12} {'ClampVal':>10} {'Fixed?':>8}")
print("=" * 170)

for r in results:
    p = r['opt_params']
    accum = p.get('accum_shift', '?')
    out_sc = p.get('output_scale', '?')
    out_sh = p.get('output_shift', '?')
    np_ = p.get('num_products', r['gen_params'].get('num_products', '?'))

    if r['opt_size'] == 0:
        status = "EMPTY_LOG"
    else:
        status = r['match_status']

    clamp_str = f"YES({r['opt_clamp_count']})" if r['opt_clamped'] else "no"
    clamp_val_str = f"{r['opt_clamp_val']:.6f}" if r['opt_clamped'] else "-"
    fixed_str = "YES" if r['is_fixed'] else ("N/A" if r['opt_size'] == 0 else "NO")

    print(f"L{r['layer']:>2} {r['name']:<20} {str(accum):>10} {str(out_sc):>10} {str(out_sh):>9} {str(np_):>9} {status:>12} {r['max_diff']:>10.6f} {clamp_str:>12} {clamp_val_str:>10} {fixed_str:>8}")

print("=" * 170)

# Counts
fixed_layers = [r for r in results if r['is_fixed']]
unfixed_layers = [r for r in results if not r['is_fixed'] and r['opt_size'] > 0]
empty_layers = [r for r in results if r['opt_size'] == 0]
match_layers = [r for r in results if r['match_status'] == 'MATCH']
mismatch_layers = [r for r in results if 'MISMATCH' in r['match_status']]
clamped_layers = [r for r in results if r['opt_clamped']]

print(f"\n=== SUMMARY ===")
print(f"Fixed (accum_shift > 0): {len(fixed_layers)}/11 - layers {[r['layer'] for r in fixed_layers]}")
print(f"Not fixed (accum_shift=0): {len(unfixed_layers)}/11 - layers {[r['layer'] for r in unfixed_layers]}")
print(f"Empty opt log: {len(empty_layers)}/11 - layers {[r['layer'] for r in empty_layers]}")
print(f"Matching outputs: {len(match_layers)}/11 - layers {[r['layer'] for r in match_layers]}")
print(f"Mismatching outputs: {len(mismatch_layers)}/11 - layers {[r['layer'] for r in mismatch_layers]}")
print(f"Clamped opt outputs: {len(clamped_layers)}/11 - layers {[r['layer'] for r in clamped_layers]}")

# Detailed comparison for each layer
print("\n" + "=" * 120)
print("DETAILED PER-LAYER COMPARISON (first 10 output values)")
print("=" * 120)

for r in results:
    p = r['opt_params']
    gp = r['gen_params']
    print(f"\n--- Layer {r['layer']} ({r['name']}) ---")
    print(f"  Opt params: accum_shift={p.get('accum_shift','?')}, output_scale={p.get('output_scale','?')}, output_shift={p.get('output_shift','?')}")
    if 'num_products' in p:
        print(f"  num_products={p['num_products']} (ic={p.get('ic','?')} kh={p.get('kh','?')} kw={p.get('kw','?')})")
    elif 'num_products' in gp:
        print(f"  num_products={gp['num_products']} (ic={gp.get('ic','?')} kh={gp.get('kh','?')} kw={gp.get('kw','?')}) [from generic]")
    print(f"  quant_step={r['quant_step']:.6f} | gen_vals={r['gen_count']}, opt_vals={r['opt_count']}")
    
    if r['opt_size'] == 0:
        print(f"  >>> OPT LOG IS EMPTY (test in progress or failed) <<<")
        continue

    g10 = r['gen_first10']
    o10 = r['opt_first10']
    if g10 and o10:
        print(f"  Generic: [{', '.join(['%10.6f' % v for v in g10])}]")
        print(f"  Opt:     [{', '.join(['%10.6f' % v for v in o10])}]")
        diffs = [abs(g - o) for g, o in zip(g10, o10)]
        print(f"  Diff:    [{', '.join(['%10.6f' % d for d in diffs])}]")
    
    if r['opt_clamped']:
        print(f"  >>> CLAMPING DETECTED: {r['opt_clamp_count']} opt values saturated to +/-{r['opt_clamp_val']:.6f} <<<")
    
    if r['is_fixed']:
        print(f"  >>> FIX APPLIED: accum_shift={p['accum_shift']} (non-zero) <<<")
    else:
        print(f"  >>> FIX NOT APPLIED: accum_shift={p.get('accum_shift', '?')} <<<")
    
    if r['match_status'] == 'MATCH':
        print(f"  >>> RESULT: MATCH - generic and opt outputs agree <<<")
    elif 'MISMATCH' in r['match_status']:
        print(f"  >>> RESULT: MISMATCH - {r['mismatches']}/20 values differ by > {r['quant_step']*1.5:.6f} (1.5x quant_step) <<<")
