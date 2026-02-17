#!/usr/bin/env python3
"""
Generate layerwise comparison plots: Generic vs Optimized conv2d outputs.
Generates per-layer plots and a summary dashboard.
"""

import re
import os
import sys

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Configuration ───────────────────────────────────────────────────
LOGDIR = "/home/sraut/ext_main/cad_rlc/executorch/operator_and_model_testing/conv2d/logs"
PLOTDIR = os.path.join(LOGDIR, "plot")
RESNET_LOG_DIR = "/home/sraut/ext_main/cad_rlc/executorch"

LAYERS = [
    (0,  "conv1"),
    (1,  "conv2_1"),
    (2,  "conv4b_1"),
    (3,  "conv4b_2"),
    (4,  "conv4a_1"),
    (5,  "conv6b_1"),
    (6,  "conv6b_2"),
    (7,  "conv6a_1"),
    (8,  "conv8b_1"),
    (9,  "conv8b_2"),
    (10, "conv8a_1"),
]

# ─── Parsing helpers ─────────────────────────────────────────────────
def parse_output_values(filepath):
    """Extract dequantized output values from a log file."""
    with open(filepath) as f:
        content = f.read()
    m = re.search(r'Output[X]? 0: tensor\(sizes=\[.*?\], \[\s*\n(.*?)\n\]\)', content, re.DOTALL)
    if not m:
        return []
    data = m.group(1)
    vals = re.findall(r'-?\d+\.\d*(?:e[+-]?\d+)?', data)
    return [float(v) for v in vals]


def parse_layer_info(filepath):
    """Extract layer shape parameters from a log file."""
    with open(filepath) as f:
        content = f.read()
    info = {}
    m = re.search(r'n=(\d+), c=(\d+), h=(\d+), w=(\d+), oc=(\d+), wc=(\d+), wh=(\d+), ww=(\d+), oh=(\d+), ow=(\d+)', content)
    if m:
        info['ic'] = int(m.group(2))
        info['ih'] = int(m.group(3))
        info['iw'] = int(m.group(4))
        info['oc'] = int(m.group(5))
        info['kh'] = int(m.group(7))
        info['kw'] = int(m.group(8))
        info['oh'] = int(m.group(9))
        info['ow'] = int(m.group(10))
    m2 = re.search(r'output_scale=([\d.]+)', content)
    if m2:
        info['output_scale'] = float(m2.group(1))
    m3 = re.search(r'accum_shift=(\d+)', content)
    if m3:
        info['accum_shift'] = int(m3.group(1))
    return info


def compute_stats(gen_vals, opt_vals):
    """Compute comparison statistics between generic and opt values."""
    n = min(len(gen_vals), len(opt_vals))
    if n == 0:
        return {}
    gen = gen_vals[:n]
    opt = opt_vals[:n]
    diffs = [abs(g - o) for g, o in zip(gen, opt)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    exact = sum(1 for d in diffs if d < 1e-9)
    return {
        'n': n,
        'gen': gen,
        'opt': opt,
        'diffs': diffs,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'exact_match': exact,
        'exact_pct': 100.0 * exact / n,
    }


# ─── Plotting functions ─────────────────────────────────────────────

def subsample(lst, max_n=5000):
    """Return a subsampled list if too large, to keep plotting fast."""
    if len(lst) <= max_n:
        return lst, list(range(len(lst)))
    step = len(lst) // max_n
    indices = list(range(0, len(lst), step))[:max_n]
    return [lst[i] for i in indices], indices


def plot_layer_scatter(gen, opt, layer_idx, layer_name, info, stats, outpath):
    """4-panel plot: generic vs optimized values for one layer."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Layer {layer_idx}: {layer_name}  |  "
                 f"ic={info.get('ic','?')} kh={info.get('kh','?')}x{info.get('kw','?')} "
                 f"oc={info.get('oc','?')} oh={info.get('oh','?')}x{info.get('ow','?')}  |  "
                 f"accumShift={info.get('accum_shift','?')}",
                 fontsize=12, fontweight='bold')

    n = stats['n']
    diffs = stats['diffs']

    # --- Subplot 1: Scatter generic vs opt (subsampled for speed) ---
    ax = axes[0, 0]
    gen_s, _ = subsample(gen, 5000)
    opt_s, _ = subsample(opt, 5000)
    ax.plot(gen_s, opt_s, '.', markersize=1, alpha=0.3, color='steelblue')
    lims = [min(min(gen), min(opt)), max(max(gen), max(opt))]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, 'r-', linewidth=0.8, label='y=x (perfect match)')
    ax.set_xlabel('Generic (float reference)')
    ax.set_ylabel('Optimized (int16 pipeline)')
    ax.set_title(f'Generic vs Optimized ({n} values)')
    ax.legend(fontsize=8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: Difference histogram ---
    ax = axes[0, 1]
    quant_step = info.get('output_scale', 0.02)
    step_diffs = [d / quant_step for d in diffs] if quant_step > 0 else diffs
    max_steps = max(step_diffs) if step_diffs else 0
    bins = min(100, max(20, int(max_steps) + 1))
    ax.hist(step_diffs, bins=bins, color='darkorange', alpha=0.8, edgecolor='black', linewidth=0.3)
    ax.axvline(x=1, color='green', linestyle='--', linewidth=1, label='1 quant step')
    ax.axvline(x=2, color='red', linestyle='--', linewidth=1, label='2 quant steps')
    ax.set_xlabel(f'|Difference| (quant steps, step={quant_step:.6f})')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution  |  max={max_steps:.1f} steps, mean={sum(step_diffs)/len(step_diffs):.2f} steps')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: Value-by-index overlay (subsampled) ---
    ax = axes[1, 0]
    show_n = min(2000, n)
    gen_show, idx_range = subsample(gen[:show_n], 2000)
    opt_show = [opt[i] for i in idx_range]
    ax.plot(idx_range, gen_show, linewidth=0.5, alpha=0.7, label='Generic', color='blue')
    ax.plot(idx_range, opt_show, linewidth=0.5, alpha=0.7, label='Optimized', color='red')
    ax.set_xlabel('Output Index')
    ax.set_ylabel('Dequantized Value')
    ax.set_title(f'Output Values Overlay (first {show_n} of {n})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Subplot 4: Difference by index (subsampled) ---
    ax = axes[1, 1]
    diffs_show = [diffs[i] for i in idx_range]
    ax.plot(idx_range, diffs_show, linewidth=0.5, color='purple', alpha=0.6)
    ax.axhline(y=quant_step, color='green', linestyle='--', linewidth=1, label=f'1 step ({quant_step:.4f})')
    ax.axhline(y=2*quant_step, color='red', linestyle='--', linewidth=1, label=f'2 steps ({2*quant_step:.4f})')
    ax.set_xlabel('Output Index')
    ax.set_ylabel('|Generic - Optimized|')
    ax.set_title(f'Absolute Error by Index (first {show_n})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(outpath)}")


def plot_summary_dashboard(all_results, outpath):
    """Summary dashboard across all layers."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ResNet18 Conv2D: Generic vs Optimized — Layer-wise Summary',
                 fontsize=14, fontweight='bold')

    layer_labels = [f"L{r['idx']}:{r['name']}" for r in all_results]
    x = list(range(len(all_results)))

    # --- Subplot 1: Max diff in quant steps per layer ---
    ax = axes[0, 0]
    max_steps = []
    for r in all_results:
        qs = r['info'].get('output_scale', 0.02)
        ms = r['stats']['max_diff'] / qs if qs > 0 else r['stats']['max_diff']
        max_steps.append(ms)
    bars = ax.bar(x, max_steps, color='tomato', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1, label='1 step')
    ax.axhline(y=2, color='orange', linestyle='--', linewidth=1, label='2 steps')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Max |diff| (quant steps)')
    ax.set_title('Max Error per Layer')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(max_steps):
        ax.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=7)

    # --- Subplot 2: Mean diff in quant steps per layer ---
    ax = axes[0, 1]
    mean_steps = []
    for r in all_results:
        qs = r['info'].get('output_scale', 0.02)
        ms = r['stats']['mean_diff'] / qs if qs > 0 else r['stats']['mean_diff']
        mean_steps.append(ms)
    ax.bar(x, mean_steps, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1, label='1 step')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean |diff| (quant steps)')
    ax.set_title('Mean Error per Layer')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mean_steps):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    # --- Subplot 3: Exact match % per layer ---
    ax = axes[1, 0]
    exact_pcts = [r['stats']['exact_pct'] for r in all_results]
    colors = ['forestgreen' if p > 90 else 'orange' if p > 50 else 'red' for p in exact_pcts]
    ax.bar(x, exact_pcts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Exact Match %')
    ax.set_title('Exact Match Percentage per Layer')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(exact_pcts):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=7)

    # --- Subplot 4: accumShift and num_products per layer ---
    ax = axes[1, 1]
    acc_shifts = [r['info'].get('accum_shift', 0) for r in all_results]
    num_prods = [r['info'].get('ic', 0) * r['info'].get('kh', 0) * r['info'].get('kw', 0) for r in all_results]
    ax2 = ax.twinx()
    b1 = ax.bar([i - 0.2 for i in x], acc_shifts, 0.4, color='mediumpurple', alpha=0.8,
                edgecolor='black', linewidth=0.5, label='accumShift')
    b2 = ax2.bar([i + 0.2 for i in x], num_prods, 0.4, color='goldenrod', alpha=0.8,
                 edgecolor='black', linewidth=0.5, label='ic×kh×kw')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('accumShift', color='mediumpurple')
    ax2.set_ylabel('ic × kh × kw', color='goldenrod')
    ax.set_title('accumShift & NumProducts per Layer')
    lines = [b1, b2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(outpath)}")


def plot_final_output(outpath):
    """Plot the final model output comparison (resnet18 generic vs opt)."""
    gen_file = os.path.join(RESNET_LOG_DIR, "resnet18_generic.log")
    opt_file = os.path.join(RESNET_LOG_DIR, "resnet18.log")

    if not os.path.exists(gen_file) or not os.path.exists(opt_file):
        print("  Skipping final output plot (resnet18 logs not found)")
        return

    gen_vals = parse_output_values(gen_file)
    opt_vals = parse_output_values(opt_file)

    if not gen_vals or not opt_vals:
        print("  Skipping final output plot (no values parsed)")
        return

    n = min(len(gen_vals), len(opt_vals))
    gen = gen_vals[:n]
    opt = opt_vals[:n]
    diffs = [abs(g - o) for g, o in zip(gen, opt)]
    quant_step = 0.0724269  # from logs

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ResNet18 Final Output [1,1000]: Generic vs Optimized',
                 fontsize=14, fontweight='bold')

    # --- Subplot 1: Overlay ---
    ax = axes[0, 0]
    idx = list(range(n))
    ax.plot(idx, gen, linewidth=0.6, alpha=0.7, label='Generic', color='blue')
    ax.plot(idx, opt, linewidth=0.6, alpha=0.7, label='Optimized', color='red')
    ax.set_xlabel('Class Index')
    ax.set_ylabel('Logit (dequantized)')
    ax.set_title('Final Logits Overlay')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: Scatter ---
    ax = axes[0, 1]
    ax.scatter(gen, opt, s=8, alpha=0.5, c='steelblue', edgecolors='none')
    lims = [min(min(gen), min(opt)), max(max(gen), max(opt))]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, 'r-', linewidth=0.8, label='y=x')
    ax.set_xlabel('Generic')
    ax.set_ylabel('Optimized')
    ax.set_title('Scatter: Generic vs Optimized')
    ax.legend(fontsize=8)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: Error histogram ---
    ax = axes[1, 0]
    step_diffs = [d / quant_step for d in diffs]
    ax.hist(step_diffs, bins=50, color='darkorange', alpha=0.8, edgecolor='black', linewidth=0.3)
    ax.axvline(x=1, color='green', linestyle='--', linewidth=1, label='1 step')
    ax.axvline(x=2, color='red', linestyle='--', linewidth=1, label='2 steps')
    ax.set_xlabel('|Difference| (quant steps)')
    ax.set_ylabel('Count')
    exact = sum(1 for d in step_diffs if d < 0.5)
    within1 = sum(1 for d in step_diffs if d < 1.5)
    within2 = sum(1 for d in step_diffs if d < 2.5)
    ax.set_title(f'Error Dist  |  exact={exact}, ≤1step={within1}, ≤2steps={within2}, max={max(step_diffs):.1f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Subplot 4: Error by class index ---
    ax = axes[1, 1]
    ax.bar(idx, diffs, width=1.0, color='purple', alpha=0.5)
    ax.axhline(y=quant_step, color='green', linestyle='--', linewidth=1, label=f'1 step ({quant_step:.4f})')
    ax.axhline(y=2*quant_step, color='red', linestyle='--', linewidth=1, label=f'2 steps')
    # Mark the top-1 class
    gen_top1 = gen.index(max(gen))
    opt_top1 = opt.index(max(opt))
    ax.axvline(x=gen_top1, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Gen Top-1: class {gen_top1}')
    if opt_top1 != gen_top1:
        ax.axvline(x=opt_top1, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Opt Top-1: class {opt_top1}')
    ax.set_xlabel('Class Index')
    ax.set_ylabel('|Generic - Optimized|')
    ax.set_title(f'Absolute Error by Class  |  Top-1: gen={gen_top1} opt={opt_top1} {"MATCH" if gen_top1==opt_top1 else "MISMATCH"}')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(outpath)}")


# ─── Main ────────────────────────────────────────────────────────────
def main():
    os.makedirs(PLOTDIR, exist_ok=True)

    all_results = []

    print(f"\n{'='*70}")
    print(f"  Generating layerwise plots: Generic vs Optimized")
    print(f"  Output directory: {PLOTDIR}")
    print(f"{'='*70}\n")

    for idx, name in LAYERS:
        gen_file = os.path.join(LOGDIR, f"conv2d_layer{idx}_{name}_generic.log")
        opt_file = os.path.join(LOGDIR, f"conv2d_layer{idx}_{name}_opt.log")

        if not os.path.exists(gen_file) or not os.path.exists(opt_file):
            print(f"Layer {idx} ({name}): SKIPPED (log file(s) missing)")
            continue

        print(f"Layer {idx} ({name}):")
        gen_vals = parse_output_values(gen_file)
        opt_vals = parse_output_values(opt_file)
        info = parse_layer_info(opt_file)
        stats = compute_stats(gen_vals, opt_vals)

        if not stats:
            print(f"  SKIPPED (no values parsed)")
            continue

        all_results.append({
            'idx': idx, 'name': name, 'info': info, 'stats': stats
        })

        # Per-layer 4-panel plot (skip if already exists unless --force)
        layer_png = os.path.join(PLOTDIR, f"layer{idx:02d}_{name}.png")
        if os.path.exists(layer_png) and '--force' not in sys.argv:
            print(f"  Exists: {os.path.basename(layer_png)} (use --force to regenerate)")
        else:
            plot_layer_scatter(
                stats['gen'], stats['opt'],
                idx, name, info, stats,
                layer_png
            )

    # Summary dashboard
    if all_results:
        print(f"\nGenerating summary dashboard...")
        plot_summary_dashboard(all_results, os.path.join(PLOTDIR, "summary_dashboard.png"))

    # Final model output plot
    print(f"\nGenerating final output comparison...")
    plot_final_output(os.path.join(PLOTDIR, "resnet18_final_output.png"))

    print(f"\n{'='*70}")
    print(f"  Done! {len(all_results)} layer plots + summary + final output")
    print(f"  Output directory: {PLOTDIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
