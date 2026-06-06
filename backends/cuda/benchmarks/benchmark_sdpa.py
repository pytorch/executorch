#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark the Triton SDPA kernel against PyTorch SDPA backends.

Measures latency across decode shapes matching the Qwen3.5 MoE model
(B=1, H_q=16, H_kv=2, D=256). The ET Triton kernel uses native GQA
(2 KV heads), while Flash/Efficient/Math require pre-expanded KV
(16 heads) since they lack native GQA support.

"""

import argparse
import warnings
from functools import partial

import torch
import torch.nn.functional as F

from executorch.backends.cuda.triton.kernels.sdpa import (
    sdpa as triton_sdpa,
    sdpa_decode_splitk as triton_splitk,
)
from torch.nn.attention import sdpa_kernel, SDPBackend
from triton.testing import do_bench


# PyTorch's Flash/Efficient backends don't support GQA (H_q != H_kv) directly.
# We expand KV heads via repeat_interleave so they can run, matching what
# the test reference does. This is fair: it measures the kernel itself, not
# the GQA dispatch overhead.


def _expand_kv(k, v, num_groups):
    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)
    return k, v


def _expand_mask(mask, H_q):
    if mask is not None and mask.shape[1] == 1 and H_q > 1:
        mask = mask.expand(-1, H_q, -1, -1)
    return mask


def _run_triton(q, k, v, attn_mask, enable_gqa):
    return triton_sdpa(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)


def _run_splitk(q, k, v, attn_mask, enable_gqa):
    return triton_splitk(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)


def _run_pytorch_default(q, k, v, attn_mask, enable_gqa):
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        enable_gqa=enable_gqa,
    )


def _make_pytorch_runner(backend: SDPBackend):
    def run(q, k, v, attn_mask, enable_gqa):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    return run


# Flash doesn't support attn_mask at all, only is_causal.
# Our benchmark mask is all-ones, so no mask is equivalent.
def _run_flash(q, k, v, attn_mask, enable_gqa):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v)


BACKENDS = {
    "triton": ("ET Triton (GQA)", _run_triton),
    "splitk": ("ET Split-K (GQA)", _run_splitk),
    "pytorch": ("PyTorch", _run_pytorch_default),
    "flash": ("Flash (expanded KV)", _run_flash),
    "efficient": (
        "Efficient (expanded KV)",
        _make_pytorch_runner(SDPBackend.EFFICIENT_ATTENTION),
    ),
    "math": ("Math (expanded KV)", _make_pytorch_runner(SDPBackend.MATH)),
}

# Backends that need KV heads expanded before calling (no native GQA support)
_NEEDS_KV_EXPAND = {"flash", "efficient", "math"}

# -- Shapes ------------------------------------------------------------------

# Qwen3.5 MoE: B=1, H_q=16, H_kv=2, D=256
QWEN35_BASE = {"B": 1, "H_q": 16, "H_kv": 2, "D": 256}

DECODE_SHAPES = [
    dict(**QWEN35_BASE, Lq=1, Lk=64),
    dict(**QWEN35_BASE, Lq=1, Lk=128),
    dict(**QWEN35_BASE, Lq=1, Lk=256),
    dict(**QWEN35_BASE, Lq=1, Lk=512),
    dict(**QWEN35_BASE, Lq=1, Lk=1024),
    dict(**QWEN35_BASE, Lq=1, Lk=2048),
    dict(**QWEN35_BASE, Lq=1, Lk=4096),
    dict(**QWEN35_BASE, Lq=1, Lk=8192),
    dict(**QWEN35_BASE, Lq=1, Lk=16384),
]

SCENARIOS = {
    "decode": DECODE_SHAPES,
}

# -- Helpers -----------------------------------------------------------------


def _make_tensors(B, H_q, H_kv, Lq, Lk, D, device="cuda", dtype=torch.bfloat16):
    q = torch.randn(B, H_q, Lq, D, device=device, dtype=dtype)
    k = torch.randn(B, H_kv, Lk, D, device=device, dtype=dtype)
    v = torch.randn(B, H_kv, Lk, D, device=device, dtype=dtype)
    mask = torch.ones(B, 1, Lq, Lk, dtype=torch.bool, device=device)
    enable_gqa = H_q != H_kv
    num_groups = H_q // H_kv
    # Pre-expanded versions for backends without native GQA
    k_exp, v_exp = _expand_kv(k, v, num_groups)
    mask_exp = _expand_mask(mask, H_q)
    return q, k, v, k_exp, v_exp, mask, mask_exp, enable_gqa


def _max_abs_error(out, ref):
    return (out.float() - ref.float()).abs().max().item()


# Cross-backend validation tolerance (bf16 vs bf16).
MAX_ABS_TOL = 1e-2


def _bench_us(fn, num_warmup, num_iters):
    """Return median latency in microseconds using triton.testing.do_bench."""
    ms = do_bench(fn, warmup=num_warmup, rep=num_iters, return_mode="median")
    return ms * 1000.0


def _try_run(run_fn, q, k, v, mask, enable_gqa):
    """Run a backend, returning output or None on failure."""
    try:
        return run_fn(q, k, v, mask, enable_gqa)
    except RuntimeError:
        return None


def _try_bench(run_fn, q, k, v, mask, enable_gqa, num_warmup, num_iters):
    """Benchmark a backend, returning median us or None on failure."""
    fn = partial(run_fn, q, k, v, mask, enable_gqa)
    try:
        run_fn(q, k, v, mask, enable_gqa)
        return _bench_us(fn, num_warmup, num_iters)
    except RuntimeError:
        return None


# -- Main --------------------------------------------------------------------


def _shape_label(shape):
    return (
        f"B={shape['B']} Hq={shape['H_q']} Hkv={shape['H_kv']} "
        f"D={shape['D']} Lq={shape['Lq']} Lk={shape['Lk']}"
    )


def _short_label(shape, scenario="decode"):
    return f"Lq={shape['Lq']},Lk={shape['Lk']}"


@torch.inference_mode()
def run_benchmark(
    scenario: str = "decode",
    num_warmup: int = 25,
    num_iters: int = 100,
):
    shapes = SCENARIOS[scenario]
    backends = [(name, *BACKENDS[name]) for name in BACKENDS]

    device_name = torch.cuda.get_device_name()
    print()
    print("=" * 100)
    print(f"SDPA Benchmark Qwen3.5-35B-A3B — {scenario}")
    print(f"  Device: {device_name}")
    print(f"  Warmup: {num_warmup}, Iters: {num_iters}")
    print(f"  Backends: {', '.join(label for _, label, _ in backends)}")
    print("=" * 100)

    # Build column specs: (header_text, unit_text, min_width)
    # Each column gets width = max(len(header), len(unit), min_width)
    max_label = max(len(_short_label(s, scenario)) for s in shapes)
    col_specs = [("Shape", "", max(8, max_label))]
    for _, label, _ in backends:
        col_specs.append((label, "(us)", 8))

    col_widths = [max(len(h), len(u), mw) for h, u, mw in col_specs]

    header = " | ".join(
        f"{h:<{w}}" if i == 0 else f"{h:>{w}}"
        for i, ((h, _, _), w) in enumerate(zip(col_specs, col_widths))
    )
    units = " | ".join(
        f"{'':>{w}}" if i == 0 else f"{u:>{w}}"
        for i, ((_, u, _), w) in enumerate(zip(col_specs, col_widths))
    )
    print(header)
    print(units)
    print("-" * len(header))

    for shape in shapes:
        q, k, v, k_exp, v_exp, mask, mask_exp, enable_gqa = _make_tensors(**shape)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Validate outputs across backends before benchmarking
            outputs = {}
            for name, _label, run_fn in backends:
                if name in _NEEDS_KV_EXPAND:
                    bk, bv, bmask = k_exp, v_exp, mask_exp
                else:
                    bk, bv, bmask = k, v, mask
                outputs[name] = _try_run(run_fn, q, bk, bv, bmask, enable_gqa)

            # Use PyTorch F.sdpa as the trusted reference — never validate
            # against our own Triton kernels.
            ref_name, ref_out = None, None
            if outputs.get("pytorch") is not None:
                ref_name, ref_out = "pytorch", outputs["pytorch"]

            if ref_out is not None:
                for name, label, _ in backends:
                    if name == ref_name or outputs[name] is None:
                        continue
                    err = _max_abs_error(outputs[name], ref_out)
                    assert err < MAX_ABS_TOL, (
                        f"Output mismatch for {_shape_label(shape)}: "
                        f"{label} vs {BACKENDS[ref_name][0]}, "
                        f"max abs error {err:.3e} >= 1e-2"
                    )
            del outputs

            # Benchmark all backends
            times = {}
            for name, _label, run_fn in backends:
                if name in _NEEDS_KV_EXPAND:
                    bk, bv, bmask = k_exp, v_exp, mask_exp
                else:
                    bk, bv, bmask = k, v, mask
                times[name] = _try_bench(
                    run_fn, q, bk, bv, bmask, enable_gqa, num_warmup, num_iters
                )

        # Format row using col_widths
        ci = 0
        row_parts = [f"{_short_label(shape, scenario):<{col_widths[ci]}}"]
        ci += 1
        for name, _, _ in backends:
            t = times[name]
            w = col_widths[ci]
            row_parts.append(f"{t:>{w}.1f}" if t is not None else f"{'N/A':>{w}}")
            ci += 1
        print(" | ".join(row_parts))

        del q, k, v, k_exp, v_exp, mask, mask_exp
        torch.cuda.empty_cache()

    print("-" * len(header))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton SDPA vs PyTorch backends"
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="all",
        help="Which shape set to benchmark (default: all)",
    )
    parser.add_argument("--num_warmup", type=int, default=25)
    parser.add_argument("--num_iters", type=int, default=100)
    args = parser.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    for s in scenarios:
        run_benchmark(
            scenario=s,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
        )


if __name__ == "__main__":
    main()
