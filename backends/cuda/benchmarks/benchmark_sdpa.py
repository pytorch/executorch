#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark the Triton SDPA kernels against PyTorch SDPA backends at decode.

Cross-backend latency comparison ("is our kernel competitive vs PyTorch /
Flash?") across a few representative decode configs and the L_kv range, in BOTH
CUDA-graph and plain timing modes. The ET Triton kernels use native GQA; the
Flash/Efficient/Math backends require pre-expanded KV (no native GQA), matching
the test reference. PyTorch (default) is the correctness reference.

Timing: CUDA-graph mode (capture+replay) is faithful to the deployed
``--cuda_graph`` runtime; plain ``do_bench`` charges each kernel its full
per-call launch/alloc overhead. Run both to see the effect (it is large for ET
split-K, which allocates partial buffers per call).

Usage:
    python benchmark_sdpa.py                 # both timing modes
    python benchmark_sdpa.py --mode cudagraph
    python benchmark_sdpa.py --mode plain
"""

import argparse
import statistics
import warnings
from functools import partial

import torch
import torch.nn.functional as F

from executorch.backends.cuda.triton.kernels.sdpa import (
    sdpa as _triton_sdpa,
    sdpa_decode_splitk as _triton_splitk,
)
from torch.nn.attention import sdpa_kernel, SDPBackend
from triton.testing import do_bench, do_bench_cudagraph


# -- Timing primitive + ET kernel runners (self-contained) -------------------
# do_bench budgets are millisecond windows (NOT iteration counts).
_WARMUP_MS = 10
_REP_MS = 50
# Warmup calls before graph capture so the Triton autotuner has cached a config
# (autotuning cannot run inside graph capture).
_GRAPH_WARMUP_CALLS = 20


def run_standard(q, k, v, attn_mask, enable_gqa):
    return _triton_sdpa(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)


def run_splitk(q, k, v, attn_mask, enable_gqa):
    return _triton_splitk(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)


def time_us(fn, cudagraph: bool = True) -> float:
    """Median latency (us). cudagraph=True is faithful to the --cuda_graph path.

    Under CUDA-graph the op is captured once (its split-K partial/LSE workspace
    is allocated once into the graph's private pool and reused across replays)
    and only replay() is timed, so the per-call buffer alloc + launch overhead
    is excluded -- exactly as the deployed runtime eliminates it. We warm up
    first so the Triton autotuner has cached a config before capture.
    """
    if cudagraph:
        for _ in range(_GRAPH_WARMUP_CALLS):
            fn()
        torch.cuda.synchronize()
        ms = do_bench_cudagraph(fn, rep=_REP_MS, return_mode="median")
    else:
        ms = do_bench(fn, warmup=_WARMUP_MS, rep=_REP_MS, return_mode="median")
    return ms * 1000.0


# Each reported number repeats the timing primitive N_RUNS times, discards the
# first N_WARMUP as warmup, and reports mean +/- std over the remaining runs.
N_RUNS = 10
N_WARMUP = 4


def measure_us(fn, cudagraph: bool):
    """Repeat time_us N_RUNS times; return (mean, std) over runs[N_WARMUP:]."""
    samples = [time_us(fn, cudagraph=cudagraph) for _ in range(N_RUNS)]
    kept = samples[N_WARMUP:]
    mean = statistics.fmean(kept)
    std = statistics.stdev(kept) if len(kept) > 1 else 0.0
    return mean, std


# PyTorch's Flash/Efficient backends don't support GQA (H_q != H_kv) directly.
# We expand KV heads via repeat_interleave so they can run, matching what the
# test reference does. This measures the kernel itself, not GQA dispatch.


def _expand_kv(k, v, num_groups):
    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)
    return k, v


def _expand_mask(mask, H_q):
    if mask is not None and mask.shape[1] == 1 and H_q > 1:
        mask = mask.expand(-1, H_q, -1, -1)
    return mask


def _run_pytorch_default(q, k, v, attn_mask, enable_gqa):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa
    )


def _make_pytorch_runner(backend: SDPBackend):
    def run(q, k, v, attn_mask, enable_gqa):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    return run


# Flash doesn't support attn_mask at all, only is_causal. Our benchmark mask is
# all-ones, so no mask is equivalent.
def _run_flash(q, k, v, attn_mask, enable_gqa):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v)


# ET Triton kernels reuse the shared helper runners (the real lowered kernels).
BACKENDS = {
    "triton": ("ET Triton (GQA)", run_standard),
    "splitk": ("ET Split-K (GQA)", run_splitk),
    "pytorch": ("PyTorch", _run_pytorch_default),
    "flash": ("Flash (exp KV)", _run_flash),
    "efficient": (
        "Efficient (exp KV)",
        _make_pytorch_runner(SDPBackend.EFFICIENT_ATTENTION),
    ),
    "math": ("Math (exp KV)", _make_pytorch_runner(SDPBackend.MATH)),
}

# Backends that need KV heads expanded before calling (no native GQA support).
_NEEDS_KV_EXPAND = {"flash", "efficient", "math"}

# Representative decode configs (label, B, H_q, H_kv, D). CTA = B * H_kv.
CONFIGS = [
    ("gemma sliding (D=256, CTA=16)", 1, 32, 16, 256),
    ("qwen (D=256, CTA=2)", 1, 16, 2, 256),
    ("head_dim=128 (D=128, CTA=16)", 1, 32, 16, 128),
]

L_KV_RANGE = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Cross-backend validation tolerance (bf16 vs bf16).
MAX_ABS_TOL = 1e-2


def _make_tensors(B, H_q, H_kv, Lq, Lk, D, device="cuda", dtype=torch.bfloat16):
    q = torch.randn(B, H_q, Lq, D, device=device, dtype=dtype)
    k = torch.randn(B, H_kv, Lk, D, device=device, dtype=dtype)
    v = torch.randn(B, H_kv, Lk, D, device=device, dtype=dtype)
    mask = torch.ones(B, 1, Lq, Lk, dtype=torch.bool, device=device)
    enable_gqa = H_q != H_kv
    num_groups = H_q // H_kv
    k_exp, v_exp = _expand_kv(k, v, num_groups)
    mask_exp = _expand_mask(mask, H_q)
    return q, k, v, k_exp, v_exp, mask, mask_exp, enable_gqa


def _max_abs_error(out, ref):
    return (out.float() - ref.float()).abs().max().item()


def _try_run(run_fn, q, k, v, mask, enable_gqa):
    try:
        return run_fn(q, k, v, mask, enable_gqa)
    except Exception:
        return None


def _try_bench(run_fn, q, k, v, mask, enable_gqa, cudagraph):
    """Benchmark one backend, returning (mean_us, std_us) or None on failure."""
    fn = partial(run_fn, q, k, v, mask, enable_gqa)
    try:
        run_fn(q, k, v, mask, enable_gqa)
        return measure_us(fn, cudagraph=cudagraph)
    except Exception:
        return None


def _bench_inputs(name, q, k, v, k_exp, v_exp, mask, mask_exp):
    """Return the (k, v, mask) a backend should use (expanded or native)."""
    if name in _NEEDS_KV_EXPAND:
        return k_exp, v_exp, mask_exp
    return k, v, mask


@torch.inference_mode()
def run_benchmark(cudagraph: bool):
    """Print a cross-backend decode latency table for each config."""
    backends = [(name, *BACKENDS[name]) for name in BACKENDS]
    mode = "CUDA-graph (capture+replay)" if cudagraph else "plain do_bench"
    device = torch.cuda.get_device_name()
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count

    print()
    print("=" * 124)
    print(f"SDPA decode cross-backend benchmark   |   timing: {mode}")
    print(f"  device: {device} (n_SM={n_sm})   L_q=1, bf16, all-ones mask")
    print(f"  backends: {', '.join(label for _, label, _ in backends)}")
    print(
        f"  each cell = mean+/-std us over last {N_RUNS - N_WARMUP} of {N_RUNS} "
        f"runs ({N_WARMUP} warmup)"
    )
    print("=" * 124)

    for label, B, H_q, H_kv, D in CONFIGS:
        print(f"\n{label}   [B={B} H_q={H_q} H_kv={H_kv} D={D}]")
        col_specs = [("L_kv", "", 6)] + [(lbl, "(us)", 13) for _, lbl, _ in backends]
        widths = [max(len(h), len(u), mw) for h, u, mw in col_specs]
        header = " | ".join(
            f"{h:<{w}}" if i == 0 else f"{h:>{w}}"
            for i, ((h, _, _), w) in enumerate(zip(col_specs, widths))
        )
        units = " | ".join(
            f"{'':>{w}}" if i == 0 else f"{u:>{w}}"
            for i, ((_, u, _), w) in enumerate(zip(col_specs, widths))
        )
        print("  " + header)
        print("  " + units)
        print("  " + "-" * len(header))

        for Lk in L_KV_RANGE:
            q, k, v, k_exp, v_exp, mask, mask_exp, enable_gqa = _make_tensors(
                B, H_q, H_kv, 1, Lk, D
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Correctness: validate every backend against PyTorch (default).
                outputs = {}
                for name, _lbl, run_fn in backends:
                    bk, bv, bmask = _bench_inputs(
                        name, q, k, v, k_exp, v_exp, mask, mask_exp
                    )
                    outputs[name] = _try_run(run_fn, q, bk, bv, bmask, enable_gqa)
                ref = outputs.get("pytorch")
                if ref is not None:
                    for name, lbl, _ in backends:
                        if name == "pytorch" or outputs[name] is None:
                            continue
                        err = _max_abs_error(outputs[name], ref)
                        assert err < MAX_ABS_TOL, (
                            f"Output mismatch {label} L_kv={Lk}: {lbl} vs PyTorch, "
                            f"max abs error {err:.3e} >= {MAX_ABS_TOL}"
                        )
                del outputs

                times = {}
                for name, _lbl, run_fn in backends:
                    bk, bv, bmask = _bench_inputs(
                        name, q, k, v, k_exp, v_exp, mask, mask_exp
                    )
                    times[name] = _try_bench(
                        run_fn, q, bk, bv, bmask, enable_gqa, cudagraph
                    )

            row = [f"{Lk:<{widths[0]}}"]
            for ci, (name, _, _) in enumerate(backends, start=1):
                t = times[name]
                if t is not None:
                    cell = f"{t[0]:.1f}\u00b1{t[1]:.1f}"
                else:
                    cell = "N/A"
                row.append(f"{cell:>{widths[ci]}}")
            print("  " + " | ".join(row))

            del q, k, v, k_exp, v_exp, mask, mask_exp
            torch.cuda.empty_cache()
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton SDPA vs PyTorch backends (decode)"
    )
    parser.add_argument(
        "--mode",
        choices=["cudagraph", "plain", "both"],
        default="both",
        help="Timing mode(s) to run (default: both).",
    )
    args = parser.parse_args()

    if args.mode in ("cudagraph", "both"):
        run_benchmark(cudagraph=True)
    if args.mode in ("plain", "both"):
        run_benchmark(cudagraph=False)


if __name__ == "__main__":
    main()
