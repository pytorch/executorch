#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark the Triton fused MoE kernel against eager and torch.compile baselines.

Measures latency across prompt lengths matching the Qwen3.5-35B-A3B model
(hidden_size=2048, num_experts=256, top_k=8, intermediate_size=512,
INT4 weight-only quantization with group_size=128).

Usage:
    python benchmark_moe.py
    python benchmark_moe.py --prompt-lengths 1,8,64,512 --num_iters 200
"""

import argparse
from functools import partial

import executorch.backends.cuda.triton.kernels  # noqa: F401 — registers triton ops

import torch
from triton.testing import do_bench


# -- Qwen3.5-35B-A3B defaults ------------------------------------------------

DEFAULTS = {
    "num_experts": 256,
    "top_k": 8,
    "hidden_size": 2048,
    "intermediate_size": 512,
    "group_size": 128,
}

PROMPT_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4095]


# -- Weight / input generation -----------------------------------------------


def _make_int4_weights(E, N, K, group_size, device="cuda"):
    """Generate random packed INT4 weights and per-group scales.

    Returns:
        w:     [E, N, K//2] int8 — two INT4 values packed per byte
        scale: [E, N, K//group_size] bf16
    """
    vals = torch.randint(0, 16, (E, N, K), dtype=torch.uint8, device=device)
    low = vals[:, :, 0::2]
    high = vals[:, :, 1::2]
    packed = (high << 4) | low
    w = packed.to(torch.int8)

    scale = (
        torch.randn(E, N, K // group_size, device=device, dtype=torch.bfloat16) * 0.01
    )
    return w, scale


# -- Dequantization ----------------------------------------------------------


def _dequant_int4(w_packed, scale, group_size):
    """Unpack INT4 weights and dequantize.

    w_packed: [E, N, K//2] int8
    scale:    [E, N, K//group_size] bf16
    Returns:  [E, N, K] bf16
    """
    w_uint8 = w_packed.to(torch.uint8)
    low = (w_uint8 & 0xF).to(torch.float32)
    high = ((w_uint8 >> 4) & 0xF).to(torch.float32)
    E, N, Khalf = w_packed.shape
    K = Khalf * 2
    vals = torch.empty(E, N, K, device=w_packed.device, dtype=torch.float32)
    vals[:, :, 0::2] = low
    vals[:, :, 1::2] = high
    vals = vals - 8.0
    scale_expanded = scale.float().repeat_interleave(group_size, dim=2)[:, :, :K]
    return (vals * scale_expanded).to(torch.bfloat16)


# -- Backends -----------------------------------------------------------------


def _run_eager(
    hidden_states,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk_weights,
    topk_ids,
    top_k,
    num_experts,
    group_size,
):
    """Loop-based eager MoE — correctness reference only (not benchmarked)."""
    M, K = hidden_states.shape
    inter = w2.shape[2] * 2

    w1_deq = _dequant_int4(w1, w1_scale, group_size)
    w2_deq = _dequant_int4(w2, w2_scale, group_size)

    output = torch.zeros(M, K, device=hidden_states.device, dtype=torch.bfloat16)
    for i in range(M):
        for j in range(top_k):
            expert_id = topk_ids[i, j].item()
            weight = topk_weights[i, j]
            x = hidden_states[i : i + 1] @ w1_deq[expert_id].T
            gate = x[:, :inter]
            up = x[:, inter:]
            x = torch.nn.functional.silu(gate) * up
            x = x @ w2_deq[expert_id].T
            output[i] += weight * x.squeeze(0)
    return output


def _run_eager_vectorized(
    hidden_states,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk_weights,
    topk_ids,
    top_k,
    num_experts,
    group_size,
):
    """Vectorized eager — gather + bmm, no Python loops."""
    M, K = hidden_states.shape
    inter = w2.shape[2] * 2

    w1_deq = _dequant_int4(w1, w1_scale, group_size)
    w2_deq = _dequant_int4(w2, w2_scale, group_size)

    flat_ids = topk_ids.reshape(-1)
    hs_rep = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(M * top_k, K)
    gemm1_out = torch.bmm(
        hs_rep.unsqueeze(1), w1_deq[flat_ids].transpose(1, 2)
    ).squeeze(1)

    gate = gemm1_out[:, :inter]
    up = gemm1_out[:, inter:]
    act = torch.nn.functional.silu(gate) * up

    gemm2_out = torch.bmm(act.unsqueeze(1), w2_deq[flat_ids].transpose(1, 2)).squeeze(1)

    return (gemm2_out.view(M, top_k, K) * topk_weights.unsqueeze(-1)).sum(dim=1)


_compiled_fn = None


def _run_compiled(
    hidden_states,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk_weights,
    topk_ids,
    top_k,
    num_experts,
    group_size,
):
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(_run_eager_vectorized)
    return _compiled_fn(
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        top_k,
        num_experts,
        group_size,
    )


def _run_triton(
    hidden_states,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk_weights,
    topk_ids,
    top_k,
    num_experts,
    group_size,
):
    return torch.ops.triton.fused_moe(
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        top_k=top_k,
        num_experts=num_experts,
        group_size=group_size,
    )


BACKENDS = {
    "eager_vec": ("Eager (vec)", _run_eager_vectorized),
    "compile": ("Compile", _run_compiled),
    "triton": ("Triton fused", _run_triton),
}

try:
    from executorch.backends.cuda.triton.kernels.fused_moe import fused_moe_batched

    def _run_triton_batched(
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        top_k,
        num_experts,
        group_size,
    ):
        return fused_moe_batched(
            hidden_states,
            w1,
            w1_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            top_k=top_k,
            num_experts=num_experts,
            group_size=group_size,
        )

    BACKENDS["triton_batched"] = ("Triton batched", _run_triton_batched)
except ImportError:
    pass


# -- Helpers ------------------------------------------------------------------


def _max_abs_error(out, ref):
    return (out.float() - ref.float()).abs().max().item()


def _bench_ms(fn, num_warmup, num_iters):
    return do_bench(fn, warmup=num_warmup, rep=num_iters, return_mode="median")


def _try_bench(run_fn, args, num_warmup, num_iters):
    fn = partial(run_fn, **args)
    try:
        fn()
        return _bench_ms(fn, num_warmup, num_iters)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


# -- Main ---------------------------------------------------------------------


@torch.inference_mode()
def run_benchmark(
    prompt_lengths,
    num_experts,
    top_k,
    hidden_size,
    intermediate_size,
    group_size,
    num_warmup,
    num_iters,
):
    backends = [(name, *BACKENDS[name]) for name in BACKENDS]

    device_name = torch.cuda.get_device_name()
    print()
    print("=" * 100)
    print("Fused MoE Benchmark — Qwen3.5-35B-A3B (W4A16)")
    print(f"  Device: {device_name}")
    print(
        f"  Experts: {num_experts}, Top-K: {top_k}, Hidden: {hidden_size}, "
        f"Intermediate: {intermediate_size}, Group: {group_size}"
    )
    print(f"  Warmup: {num_warmup}, Iters: {num_iters}")
    print(f"  Backends: {', '.join(label for _, label, _ in backends)}")
    print("=" * 100)

    # Generate weights once (shared across prompt lengths)
    w1, w1_scale = _make_int4_weights(
        num_experts, 2 * intermediate_size, hidden_size, group_size
    )
    w2, w2_scale = _make_int4_weights(
        num_experts, hidden_size, intermediate_size, group_size
    )

    # Column layout: Shape | backend1 | backend2 | ... (dynamic widths)
    col_specs = [("M (tokens)", "", 10)]
    for _, label, _ in backends:
        col_specs.append((label, "(ms)", max(8, len(label))))

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

    for M in prompt_lengths:
        hidden_states = torch.randn(M, hidden_size, device="cuda", dtype=torch.bfloat16)
        router_logits = torch.randn(M, num_experts, device="cuda", dtype=torch.float32)
        topk_w, topk_i = torch.topk(router_logits, top_k, dim=-1)
        topk_w = torch.softmax(topk_w, dim=-1)
        topk_i = topk_i.to(torch.int64)

        common_args = {
            "hidden_states": hidden_states,
            "w1": w1,
            "w1_scale": w1_scale,
            "w2": w2,
            "w2_scale": w2_scale,
            "topk_weights": topk_w,
            "topk_ids": topk_i,
            "top_k": top_k,
            "num_experts": num_experts,
            "group_size": group_size,
        }

        # Correctness: triton vs loop-based eager reference.
        # Only check at small M to avoid slow eager loop + OOM on large M.
        if M <= 64:
            ref_out = _run_eager(**common_args)
            tri_out = _run_triton(**common_args)
            err = _max_abs_error(tri_out, ref_out)
            assert err < 2.0e-1, (
                f"Triton vs eager mismatch at M={M}: "
                f"max abs error {err:.3e} >= 2.0e-1"
            )
            del ref_out, tri_out

        # Benchmark
        times = {}
        for name, _label, run_fn in backends:
            times[name] = _try_bench(run_fn, common_args, num_warmup, num_iters)

        ci = 0
        row_parts = [f"{f'M={M}':<{col_widths[ci]}}"]
        ci += 1
        for name, _, _ in backends:
            t = times[name]
            w = col_widths[ci]
            row_parts.append(f"{t:>{w}.3f}" if t is not None else f"{'OOM':>{w}}")
            ci += 1
        print(" | ".join(row_parts))

        del hidden_states, topk_w, topk_i
        torch.cuda.empty_cache()

    print("-" * len(header))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton fused MoE vs eager/compile baselines"
    )
    parser.add_argument("--num-experts", type=int, default=DEFAULTS["num_experts"])
    parser.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    parser.add_argument("--hidden-size", type=int, default=DEFAULTS["hidden_size"])
    parser.add_argument(
        "--intermediate-size", type=int, default=DEFAULTS["intermediate_size"]
    )
    parser.add_argument("--group-size", type=int, default=DEFAULTS["group_size"])
    parser.add_argument("--num_warmup", type=int, default=25)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument(
        "--prompt-lengths",
        type=str,
        default=None,
        help="Comma-separated list of prompt lengths (default: standard sweep)",
    )
    args = parser.parse_args()

    prompt_lengths = PROMPT_LENGTHS
    if args.prompt_lengths:
        prompt_lengths = [int(x.strip()) for x in args.prompt_lengths.split(",")]

    run_benchmark(
        prompt_lengths=prompt_lengths,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        group_size=args.group_size,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )


if __name__ == "__main__":
    main()
