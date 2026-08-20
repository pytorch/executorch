# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Microbenchmark for the ``channelwise_gated_delta_rule`` custom op.

Times the fused recurrent kernel in isolation (no model) at representative
GatedDeltaNet sizes, for both the decode (T=1) and prefill (T>1) regimes. Use it
to compare kernel variants (e.g. naive vs. fused) — rebuild the custom op, run
this, and diff the numbers.

Run (in an env where the custom op is built):
    python -m executorch.extension.llm.custom_ops.bench_channelwise_gated_delta_rule
"""

import time

import torch

from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

_OP = torch.ops.llama.channelwise_gated_delta_rule.default


def _make_inputs(b: int, h: int, t: int, k: int, v: int):
    return (
        torch.randn(b, h, t, k),
        torch.randn(b, h, t, k),
        torch.randn(b, h, t, v),
        torch.rand(b, h, t, k),
        torch.rand(b, h, t),
        torch.randn(b, h, k, v),
    )


def _bench_one(
    b: int, h: int, t: int, k: int, v: int, iters: int, warmup: int
) -> tuple[float, float]:
    q, key, val, decay, beta, state = _make_inputs(b, h, t, k, v)
    for _ in range(warmup):
        _OP(q, key, val, decay, beta, state)
    start = time.perf_counter()
    for _ in range(iters):
        _OP(q, key, val, decay, beta, state)
    elapsed = time.perf_counter() - start
    ms_per_call = elapsed / iters * 1e3
    us_per_token = ms_per_call * 1e3 / t
    return ms_per_call, us_per_token


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)  # single-thread: the kernel has no internal threading

    k = v = 128  # head dim (KDA / Qwen3-Next GatedDeltaNet)
    # (label, batch, heads, seq_len, iters)
    configs = [
        ("decode  T=1", 1, 32, 1, 2000),
        ("prefill T=128", 1, 32, 128, 200),
        ("prefill T=512", 1, 32, 512, 50),
    ]

    print(
        f"channelwise_gated_delta_rule microbenchmark "
        f"(K=V={k}, fp32, 1 thread, state={32 * k * v * 4 // 1024}KB)"
    )
    print(f"{'config':<15}{'B':>3}{'H':>4}{'T':>6}{'ms/call':>12}{'us/token':>12}")
    for label, b, h, t, iters in configs:
        ms, us = _bench_one(b, h, t, k, v, iters=iters, warmup=max(10, iters // 10))
        print(f"{label:<15}{b:>3}{h:>4}{t:>6}{ms:>12.4f}{us:>12.3f}")


if __name__ == "__main__":
    main()
