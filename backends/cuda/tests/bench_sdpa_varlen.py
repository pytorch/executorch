# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark triton::sdpa_varlen against the cell layout's triton::sdpa.

Muse Glimmer 30B attention (H_q=32, H_kv=2, D=128). Each case is a step of
the batching runtime: per sequence, the query rows it adds and the context it
then holds. The cell baseline sweeps one shared pool holding every sequence,
masked per row; sdpa_varlen reads each sequence's own column.

usage: python bench_sdpa_varlen.py [--check]
"""

import argparse

import torch
import triton

from executorch.backends.cuda.triton.kernels.sdpa import sdpa
from executorch.backends.cuda.triton.kernels.sdpa_varlen import sdpa_varlen

HQ, HKV, D = 32, 2, 128
WINDOW = 2048

# (name, [(query rows, context after the step)] per sequence)
CASES = [
    ("decode c1 ctx600", [(1, 600)]),
    ("decode c4 ctx600", [(1, 600)] * 4),
    ("decode c4 ctx4k", [(1, 4096)] * 4),
    ("decode c4 mixed", [(1, 300), (1, 900), (1, 2500), (1, 4096)]),
    ("verify c1 16x600", [(16, 600)]),
    ("verify c4 16x600", [(16, 600)] * 4),
    ("verify c4 16x4k", [(16, 4096)] * 4),
    ("prefill 97", [(97, 97)]),
    ("prefill 512 + 3 decode", [(512, 512), (1, 600), (1, 600), (1, 600)]),
    ("prefill 4096", [(4096, 4096)]),
]


def _step(case, window, device="cuda"):
    t = sum(q for q, _ in case)
    cols = len(case) + 1
    rows = max(c for _, c in case)
    q = torch.randn(1, HQ, t, D, device=device, dtype=torch.bfloat16)
    k_seq = torch.randn(rows, cols, HKV, D, device=device, dtype=torch.bfloat16)
    v_seq = torch.randn_like(k_seq)
    info, start = [], 0
    for col, (q_len, ctx) in enumerate(case):
        info.append((start, q_len, col, ctx))
        start += q_len
    info = torch.tensor(info, dtype=torch.int32, device=device)

    # Cell layout: every sequence's context packed into one pool, each row's
    # visibility in an explicit mask, the sweep bounded by the occupied extent.
    pool = sum(c for _, c in case)
    k_pool = torch.empty(1, pool, HKV, D, device=device, dtype=torch.bfloat16)
    v_pool = torch.empty_like(k_pool)
    mask = torch.zeros(1, 1, t, pool, device=device, dtype=torch.bool)
    base, row = 0, 0
    for col, (q_len, ctx) in enumerate(case):
        k_pool[0, base : base + ctx] = k_seq[:ctx, col]
        v_pool[0, base : base + ctx] = v_seq[:ctx, col]
        for r in range(q_len):
            p = ctx - q_len + r
            lo = max(0, p - window + 1) if window else 0
            mask[0, 0, row + r, base + lo : base + p + 1] = True
        base += ctx
        row += q_len
    read_len = torch.tensor([pool], device=device, dtype=torch.int64)
    return q, k_seq, v_seq, info, k_pool, v_pool, mask, read_len


def _time(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return triton.testing.do_bench(graph.replay, warmup=25, rep=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    scale = D**-0.5
    print(f"{'case':28s} {'window':>6s} {'cell ms':>9s} {'varlen ms':>10s} {'speedup':>8s}")
    for name, case in CASES:
        for window in (0, WINDOW):
            q, k, v, info, k_pool, v_pool, mask, read_len = _step(case, window)
            varlen = lambda: sdpa_varlen(q, k, v, info, scale, True, window, 0)
            cell = lambda: sdpa(
                q, k_pool.transpose(1, 2), v_pool.transpose(1, 2), mask, 0.0,
                False, scale, True, read_len,
            )
            if args.check:
                torch.testing.assert_close(
                    varlen().float(), cell().float(), atol=2e-2, rtol=2e-2
                )
            t_cell, t_var = _time(cell), _time(varlen)
            print(f"{name:28s} {window:6d} {t_cell:9.4f} {t_var:10.4f} {t_cell / t_var:7.2f}x")


if __name__ == "__main__":
    main()
