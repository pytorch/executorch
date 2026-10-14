# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Large-row Q4_K/Q5_K/Q6_K GEMM with groupwise Q8 activations.

Packed weights are expanded into two signed INT8 byte matrices in a preparation
kernel. The low and high bytes exactly represent integer weight codes multiplied
by integer group scales. Q4/Q5 centering residuals use explicit FMA before BF16 rounding and are restored
with a third dot, preserving the original Q8 MMQ arithmetic. BF16 activations use
Q8 K256 blocks, FP32 scales, and INT32 sums over K32 groups.

The byte matrices are temporary for each linear call. Moving unpacking and scale
multiplication out of the GEMM loop reduces work repeated across output row tiles,
at the cost of roughly two bytes of temporary storage per weight element.
"""
import torch
import triton
import triton.language as tl
from executorch.backends.cuda.triton.kernels.q8_mmq_autotune import (
    _precompile_untuned_configs,
    _tuned_kernel,
    autotune_do_bench,
)
from torch.library import wrap_triton

MMQ_BUCKETS = (128, 256, 512, 1024, 2048, 4096)


@triton.jit
def _quantize_mmq_activation(
    x,
    qx,
    scales,
    sums,
    M,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    SUM_GROUP: tl.constexpr = 0,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    groups = tl.program_id(1) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    ks = groups[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    v = tl.load(
        x + rows[:, None, None] * K + ks[None, :, :],
        (rows[:, None, None] < M) & (ks[None, :, :] < K),
        other=0,
    ).to(tl.float32)
    maximum = tl.max(tl.abs(v), 2)
    scale = tl.where(maximum > 0, maximum * (1.0 / 127.0), 1.0)
    quantized = tl.extra.cuda.libdevice.nearbyint(tl.div_rn(v, scale[:, :, None]))
    quantized = tl.minimum(tl.maximum(quantized, -127.0), 127.0).to(tl.int8)
    tl.store(
        qx + rows[:, None, None] * K + ks[None, :, :],
        quantized,
        (rows[:, None, None] < M) & (ks[None, :, :] < K),
    )
    meta = rows[:, None] * (K // GROUP_SIZE) + groups[None, :]
    mask = (rows[:, None] < M) & (groups[None, :] < K // GROUP_SIZE)
    tl.store(scales + meta, scale, mask)
    sum_size: tl.constexpr = SUM_GROUP if SUM_GROUP else GROUP_SIZE
    subgroups: tl.constexpr = GROUP_SIZE // sum_size
    grouped_q = tl.reshape(quantized, (BLOCK_ROWS, BLOCK_GROUPS, subgroups, sum_size))
    sum_values = tl.sum(grouped_q.to(tl.int32), 3)
    sum_groups = groups[:, None] * subgroups + tl.arange(0, subgroups)[None, :]
    tl.store(
        sums + rows[:, None, None] * (K // sum_size) + sum_groups[None, :, :],
        sum_values,
        (rows[:, None, None] < M) & (sum_groups[None, :, :] < K // sum_size),
    )


@triton.jit
def _prepare_mmq_weight(
    ql,
    qh,
    sc,
    step,
    zc,
    zstep,
    lo,
    hi,
    res,
    N: tl.constexpr,
    K: tl.constexpr,
    BITS: tl.constexpr,
    ROWS: tl.constexpr = 4,
):
    ns = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    ks = tl.program_id(1) * 256 + tl.arange(0, 256)
    raw = tl.load(
        ql + ns[:, None] * (K // 2) + ks[None, :] // 2, ns[:, None] < N, other=0
    ).to(tl.uint32)
    q = (raw >> ((ks[None, :] % 2) * 4)) & 15
    if BITS == 5:
        high = tl.load(
            qh + ns[:, None] * (K // 8) + ks[None, :] // 8, ns[:, None] < N, other=0
        ).to(tl.uint32)
        shift = (ks // 2) % 4 + (ks % 2) * 4
        q |= ((high >> shift[None, :]) & 1) << 4
    elif BITS == 6:
        chunk = ks % 32
        off = (ks // 256) * 64 + ((ks % 256) // 32) * 8 + (chunk % 2) * 4 + chunk // 8
        high = tl.load(
            qh + ns[:, None] * (K // 4) + off[None, :], ns[:, None] < N, other=0
        ).to(tl.uint32)
        q = (q | (((high >> (((chunk % 8) // 2) * 2)[None, :]) & 3) << 4)).to(
            tl.int32
        ) - 32
    GS: tl.constexpr = 16 if BITS == 6 else 32
    code = tl.load(
        sc + ns[:, None] * (K // GS) + ks[None, :] // GS, ns[:, None] < N, other=0
    )
    if BITS == 6:
        code = code.to(tl.int8)
    value = q.to(tl.int32) * code.to(tl.int32)
    if BITS != 6:
        zero = tl.load(
            zc + ns[:, None] * (K // 32) + ks[None, :] // 32, ns[:, None] < N, other=0
        ).to(tl.float32)
        dz = tl.load(zstep + ns * (K // 256) + tl.program_id(1), ns < N, other=0).to(
            tl.float32
        )
        center = code.to(tl.float32) * zero * dz[:, None]
        integer = tl.minimum(
            tl.maximum(tl.extra.cuda.libdevice.nearbyint(center), -8192.0), 8192.0
        ).to(tl.int32)
        value -= integer
        # The old GEMM fuses center_integer - code_product * zstep into FMA.
        # Preserve that rounding when preparing a BF16 residual in global memory.
        tl.store(
            res + ns[:, None] * (K // 32) + ks[None, :] // 32,
            tl.fma(
                -(code.to(tl.int32) * zero.to(tl.int32)).to(tl.float32),
                dz[:, None],
                integer.to(tl.float32),
            ).to(tl.bfloat16),
            (ns[:, None] < N) & (ks[None, :] % 32 == 0),
        )
    low = value.to(tl.int8)
    tl.store(lo + ns[:, None] * K + ks[None, :], low, ns[:, None] < N)
    tl.store(
        hi + ns[:, None] * K + ks[None, :],
        ((value - low.to(tl.int32)) >> 8).to(tl.int8),
        ns[:, None] < N,
    )


@triton.autotune(
    configs=[
        triton.Config({"BM": m, "BN": n, "BK": k}, num_warps=w, num_stages=s)
        for m, n, k, w, s in [
            (64, 64, 256, 4, 2),
            (128, 64, 256, 8, 2),
            (128, 128, 256, 8, 1),
            (128, 128, 256, 8, 2),
            (256, 64, 256, 8, 1),
            (64, 128, 256, 8, 2),
            (128, 128, 256, 8, 3),
            (64, 256, 256, 8, 2),
            (128, 128, 128, 8, 2),
            (256, 64, 128, 8, 2),
            (128, 256, 128, 8, 2),
            (128, 128, 128, 8, 3),
        ]
    ],
    key=["M", "N", "K", "BITS"],
    do_bench=autotune_do_bench,
)
@triton.jit
def _q8_mmq_kernel(
    qx,
    sx,
    sums,
    lo,
    hi,
    res,
    step,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    mt = tl.cdiv(M, BM)
    nt = tl.cdiv(N, BN)
    pid = tl.program_id(0)
    first = (pid // (8 * nt)) * 8
    gm = tl.minimum(mt - first, 8)
    rows = (first + (pid % (8 * nt)) % gm) * BM + tl.arange(0, BM)
    cols = ((pid % (8 * nt)) // gm) * BN + tl.arange(0, BN)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(K // 256):
        dl = tl.full((BM, BN), 0, tl.int32)
        dh = tl.full((BM, BN), 0, tl.int32)
        for sub in tl.static_range(256 // BK):
            ks = block * 256 + sub * BK + tl.arange(0, BK)
            a = tl.load(
                qx + rows[:, None] * K + ks[None, :], rows[:, None] < M, other=0
            )
            bl = tl.load(
                lo + cols[:, None] * K + ks[None, :], cols[:, None] < N, other=0
            )
            bh = tl.load(
                hi + cols[:, None] * K + ks[None, :], cols[:, None] < N, other=0
            )
            dl = tl.dot(a, tl.trans(bl), dl)
            dh = tl.dot(a, tl.trans(bh), dh)
        dd = (dl + dh * 256).to(tl.float32)
        if BITS != 6:
            gs = block * 8 + tl.arange(0, 32)
            sm = tl.load(
                sums + rows[:, None] * (K // 32) + gs[None, :],
                (rows[:, None] < M) & (tl.arange(0, 32)[None, :] < 8),
                other=0,
            ).to(tl.bfloat16)
            rr = tl.load(
                res + cols[:, None] * (K // 32) + gs[None, :],
                (cols[:, None] < N) & (tl.arange(0, 32)[None, :] < 8),
                other=0,
            )
            dd += tl.dot(sm, tl.trans(rr))
        xs = tl.load(sx + rows * (K // 256) + block, rows < M, other=0)
        ws = tl.load(step + cols * (K // 256) + block, cols < N, other=0).to(tl.float32)
        acc += dd * (xs[:, None] * ws[None, :])
    tl.store(
        out + rows[:, None] * N + cols[None, :],
        acc.to(tl.bfloat16),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def launch_q8_mmq(x, ql, qh, sc, step, zc, zstep, *, bits):
    m, k = x.shape
    n = ql.shape[0]
    qx = torch.empty((m, k), device=x.device, dtype=torch.int8)
    sx = torch.empty((m, k // 256), device=x.device, dtype=torch.float32)
    sums = torch.empty((m, k // 32), device=x.device, dtype=torch.int32)
    lo = torch.empty((n, k), device=x.device, dtype=torch.int8)
    hi = torch.empty_like(lo)
    res = torch.empty((n, k // 32), device=x.device, dtype=torch.bfloat16)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    wrap_triton(_quantize_mmq_activation)[
        (triton.cdiv(m, 8), triton.cdiv(k // 256, 4))
    ](x, qx, sx, sums, m, k, 256, 8, 4, 32)
    wrap_triton(_prepare_mmq_weight)[(triton.cdiv(n, 4), k // 256)](
        ql, qh, sc, step, zc, zstep, lo, hi, res, n, k, bits, 4
    )
    args = (qx, sx, sums, lo, hi, res, step, out, m, n, k)
    kwargs = {"BITS": bits}
    _precompile_untuned_configs(_q8_mmq_kernel, args, kwargs)
    wrap_triton(_tuned_kernel(_q8_mmq_kernel, args, kwargs))[
        lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
    ](*args, **kwargs)
    return out
