# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""MXFP8 x MXFP8 prefill GEMM for NVFP4-stored weights.

Activations are quantized to MXFP8 in one pass: E4M3 elements with one UE8M0
scale per 32 elements along K, the scale exponent being
``ceil(log2(amax / 448))`` so the block maximum never saturates. (The OCP MX
specification and torchao default to ``floor(log2(amax)) - 8``, which can clip
the block maximum; this module rounds the exponent up instead.)

Weights stay packed NVFP4 in memory. Inside the GEMM each weight tile is
converted in registers: E2M1 codes decode exactly to FP16
(``cvt.rn.f16x2.e2m1x2``), each K32 group gets a UE8M0 exponent
``e = ceil(log2(6 * max(s16 pair) / 448))``, every element becomes
``code * (s16 * 2**-e)`` in FP16 (exact: at most 6 significant bits) and is
rounded once, to nearest even, to E4M3. The product runs on the block-scaled
tensor cores (``tl.dot_scaled`` e4m3 x e4m3 with ue8m0 scales) with FP32
accumulation, and the per-tensor scale is applied in the epilogue.

The conversion is exact for products with at most 4 significant bits; the
others (rare in practice) move by at most half an E4M3 ulp.

Packed weights: low nibble first; scales N-major (N, K/16) E4M3.
"""

import torch
import triton
import triton.language as tl

from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import _check_operands
from executorch.backends.cuda.triton.kernels.nvfp4_tuning import (
    autotune_do_bench,
    launch_nvfp4_kernel,
    prune_nvfp4_stages,
)
from torch.library import triton_op


@triton.jit
def _ceil_log2(r):
    # ceil(log2(r)) for r >= 0 (r == 0 -> -127), clamped to the UE8M0 range.
    bits = r.to(tl.int32, bitcast=True)
    ex = (bits >> 23) & 255
    e = ex - 127 + ((bits & 8388607) != 0).to(tl.int32)
    e = tl.where(ex == 0, -127, e)
    return tl.minimum(tl.maximum(e, -127), 127)


@triton.jit
def _pow2(e):
    # 2**e for e in [-126, 127] as an exact FP32.
    return ((tl.minimum(tl.maximum(e, -126), 127) + 127) << 23).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _load_block_scales(
    BS, n, block, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    """E4M3 K16 block scales of a [BN, BK] weight tile as FP32, shape [BN, BK // 16]."""
    si = block * (BK // 16) + tl.arange(0, BK // 16)
    s = tl.load(
        BS + n[:, None] * (K // 16) + si[None, :],
        (n[:, None] < N) & (si[None, :] < K // 16),
        0,
    )
    return s.to(tl.float8e4nv, bitcast=True).to(tl.float32)


@triton.jit
def _decode_fp8_tile(
    B,
    n,
    block,
    mult,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """E4M3 [BN, BK] tile of e2m1 * mult, mult [BN, BK // 16] = s16 * 2**-e.

    cvt.e2m1x2 gives exact FP16 codes; mult is an E4M3 value shifted by a power
    of two, so code * mult is exact in FP16 and the only rounding is the final
    RNE cast to E4M3."""
    pi = block * (BK // 2) + tl.arange(0, BK // 2)
    q = tl.load(
        B + n[:, None] * (K // 2) + pi[None, :],
        (n[:, None] < N) & (pi[None, :] < K // 2),
        0,
    )
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 tmp,z1,z2,z3; mov.b32 {tmp,z1,z2,z3}, $1; cvt.rn.f16x2.e2m1x2 $0, tmp; }",
        constraints="=r,r",
        args=[q.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    m16 = mult.to(tl.float16)[:, :, None]
    lo = (packed & 65535).to(tl.uint16).to(tl.float16, bitcast=True).reshape(
        (BN, BK // 16, 8)
    ) * m16
    hi = (packed >> 16).to(tl.uint16).to(tl.float16, bitcast=True).reshape(
        (BN, BK // 16, 8)
    ) * m16
    return (
        tl.join(lo.reshape((BN, BK // 2)), hi.reshape((BN, BK // 2)))
        .reshape((BN, BK))
        .to(tl.float8e4nv)
    )


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (4, 8, 16)],
    key=["M", "K"],
    do_bench=autotune_do_bench,
)
@triton.jit
def _act_mxfp8(X, Q, S, M, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    group = tl.arange(0, BK // 32)
    idx = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + row * K + idx, idx < K, 0).to(tl.float32)
    e = _ceil_log2(tl.max(tl.abs(x), 1) * (1.0 / 448.0))
    q = (x * _pow2(-e)[:, None]).to(tl.float8e4nv)
    tl.store(Q + row * K + idx, q.to(tl.uint8, bitcast=True), idx < K)
    tl.store(S + row * (K // 32) + group, (e + 127).to(tl.uint8), group < K // 32)


_CONFIGS = [
    triton.Config(
        {"BM": bm, "BN": bn, "BK": bk, "GROUP_M": gm}, num_warps=w, num_stages=st
    )
    for bm, bn, bk, gm, w, st in [
        (64, 64, 128, 1, 4, 3),
        (64, 128, 128, 8, 4, 3),
        (128, 64, 128, 8, 4, 3),
        (128, 128, 64, 8, 4, 3),
        (128, 128, 64, 8, 8, 3),
        (128, 128, 128, 8, 4, 3),
        (128, 128, 128, 8, 8, 3),
        (128, 128, 128, 8, 8, 2),
        (256, 128, 64, 8, 8, 3),
        (256, 128, 128, 8, 8, 2),
        (128, 256, 64, 8, 8, 3),
        (128, 256, 128, 8, 8, 2),
        (256, 64, 128, 8, 8, 3),
    ]
]


@triton.jit
def _tile(
    M, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    first = (pid // (GROUP_M * nn)) * GROUP_M
    gm = tl.minimum(nm - first, GROUP_M)
    m = (first + pid % gm) * BM + tl.arange(0, BM)
    n = ((pid % (GROUP_M * nn)) // gm) * BN + tl.arange(0, BN)
    return m, n


@triton.autotune(
    configs=_CONFIGS,
    key=["M", "N", "K", "ROW_PTS"],
    prune_configs_by={"early_config_prune": prune_nvfp4_stages},
    do_bench=autotune_do_bench,
)
@triton.jit
def _w4a8_mx(
    A,
    AS,
    B,
    BS,
    PTS,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    ROW_PTS: tl.constexpr = False,
):
    m, n = _tile(M, N, BM, BN, GROUP_M)
    m64 = m.to(tl.int64)
    ki = tl.arange(0, BK)
    si = tl.arange(0, BK // 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + ki
        ss = block * (BK // 32) + si
        a = tl.load(
            A + m64[:, None] * K + kk[None, :],
            (m[:, None] < M) & (kk[None, :] < K),
            0,
        ).to(tl.float8e4nv, bitcast=True)
        sa = tl.load(
            AS + m64[:, None] * (K // 32) + ss[None, :],
            (m[:, None] < M) & (ss[None, :] < K // 32),
            0,
        )
        # UE8M0 per 32 from the two K16 scales: |e2m1 * s16| <= 6 * max(s16 pair).
        pair = _load_block_scales(BS, n, block, N, K, BN, BK).reshape((BN, BK // 32, 2))
        e = _ceil_log2(tl.max(pair, 2) * (6.0 / 448.0))
        mult = (pair * _pow2(-e)[:, :, None]).reshape((BN, BK // 16))
        w8 = _decode_fp8_tile(B, n, block, mult, N, K, BN, BK)
        acc = tl.dot_scaled(
            a, sa, "e4m3", tl.trans(w8), (e + 127).to(tl.uint8), "e4m3", acc
        )
    if ROW_PTS:
        pts = tl.load(PTS + n, n < N, 0)[None, :]
    else:
        pts = tl.load(PTS)
    tl.store(
        C + m64[:, None] * N + n[None, :],
        acc * pts,
        (m[:, None] < M) & (n[None, :] < N),
    )


def quantize_activation_mxfp8(x):
    """BF16 [M, K] -> (E4M3 codes [M, K] uint8, UE8M0 scales [M, K/32] uint8)."""
    m, k = x.shape
    q = torch.empty((m, k), device=x.device, dtype=torch.uint8)
    s = torch.empty((m, k // 32), device=x.device, dtype=torch.uint8)
    launch_nvfp4_kernel(_act_mxfp8, (m,), (x, q, s, m, k, triton.next_power_of_2(k)), 3)
    return q, s


@triton_op("triton::nvfp4_w4a8", mutates_args={})
def nvfp4_w4a8(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
) -> torch.Tensor:
    """y[M, N] = x[M, K] @ W^T: MXFP8 activations x NVFP4 weights converted to MXFP8."""
    _check_operands(x, qdata, scale, per_tensor_scale)
    m, k = x.shape
    n = qdata.shape[0]
    if k % 32:
        raise ValueError("nvfp4_w4a8 requires K % 32 == 0")
    q, s = quantize_activation_mxfp8(x)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)

    launch_nvfp4_kernel(
        _w4a8_mx,
        grid,
        (q, s, qdata, scale, per_tensor_scale, out, m, n, k),
        6,
        ROW_PTS=per_tensor_scale.numel() != 1,
    )
    return out


def nvfp4_to_mxfp8_weight_reference(qdata, scale):
    """Host reference of the in-kernel weight conversion (tests): returns the
    MXFP8 weight values and the exact NVFP4 values, per-tensor scale excluded."""
    n, half = qdata.shape
    k = half * 2
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=qdata.device,
    )
    codes = torch.stack((qdata & 15, qdata >> 4), dim=-1).reshape(n, k).long()
    sf = scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    exact = lut[codes] * sf
    g = exact.reshape(n, k // 32, 32)
    smax = scale.view(torch.float8_e4m3fn).float().reshape(n, k // 32, 2).amax(-1)
    r = smax * 6.0 / 448.0
    e = torch.where(r > 0, torch.ceil(torch.log2(r)), torch.full_like(r, -127.0)).clamp(
        -127, 127
    )
    conv = (g * torch.exp2(-e)[..., None]).to(torch.float8_e4m3fn).float()
    return (conv * torch.exp2(e)[..., None]).reshape(n, k), exact
