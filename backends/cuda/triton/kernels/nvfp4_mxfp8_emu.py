# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""MXFP8 x MXFP8 linears for NVFP4 weights on GPUs without FP8 tensor cores.

Emulates ``nvfp4_w4a8`` (``nvfp4_w4a8.py``) on the dequant + BF16 GEMM path:

- activations are quantized to MXFP8 (E4M3 x UE8M0 per 32 along K, the same
  ``ceil(log2(amax / 448))`` exponent) and dequantized in one pass; MXFP8
  values are exact in BF16;
- the weight dequant applies the same NVFP4 -> MXFP8 conversion the native
  kernel does per tile (exact E2M1 x E4M3 decode, UE8M0 per 32 from the two
  K16 scales, FP16 product, RNE cast to E4M3, satfinite), so the BF16 weights
  sit exactly on that kernel's grid;
- the BF16 GEMM has exact products and FP32 accumulation.

Only the accumulation order differs from the block-scaled MMA.
"""

import torch
import triton
import triton.language as tl

from executorch.backends.cuda.triton.kernels.nvfp4_kv import (
    _e4m3_encode_nonneg,
    cuda_arch,
)
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    _e4m3_to_f32,
    _fp4,
)
from executorch.backends.cuda.triton.kernels.nvfp4_w4a8 import _ceil_log2, _pow2
from torch.library import triton_op, wrap_triton


@triton.jit
def _e4m3_value(v, ARCH: tl.constexpr):
    """RNE(v) on the E4M3 grid, satfinite (+-448), as FP32."""
    mag = _e4m3_to_f32(
        _e4m3_encode_nonneg(tl.minimum(tl.abs(v), 448.0), ARCH), ARCH >= 89
    )
    sign = v.to(tl.uint32, bitcast=True) & 0x80000000
    return (mag.to(tl.uint32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@triton.jit
def _act_mxfp8_qdq_kernel(
    X,
    Y,
    Q,
    S,
    K: tl.constexpr,
    BK: tl.constexpr,
    STORE_CODES: tl.constexpr,
    ARCH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.arange(0, BK // 32)
    idx = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + row * K + idx, idx < K, 0).to(tl.float32)
    e = _ceil_log2(tl.max(tl.abs(x), 1) * (1.0 / 448.0))
    q = _e4m3_value(x * _pow2(-e)[:, None], ARCH)
    tl.store(Y + row * K + idx, (q * _pow2(e)[:, None]).to(Y.dtype.element_ty), idx < K)
    if STORE_CODES:
        sign = ((q.to(tl.uint32, bitcast=True) >> 24) & 128).to(tl.uint8)
        code = _e4m3_encode_nonneg(tl.minimum(tl.abs(q), 448.0), ARCH) | sign
        tl.store(Q + row * K + idx, code, idx < K)
        tl.store(S + row * (K // 32) + group, (e + 127).to(tl.uint8), group < K // 32)


def act_mxfp8_qdq_codes(x, store_codes=False):
    """(BF16 QDQ values, E4M3 codes, UE8M0 scales); codes only with ``store_codes``."""
    m, k = x.shape
    y = torch.empty((m, k), device=x.device, dtype=torch.bfloat16)
    q = torch.empty((m, k) if store_codes else (1,), device=x.device, dtype=torch.uint8)
    s = torch.empty(
        (m, k // 32) if store_codes else (1,), device=x.device, dtype=torch.uint8
    )
    wrap_triton(_act_mxfp8_qdq_kernel)[(m,)](
        x,
        y,
        q,
        s,
        k,
        triton.next_power_of_2(k),
        store_codes,
        cuda_arch(),
        num_warps=8,
    )
    return y, q, s


@triton_op("triton::nvfp4_mxfp8_act_qdq", mutates_args={})
def mxfp8_act_qdq(x: torch.Tensor) -> torch.Tensor:
    """BF16 [M, K] -> BF16 values exactly on the MXFP8 grid (E4M3 x UE8M0 per 32)."""
    return act_mxfp8_qdq_codes(x)[0]


@triton.jit
def _dequant_mxfp8_kernel(
    Q,
    S,
    Y,
    N,
    K: tl.constexpr,
    BN: tl.constexpr,
    BKH: tl.constexpr,
    ARCH: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    kk = tl.program_id(1) * BKH + tl.arange(0, BKH)
    sk = tl.program_id(1) * (BKH // 8) + tl.arange(0, BKH // 8)
    mask = (n[:, None] < N) & (kk[None, :] < K // 2)
    q = tl.load(Q + n[:, None].to(tl.int64) * (K // 2) + kk[None, :], mask, 0)
    s = _e4m3_to_f32(
        tl.load(
            S + n[:, None].to(tl.int64) * (K // 16) + sk[None, :],
            (n[:, None] < N) & (sk[None, :] < K // 16),
            0,
        ),
        ARCH >= 89,
    )
    # UE8M0 per 32 from the two K16 scales: |e2m1 * s16| <= 6 * max(s16 pair).
    pair = s.reshape((BN, BKH // 16, 2))
    e = _ceil_log2(tl.max(pair, 2) * (6.0 / 448.0))
    m16 = (
        (pair * _pow2(-e)[:, :, None])
        .reshape((BN, BKH // 8))
        .to(tl.float16)
        .to(tl.float32)
    )
    mb = tl.broadcast_to(m16[:, :, None], (BN, BKH // 8, 8)).reshape((BN, BKH))
    lo = _e4m3_value((_fp4(q & 15) * mb).to(tl.float16).to(tl.float32), ARCH)
    hi = _e4m3_value((_fp4(q >> 4) * mb).to(tl.float16).to(tl.float32), ARCH)
    p2 = tl.broadcast_to(_pow2(e)[:, :, None], (BN, BKH // 16, 16)).reshape((BN, BKH))
    w = tl.join(
        (lo * p2).to(Y.dtype.element_ty), (hi * p2).to(Y.dtype.element_ty)
    ).reshape((BN, 2 * BKH))
    k = tl.program_id(1) * (2 * BKH) + tl.arange(0, 2 * BKH)
    tl.store(
        Y + n[:, None].to(tl.int64) * K + k[None, :],
        w,
        (n[:, None] < N) & (k[None, :] < K),
    )


@triton_op("triton::nvfp4_dequant_mxfp8", mutates_args={})
def nvfp4_dequant_mxfp8(qdata: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed NVFP4 [N, K/2] + E4M3 [N, K/16] -> BF16 [N, K] on the MXFP8 grid
    (per-tensor scale excluded), bit-identical to the native kernel's conversion."""
    n, half_k = qdata.shape
    k = half_k * 2
    out = torch.empty((n, k), device=qdata.device, dtype=torch.bfloat16)
    bn, bkh = 32, 128
    wrap_triton(_dequant_mxfp8_kernel)[(triton.cdiv(n, bn), triton.cdiv(half_k, bkh))](
        qdata, scale, out, n, k, bn, bkh, cuda_arch(), num_warps=4
    )
    return out
