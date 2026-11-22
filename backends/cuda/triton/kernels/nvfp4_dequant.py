# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Whole-weight NVFP4 -> BF16/FP16 dequantization for dequant + matmul prefill.

The same path the integer GGUF formats take for M > 64
(``_gemm_family_dispatch.chunked_dequant_linear``): dequantize the packed
weight once per forward and let Inductor compile the dense matmul. Each value
is ``(e2m1 * e4m3_scale * per_tensor_scale).to(dtype)``, bit-identical to the
weights the fused W4A16 kernel builds in registers, so only the GEMM's
accumulation order differs.
"""

import torch
import triton
import triton.language as tl

from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    _e4m3_f16,
    _e4m3_to_f32,
    _fp4_pair_f16,
    _has_native_fp4_conversion,
    _scaled_pair,
    _SOFT_FP4_RESCALE,
)
from torch.library import triton_op, wrap_triton


@triton.jit
def _nvfp4_dequant_kernel(
    Q,
    S,
    PTS,
    Y,
    N,
    K: tl.constexpr,
    BN: tl.constexpr,
    BKH: tl.constexpr,
    NATIVE: tl.constexpr,
    ROW_PTS: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    kk = tl.program_id(1) * BKH + tl.arange(0, BKH)
    mask = (n[:, None] < N) & (kk[None, :] < K // 2)
    q = tl.load(Q + n[:, None].to(tl.int64) * (K // 2) + kk[None, :], mask, 0)
    b = tl.load(S + n[:, None].to(tl.int64) * (K // 16) + kk[None, :] // 8, mask, 0)
    if NATIVE:
        lo, hi = _scaled_pair(q, _e4m3_to_f32(b, NATIVE), False)
    else:
        lo, hi = _fp4_pair_f16(q)
        sf = _e4m3_f16(b) * _SOFT_FP4_RESCALE
        lo = lo * sf
        hi = hi * sf
    if ROW_PTS:
        pts = tl.load(PTS + n, n < N, 1.0)[:, None]
    else:
        pts = tl.load(PTS)
    w0 = (lo * pts).to(Y.dtype.element_ty)
    w1 = (hi * pts).to(Y.dtype.element_ty)
    w = tl.join(w0, w1).reshape((BN, 2 * BKH))
    k = tl.program_id(1) * (2 * BKH) + tl.arange(0, 2 * BKH)
    tl.store(
        Y + n[:, None].to(tl.int64) * K + k[None, :],
        w,
        (n[:, None] < N) & (k[None, :] < K),
    )


@triton_op("triton::nvfp4_dequant", mutates_args={})
def nvfp4_dequant(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Packed NVFP4 [N, K/2] + E4M3 [N, K/16] (+ global scale) -> dense [N, K]."""
    n, half_k = qdata.shape
    k = half_k * 2
    out = torch.empty((n, k), device=qdata.device, dtype=dtype)
    bn, bkh = 32, 128
    wrap_triton(_nvfp4_dequant_kernel)[(triton.cdiv(n, bn), triton.cdiv(half_k, bkh))](
        qdata,
        scale,
        per_tensor_scale,
        out,
        n,
        k,
        bn,
        bkh,
        _has_native_fp4_conversion(),
        per_tensor_scale.numel() != 1,
        num_warps=4,
    )
    return out
