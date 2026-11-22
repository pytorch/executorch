# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Whole-weight NVFP4 -> BF16/FP16 dequantization for dequant + matmul prefill.

The same path the integer GGUF formats take above their buckets
(``_gemm_family_dispatch.chunked_dequant_linear``): dequantize the packed
weight once per forward and let Inductor compile the dense matmul. Each value
is ``(e2m1 * e4m3_scale * per_tensor_scale).to(dtype)``, bit-identical to the
weights the fused W4A16 kernel builds in registers (the exact software decode
on every GPU: the kernel is memory bound), so only the GEMM's accumulation
order differs.
"""

import torch
import triton
import triton.language as tl

from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    _check_weight,
    _decode_pair,
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
    ROW_PTS: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    kk = tl.program_id(1) * BKH + tl.arange(0, BKH)
    mask = (n[:, None] < N) & (kk[None, :] < K // 2)
    q = tl.load(Q + n[:, None].to(tl.int64) * (K // 2) + kk[None, :], mask, 0)
    b = tl.load(S + n[:, None].to(tl.int64) * (K // 16) + kk[None, :] // 8, mask, 0)
    lo, hi = _decode_pair(q, b, False)
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
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Packed NVFP4 [N, K/2] + E4M3 [N, K/16] (+ global scale) -> dense [N, K].

    Scale bytes must be finite E4M3 (no 0x7F/0xFF), as
    ``CudaNVFP4Tensor.from_exportable`` checks; it is not checked here."""
    _check_weight(qdata, scale, per_tensor_scale)
    if output_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"NVFP4 dequant supports BF16/FP16 output, got {output_dtype}")
    n, half_k = qdata.shape
    k = half_k * 2
    out = torch.empty((n, k), device=qdata.device, dtype=output_dtype)
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
        per_tensor_scale.numel() != 1,
        num_warps=4,
    )
    return out
