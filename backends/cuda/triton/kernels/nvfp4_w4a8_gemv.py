# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 W4A8 decode GEMV for GPUs without native FP4 (compute capability
below 12).

Without FP4 tensor cores the W4A16 GEMV is ALU bound: every weight is decoded to
floating point and multiplied in FP32. Here activations are quantized to int8
per 16-element block (amax / 127, the NVFP4 block size), each 32-bit word of
eight E2M1 codes becomes two words of int8 (2 * value: 0,1,2,3,4,6,8,12 and
negatives) through three PRMT byte permutes per word, and dp4a accumulates
exact integer dot products per block. Each block's integer sum is scaled by its
weight E4M3 scale and its activation scale in FP32. The weights are unchanged;
only the activation precision differs from W4A16 (like llama.cpp's MMQ q8
activations). Without native FP4, `triton::nvfp4_auto_decode` times this
against the W4A16 GEMV (with split-K) at AOTI compile time and keeps the faster
one; `ET_CUDA_NVFP4_ACT=w4a8` forces it.
"""

import torch
import triton
import triton.language as tl

from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_param,
    InvalidLaunchParam,
)
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    _check_operands,
    _check_split_workspace,
    _e4m3_f16,
    _max_rows,
    _reduce_partials,
    SPLIT_K_CANDIDATES,
)
from executorch.backends.cuda.triton.kernels.nvfp4_tuning import (
    autotune_do_bench,
    launch_nvfp4_kernel,
    nvfp4_launch_bench,
    prune_nvfp4_stages,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    _compile_every_candidate_at_once,
    _QUEUING_ONLY,
)
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice


@triton.jit
def _act_q8(X, XQ, XS, K: tl.constexpr, BG: tl.constexpr):
    """Per-16 symmetric int8 activation blocks: scale = amax / 127."""
    row = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1) * BG + tl.arange(0, BG)
    k = g[:, None] * 16 + tl.arange(0, 16)[None, :]
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    inv = tl.where(amax > 0, 127.0 / amax, 0.0)
    q = libdevice.rint(x * inv[:, None])
    # NaN elements quantize to 0, as in the W4A4 activation quantizer.
    q = tl.where(q == q, q, 0.0)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(XQ + row * K + k, q, k < K)
    tl.store(XS + row * (K // 16) + g, amax / 127.0, g < K // 16)


@triton.jit
def _e2m1x8_to_s8(q):
    """Eight E2M1 codes (K order, low nibble first) -> two words of int8 2*value."""
    return tl.inline_asm_elementwise(
        """{
        .reg .b32 t0, t1, t2, t3, s, sel, lo, hi;
        mov.b32 t0, 0x03020100;
        mov.b32 t1, 0x0C080604;
        mov.b32 t2, 0xFDFEFF00;
        mov.b32 t3, 0xF4F8FAFC;
        and.b32 s, $2, 0x77777777;
        and.b32 sel, $2, 0x88888888;
        shr.b32 sel, sel, 1;
        or.b32 sel, sel, 0x32103210;
        prmt.b32 lo, t0, t1, s;
        prmt.b32 hi, t2, t3, s;
        prmt.b32 $0, lo, hi, sel;
        shr.b32 s, s, 16;
        shr.b32 sel, sel, 16;
        prmt.b32 lo, t0, t1, s;
        prmt.b32 hi, t2, t3, s;
        prmt.b32 $1, lo, hi, sel;
        }""",
        constraints="=r,=r,r",
        args=[q],
        dtype=(tl.int32, tl.int32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _dp4a_s8(a, b, c):
    return tl.inline_asm_elementwise(
        "dp4a.s32.s32 $0, $1, $2, $3;",
        constraints="=r,r,r,r",
        args=[a, b, c],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.autotune(
    configs=[
        triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
        for bn, bk, w in [
            (8, 256, 4),
            (8, 512, 4),
            (16, 256, 4),
            (16, 512, 4),
            (16, 1024, 8),
            (32, 256, 4),
            (32, 512, 8),
            (64, 256, 8),
        ]
        for st in (1, 2, 3)
    ],
    key=["M", "N", "K", "SPLIT", "ROW_PTS"],
    prune_configs_by={"early_config_prune": prune_nvfp4_stages},
    do_bench=autotune_do_bench,
)
@triton.jit
def _w4a8_gemv(
    XQ,
    XS,
    Q,
    S,
    PTS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ROW_PTS: tl.constexpr = False,
):
    # XQ: int8 activations viewed as int32 words [M, K/4]; Q: packed E2M1 viewed
    # as int32 words [N, K/8]; S: E4M3 block scales [N, K/16]; XS: fp32 [M, K/16].
    # Byte tensors read as 32-bit words (no dtype view in the AOTI graph).
    XQ = XQ.to(tl.pointer_type(tl.int32))
    Q = Q.to(tl.pointer_type(tl.int32))
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    m = tl.program_id(1)
    split = tl.program_id(2)
    chunk: tl.constexpr = tl.cdiv(K, BK * SPLIT) * BK
    W: tl.constexpr = BK // 8
    G: tl.constexpr = BK // 16
    w0 = split * (chunk // 8)
    acc = tl.zeros((BN, G), tl.float32)
    for off in range(tl.cdiv(chunk, BK)):
        wi = w0 + off * W + tl.arange(0, W)
        wv = wi < K // 8
        q = tl.load(
            Q + n[:, None] * (K // 8) + wi[None, :], (n[:, None] < N) & wv[None, :], 0
        )
        lo, hi = _e2m1x8_to_s8(q)
        xa = tl.broadcast_to(
            tl.load(XQ + m * (K // 4) + 2 * wi, wv, 0)[None, :], (BN, W)
        )
        xb = tl.broadcast_to(
            tl.load(XQ + m * (K // 4) + 2 * wi + 1, wv, 0)[None, :], (BN, W)
        )
        d = _dp4a_s8(lo, xa, tl.zeros((BN, W), tl.int32))
        d = _dp4a_s8(hi, xb, d)
        d16 = tl.sum(tl.reshape(d, (BN, G, 2)), 2)
        gi = w0 // 2 + off * G + tl.arange(0, G)
        gv = gi < K // 16
        sw = _e4m3_f16(
            tl.load(
                S + n[:, None] * (K // 16) + gi[None, :],
                (n[:, None] < N) & gv[None, :],
                0,
            )
        )
        sx = tl.load(XS + m * (K // 16) + gi, gv, 0)
        acc += d16.to(tl.float32) * sw * sx[None, :]
    # sw carries 2**-8 and the int8 weights are 2 * E2M1: rescale by 2**8 / 2.
    val = tl.sum(acc, 1) * 128.0
    if ROW_PTS:
        val = val * tl.load(PTS + n, n < N, 0)
    else:
        val = val * tl.load(PTS)
    tl.store(Y + (split.to(tl.int64) * M + m) * N + n, val, n < N)


def launch_w4a8_gemv(x, qdata, scale, pts, max_rows, *, split_k):
    m, k = x.shape
    n = qdata.shape[0]
    if split_k not in SPLIT_K_CANDIDATES or k % 16 or split_k > triton.cdiv(k, 256):
        raise InvalidLaunchParam(f"Illegal NVFP4 W4A8 GEMV split-K={split_k} for K={k}")
    _check_split_workspace(split_k, max_rows, n)
    xq = torch.empty((m, k), device=x.device, dtype=torch.int8)
    xs = torch.empty((m, k // 16), device=x.device, dtype=torch.float32)
    if not _QUEUING_ONLY.get():
        wrap_triton(_act_q8)[(m, triton.cdiv(k // 16, 64))](x, xq, xs, k, 64)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    partial = (
        torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
        if split_k > 1
        else out
    )
    launch_nvfp4_kernel(
        _w4a8_gemv,
        lambda meta: (triton.cdiv(n, meta["BN"]), m, split_k),
        (xq, xs, qdata, scale, pts, partial, m, n, k, split_k),
        6,
        ROW_PTS=pts.numel() != 1,
    )
    if split_k > 1 and not _QUEUING_ONLY.get():
        wrap_triton(_reduce_partials)[(triton.cdiv(m * n, 256),)](
            partial, pts, pts, out, m, n, split_k, False, 256, ROW_PTS=False
        )
    return out


_launch_w4a8 = autotune_launch_param(
    "split_k",
    SPLIT_K_CANDIDATES,
    candidate_prepass=_compile_every_candidate_at_once,
    do_bench=nvfp4_launch_bench,
)(launch_w4a8_gemv)


@triton_op("triton::nvfp4_w4a8_gemv", mutates_args={})
def nvfp4_w4a8_gemv(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    split_k: int = -1,
) -> torch.Tensor:
    """NVFP4 W4A8 GEMV for small M (M is grid axis 1, so at most 65535); the
    output keeps ``x.dtype``. NVIDIA only. Scale bytes must be finite E4M3
    (no 0x7F/0xFF), as ``CudaNVFP4Tensor.from_exportable`` checks; it is not
    checked here."""
    if torch.version.hip is not None:
        raise RuntimeError("NVFP4 W4A8 (dp4a PTX) requires an NVIDIA GPU")
    _check_operands(x, qdata, scale, per_tensor_scale)
    max_rows = _max_rows(x.shape[0])
    if max_rows is None or max_rows > 65535:
        raise ValueError("NVFP4 W4A8 GEMV supports at most 65535 rows")
    args = (x, qdata, scale, per_tensor_scale, max_rows)
    if split_k < 0:
        return _launch_w4a8(*args)
    return launch_w4a8_gemv(*args, split_k=split_k)
