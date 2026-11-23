# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Winograd F(4x4, 3x3) for strict-FP32 3x3 stride-1 convolutions.

A schedule of the implicit-GEMM template (``winograd_f4``): it computes each
4x4 output tile from a 6x6 input tile with 36 elementwise products instead of
144 multiply-adds per channel pair. Four kernels run in order, all in FP32:

- the weight transform ``U = G g G^T`` into ``[36, C_out, C_in]``;
- the input transform ``V = B^T d B`` of every 6x6 tile into ``[36, C_in, T]``;
- 36 GEMMs ``M[z] = U[z] V[z]`` with ``tl.dot`` in IEEE FP32 (FMA, no TF32);
- the output transform ``Y = A^T M A``, plus the bias, stored through the output
  strides.

The transform points are 0, +-1/sqrt(2), +-sqrt(2) and infinity, which keep the
transform coefficients small (balanced F(4x4, 3x3)) and the error close to
that of a direct convolution (see the tests).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from executorch.backends.cuda.triton.kernels.conv.core.common import (
    ConvCall,
    max_offset,
)
from torch.library import wrap_triton

_TILE = 4
_INT32_MAX = 2**31 - 1


def reason(call: ConvCall) -> Optional[str]:
    """Why Winograd F(4x4, 3x3) cannot run ``call``, or None."""
    x, w, p = call.input, call.weight, call.params
    if (
        p.rank != 2
        or p.transposed
        or p.groups != 1
        or tuple(w.shape[2:]) != (3, 3)
        or p.stride != (1, 1)
        or p.dilation != (1, 1)
        or p.padding_before != (1, 1)
        or p.padding_after != (1, 1)
    ):
        return "winograd_f4 needs a dense 3x3 stride-1 convolution with padding 1"
    if x.dtype != torch.float32:
        return f"winograd_f4 needs FP32 operands, not {x.dtype}"
    tiles = x.shape[0] * triton.cdiv(x.shape[2], _TILE) * triton.cdiv(x.shape[3], _TILE)
    scratch = 36 * max(w.shape[0], x.shape[1]) * max(tiles, w.shape[0])
    if max(max_offset(x), max_offset(w), scratch) > _INT32_MAX:
        return "winograd_f4 uses 32-bit indices"
    return None


@triton.jit
def _g(x0, x1, x2):
    root2: tl.constexpr = 1.4142135623730951
    return (
        x0,
        -(2.0 * x0 + root2 * x1 + x2) * (1.0 / 3.0),
        (-2.0 * x0 + root2 * x1 - x2) * (1.0 / 3.0),
        (x0 + root2 * x1 + 2.0 * x2) * (1.0 / 6.0),
        (x0 - root2 * x1 + 2.0 * x2) * (1.0 / 6.0),
        x2,
    )


@triton.jit
def _bt(x0, x1, x2, x3, x4, x5):
    root2: tl.constexpr = 1.4142135623730951
    p = x4 - 2.0 * x2
    q = root2 * (0.5 * x3 - x1)
    s = x4 - 0.5 * x2
    t = root2 * (x3 - 0.5 * x1)
    return x0 - 2.5 * x2 + x4, p + q, p - q, s + t, s - t, x1 - 2.5 * x3 + x5


@triton.jit
def _at(x0, x1, x2, x3, x4, x5):
    root2: tl.constexpr = 1.4142135623730951
    s12 = x1 + x2
    d12 = x1 - x2
    s34 = x3 + x4
    d34 = x3 - x4
    return (
        x0 + s12 + s34,
        (0.5 * root2) * (d12 + 2.0 * d34),
        0.5 * (s12 + 4.0 * s34),
        (0.25 * root2) * (d12 + 8.0 * d34) + x5,
    )


@triton.jit
def _store6(ptr, step, mask, v0, v1, v2, v3, v4, v5):
    tl.store(ptr, v0, mask=mask)
    tl.store(ptr + step, v1, mask=mask)
    tl.store(ptr + 2 * step, v2, mask=mask)
    tl.store(ptr + 3 * step, v3, mask=mask)
    tl.store(ptr + 4 * step, v4, mask=mask)
    tl.store(ptr + 5 * step, v5, mask=mask)


@triton.jit
def _load6(ptr, step, mask):
    return (
        tl.load(ptr, mask=mask, other=0.0),
        tl.load(ptr + step, mask=mask, other=0.0),
        tl.load(ptr + 2 * step, mask=mask, other=0.0),
        tl.load(ptr + 3 * step, mask=mask, other=0.0),
        tl.load(ptr + 4 * step, mask=mask, other=0.0),
        tl.load(ptr + 5 * step, mask=mask, other=0.0),
    )


@triton.jit
def _weight_transform(w_ptr, u_ptr, C, CO, SW_O, SW_C, SW_H, SW_W, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < CO * C
    g = w_ptr + p // C * SW_O + p % C * SW_C
    a0, a1, a2, a3, a4, a5 = _g(
        tl.load(g, mask=mask),
        tl.load(g + SW_W, mask=mask),
        tl.load(g + 2 * SW_W, mask=mask),
    )
    g += SW_H
    b0, b1, b2, b3, b4, b5 = _g(
        tl.load(g, mask=mask),
        tl.load(g + SW_W, mask=mask),
        tl.load(g + 2 * SW_W, mask=mask),
    )
    g += SW_H
    c0, c1, c2, c3, c4, c5 = _g(
        tl.load(g, mask=mask),
        tl.load(g + SW_W, mask=mask),
        tl.load(g + 2 * SW_W, mask=mask),
    )
    # U[6 i + j] holds row i, column j of G g G^T.
    plane = CO * C
    u = u_ptr + p
    _store6(u, 6 * plane, mask, *_g(a0, b0, c0))
    _store6(u + plane, 6 * plane, mask, *_g(a1, b1, c1))
    _store6(u + 2 * plane, 6 * plane, mask, *_g(a2, b2, c2))
    _store6(u + 3 * plane, 6 * plane, mask, *_g(a3, b3, c3))
    _store6(u + 4 * plane, 6 * plane, mask, *_g(a4, b4, c4))
    _store6(u + 5 * plane, 6 * plane, mask, *_g(a5, b5, c5))


@triton.jit
def _input_row(x_ptr, y, x0, H, W, SX_H, SX_W, mask):
    """B^T applied to one padded 6-wide row of an input tile."""
    ok = mask & (y >= 0) & (y < H)
    row = x_ptr + y * SX_H + x0 * SX_W
    return _bt(
        tl.load(row, mask=ok & (x0 >= 0) & (x0 < W), other=0.0),
        tl.load(row + SX_W, mask=ok & (x0 + 1 >= 0) & (x0 + 1 < W), other=0.0),
        tl.load(row + 2 * SX_W, mask=ok & (x0 + 2 < W), other=0.0),
        tl.load(row + 3 * SX_W, mask=ok & (x0 + 3 < W), other=0.0),
        tl.load(row + 4 * SX_W, mask=ok & (x0 + 4 < W), other=0.0),
        tl.load(row + 5 * SX_W, mask=ok & (x0 + 5 < W), other=0.0),
    )


@triton.jit
def _input_transform(
    x_ptr, v_ptr, C, H, W, TH, TW, T, SX_N, SX_C, SX_H, SX_W, BLOCK: tl.constexpr
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < C * T
    t = p % T
    tile = t % (TH * TW)
    y0 = tile // TW * 4 - 1
    x0 = tile % TW * 4 - 1
    x = x_ptr + t // (TH * TW) * SX_N + p // T * SX_C
    h00, h01, h02, h03, h04, h05 = _input_row(x, y0, x0, H, W, SX_H, SX_W, mask)
    h10, h11, h12, h13, h14, h15 = _input_row(x, y0 + 1, x0, H, W, SX_H, SX_W, mask)
    h20, h21, h22, h23, h24, h25 = _input_row(x, y0 + 2, x0, H, W, SX_H, SX_W, mask)
    h30, h31, h32, h33, h34, h35 = _input_row(x, y0 + 3, x0, H, W, SX_H, SX_W, mask)
    h40, h41, h42, h43, h44, h45 = _input_row(x, y0 + 4, x0, H, W, SX_H, SX_W, mask)
    h50, h51, h52, h53, h54, h55 = _input_row(x, y0 + 5, x0, H, W, SX_H, SX_W, mask)
    plane = C * T
    v = v_ptr + p
    _store6(v, 6 * plane, mask, *_bt(h00, h10, h20, h30, h40, h50))
    _store6(v + plane, 6 * plane, mask, *_bt(h01, h11, h21, h31, h41, h51))
    _store6(v + 2 * plane, 6 * plane, mask, *_bt(h02, h12, h22, h32, h42, h52))
    _store6(v + 3 * plane, 6 * plane, mask, *_bt(h03, h13, h23, h33, h43, h53))
    _store6(v + 4 * plane, 6 * plane, mask, *_bt(h04, h14, h24, h34, h44, h54))
    _store6(v + 5 * plane, 6 * plane, mask, *_bt(h05, h15, h25, h35, h45, h55))


CONFIGS = [
    triton.Config(
        {"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": 16, "GROUP_M": 8},
        num_warps=warps,
        num_stages=stages,
    )
    for m, n, warps, stages in (
        (64, 64, 4, 3),
        (128, 64, 8, 3),
        (64, 128, 8, 3),
        (64, 32, 4, 3),
        (128, 32, 4, 2),
    )
]


@triton.autotune(configs=CONFIGS, key=["C", "CO", "T"])
@triton.jit
def _gemm(
    v_ptr,
    u_ptr,
    m_ptr,
    C,
    CO,
    T,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """M[z] (C_out x T) = U[z] (C_out x C_in) V[z] (C_in x T) for z in 0..35."""
    grid_m = tl.cdiv(T, BLOCK_M)
    grid_n = tl.cdiv(CO, BLOCK_N)
    pid = tl.program_id(0)
    z = tl.program_id(1)
    first_m = pid // (GROUP_M * grid_n) * GROUP_M
    rows = tl.minimum(grid_m - first_m, GROUP_M)
    pid_m = first_m + pid % (GROUP_M * grid_n) % rows
    pid_n = pid % (GROUP_M * grid_n) // rows
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < T
    n_mask = offs_n < CO
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, C, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < C
        v = tl.load(
            v_ptr + (z * C + k)[None, :] * T + offs_m[:, None],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        u = tl.load(
            u_ptr + (z * CO + offs_n)[None, :] * C + k[:, None],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(v, u, acc, input_precision="ieee")
    tl.store(
        m_ptr + (z * CO + offs_n)[None, :] * T + offs_m[:, None],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _output_row(out, oy, ox, H, W, SO_H, SO_W, mask, bias, y0, y1, y2, y3):
    ok = mask & (oy < H)
    row = out + oy * SO_H + ox * SO_W
    tl.store(row, (y0 + bias).to(out.dtype.element_ty), mask=ok & (ox < W))
    tl.store(row + SO_W, (y1 + bias).to(out.dtype.element_ty), mask=ok & (ox + 1 < W))
    tl.store(
        row + 2 * SO_W, (y2 + bias).to(out.dtype.element_ty), mask=ok & (ox + 2 < W)
    )
    tl.store(
        row + 3 * SO_W, (y3 + bias).to(out.dtype.element_ty), mask=ok & (ox + 3 < W)
    )


@triton.jit
def _output_transform(
    m_ptr,
    b_ptr,
    out_ptr,
    CO,
    H,
    W,
    TH,
    TW,
    T,
    SO_N,
    SO_C,
    SO_H,
    SO_W,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < CO * T
    co = p // T
    t = p % T
    plane = CO * T
    m = m_ptr + p
    h00, h10, h20, h30 = _at(*_load6(m, 6 * plane, mask))
    h01, h11, h21, h31 = _at(*_load6(m + plane, 6 * plane, mask))
    h02, h12, h22, h32 = _at(*_load6(m + 2 * plane, 6 * plane, mask))
    h03, h13, h23, h33 = _at(*_load6(m + 3 * plane, 6 * plane, mask))
    h04, h14, h24, h34 = _at(*_load6(m + 4 * plane, 6 * plane, mask))
    h05, h15, h25, h35 = _at(*_load6(m + 5 * plane, 6 * plane, mask))
    if HAS_BIAS:
        bias = tl.load(b_ptr + co, mask=mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros((BLOCK,), dtype=tl.float32)
    tile = t % (TH * TW)
    oy = tile // TW * 4
    ox = tile % TW * 4
    out = out_ptr + t // (TH * TW) * SO_N + co * SO_C
    _output_row(
        out, oy, ox, H, W, SO_H, SO_W, mask, bias, *_at(h00, h01, h02, h03, h04, h05)
    )
    _output_row(
        out,
        oy + 1,
        ox,
        H,
        W,
        SO_H,
        SO_W,
        mask,
        bias,
        *_at(h10, h11, h12, h13, h14, h15),
    )
    _output_row(
        out,
        oy + 2,
        ox,
        H,
        W,
        SO_H,
        SO_W,
        mask,
        bias,
        *_at(h20, h21, h22, h23, h24, h25),
    )
    _output_row(
        out,
        oy + 3,
        ox,
        H,
        W,
        SO_H,
        SO_W,
        mask,
        bias,
        *_at(h30, h31, h32, h33, h34, h35),
    )


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    config: Optional[triton.Config] = None,
) -> None:
    """Computes the convolution into ``out``; the transforms and products use
    FP32 scratch allocated here. ``config`` replaces the GEMM's autotuned
    config."""
    n, c, h, w = x.shape
    co = weight.shape[0]
    th, tw = triton.cdiv(h, _TILE), triton.cdiv(w, _TILE)
    tiles = n * th * tw
    u = x.new_empty((36, co, c))
    v = x.new_empty((36, c, tiles))
    m = x.new_empty((36, co, tiles))
    block = 64
    wrap_triton(_weight_transform)[(triton.cdiv(co * c, block),)](
        weight, u, c, co, *weight.stride(), BLOCK=block, num_warps=2
    )
    wrap_triton(_input_transform)[(triton.cdiv(c * tiles, block),)](
        x, v, c, h, w, th, tw, tiles, *x.stride(), BLOCK=block, num_warps=2
    )
    gemm, options = _gemm, {}
    if config is not None:
        gemm = _gemm.fn
        options = {
            **config.kwargs,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
    wrap_triton(gemm)[
        lambda meta: (
            triton.cdiv(tiles, meta["BLOCK_M"]) * triton.cdiv(co, meta["BLOCK_N"]),
            36,
        )
    ](v, u, m, c, co, tiles, **options)
    wrap_triton(_output_transform)[(triton.cdiv(co * tiles, block),)](
        m,
        x if bias is None else bias,
        out,
        co,
        h,
        w,
        th,
        tw,
        tiles,
        *out.stride(),
        HAS_BIAS=bias is not None,
        BLOCK=block,
        num_warps=2,
    )
