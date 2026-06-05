#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""Shared GGUF **Q6_K** primitives for the MLX custom ops.

This module holds the pieces common to every Q6_K kernel (linear matmul/matvec
and the embedding gather), so format-specific op modules import from here rather
than from each other:

* ``QK_K`` / ``Q6K_BLOCK_BYTES`` and the per-super-block byte layout constants.
* ``dequantize_q6_k`` -- the pure-torch dequant oracle (eager fallback + tests).
* ``_Q6K_HEADER`` -- the Metal header (the ``block_q6_K`` struct plus the
  per-element and vectorized dequant helpers) shared by all Q6_K Metal kernels.

Adding another GGUF format (e.g. Q4_K) should mirror this module (``q4k.py``)
and the op handlers in :mod:`.linear` / :mod:`.embedding` dispatch on ``format``.

Q6_K layout (per 256-element super-block, 210 bytes, see llama.cpp
``block_q6_K`` in ``ggml-common.h``)::

    uint8  ql[128]    # quants, lower 4 bits
    uint8  qh[64]     # quants, upper 2 bits
    int8   scales[16] # per-16-element sub-block scales (8-bit)
    half   d          # super-block scale

The dequantized value for a 6-bit code ``q`` (0..63) in sub-block ``s`` is
``d * scales[s] * (q - 32)``.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Q6_K constants
# ---------------------------------------------------------------------------

QK_K = 256
# Per-super-block byte counts.
_Q6K_QL_BYTES = QK_K // 2  # 128
_Q6K_QH_BYTES = QK_K // 4  # 64
_Q6K_SCALES = QK_K // 16  # 16
_Q6K_D_BYTES = 2  # one fp16
Q6K_BLOCK_BYTES = _Q6K_QL_BYTES + _Q6K_QH_BYTES + _Q6K_SCALES + _Q6K_D_BYTES  # 210


# ---------------------------------------------------------------------------
# Pure-torch dequant reference
# ---------------------------------------------------------------------------


def dequantize_q6_k(weight: Tensor, K: int) -> Tensor:
    """Dequantize a GGUF Q6_K blob to float32.

    Args:
        weight: ``(N, n_blocks * 210)`` uint8, GGUF ``block_q6_K`` layout.
        K: number of logical input features (``n_blocks * 256``).

    Returns:
        ``(N, K)`` float32 dequantized weight.
    """
    if weight.dtype != torch.uint8:
        raise ValueError(f"gguf_linear: weight must be uint8; got {weight.dtype}")
    N = weight.shape[0]
    nb = K // QK_K
    if weight.shape[-1] != nb * Q6K_BLOCK_BYTES:
        raise ValueError(
            f"gguf_linear: weight row bytes {weight.shape[-1]} != "
            f"{nb} blocks * {Q6K_BLOCK_BYTES}"
        )

    blocks = weight.view(N, nb, Q6K_BLOCK_BYTES)
    ql = blocks[..., 0:_Q6K_QL_BYTES].to(torch.int32)
    qh = blocks[..., _Q6K_QL_BYTES : _Q6K_QL_BYTES + _Q6K_QH_BYTES].to(torch.int32)
    sc_off = _Q6K_QL_BYTES + _Q6K_QH_BYTES
    scales = (
        blocks[..., sc_off : sc_off + _Q6K_SCALES]
        .contiguous()
        .view(torch.int8)
        .to(torch.float32)
    )
    d = (
        blocks[..., sc_off + _Q6K_SCALES : sc_off + _Q6K_SCALES + _Q6K_D_BYTES]
        .contiguous()
        .view(torch.float16)
        .to(torch.float32)
    )  # (N, nb, 1)

    y = torch.empty(N, nb, QK_K, dtype=torch.float32, device=weight.device)
    # is = l // 16 over l in 0..31 -> selects which of the 8 half-scales.
    is_idx = (torch.arange(32, device=weight.device) // 16).long()  # (32,)

    for h in range(2):  # two 128-element halves
        ql_h = ql[..., h * 64 : h * 64 + 64]  # (N, nb, 64)
        qh_h = qh[..., h * 32 : h * 32 + 32]  # (N, nb, 32)
        sc_h = scales[..., h * 8 : h * 8 + 8]  # (N, nb, 8)

        ql_lo = ql_h[..., 0:32]
        ql_hi = ql_h[..., 32:64]

        q1 = (ql_lo & 0xF) | ((qh_h & 0x3) << 4)
        q2 = (ql_hi & 0xF) | (((qh_h >> 2) & 0x3) << 4)
        q3 = (ql_lo >> 4) | (((qh_h >> 4) & 0x3) << 4)
        q4 = (ql_hi >> 4) | (((qh_h >> 6) & 0x3) << 4)

        sc0 = sc_h[..., is_idx + 0]
        sc2 = sc_h[..., is_idx + 2]
        sc4 = sc_h[..., is_idx + 4]
        sc6 = sc_h[..., is_idx + 6]

        base = h * 128
        y[..., base + 0 : base + 32] = d * sc0 * (q1 - 32).to(torch.float32)
        y[..., base + 32 : base + 64] = d * sc2 * (q2 - 32).to(torch.float32)
        y[..., base + 64 : base + 96] = d * sc4 * (q3 - 32).to(torch.float32)
        y[..., base + 96 : base + 128] = d * sc6 * (q4 - 32).to(torch.float32)

    return y.reshape(N, K)


# ---------------------------------------------------------------------------
# Shared Metal header
# ---------------------------------------------------------------------------

# The GGUF block_q6_K struct (matches llama.cpp ggml-common.h; sizeof == 210, no
# padding since max align is 2) plus dequant helpers for both per-element
# (embedding) and vectorized (matmul) use.
_Q6K_HEADER = """
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define QK_K 256

typedef struct {
    uint8_t ql[QK_K/2];      // lower 4 bits
    uint8_t qh[QK_K/4];      // upper 2 bits
    int8_t  scales[QK_K/16]; // per-16-element sub-block scales
    half    d;               // super-block scale
} block_q6_K;

// Dequantize a single element at within-block position p (0..255) of a
// block_q6_K. Used by the embedding kernel.
inline float dequant_q6k_elem(device const block_q6_K * blk, int p) {
    const int h  = p >> 7;     // which 128-element half (0/1)
    const int pp = p & 127;    // position within half (0..127)
    const int g  = pp >> 5;    // group: 0=q1, 1=q2, 2=q3, 3=q4
    const int l  = pp & 31;    // 0..31
    device const uint8_t * ql = blk->ql + h * 64;
    device const uint8_t * qh = blk->qh + h * 32;
    device const int8_t  * sc = blk->scales + h * 8;
    const int is = l >> 4;     // 0/1
    const uint8_t qhb = qh[l];
    int q;
    if (g == 0)      { q = (ql[l]      & 0xF) | ((qhb & 0x03) << 4); }
    else if (g == 1) { q = (ql[l + 32] & 0xF) | ((qhb & 0x0C) << 2); }
    else if (g == 2) { q = (ql[l]      >> 4)  | ((qhb & 0x30) << 0); }
    else             { q = (ql[l + 32] >> 4)  | ((qhb & 0xC0) >> 2); }
    const float scale = (float) sc[is + 2 * g];
    return (float) blk->d * scale * (float)(q - 32);
}

// Vectorized Q6_K dequantize: decodes 16 values per call into a 4x4 half
// register. Ported from llama.cpp dequantize_q6_K. `il` ranges 0..15 and
// selects which 16-element slice of the 256-element block to decode.
inline void dequantize_q6_K_16(device const block_q6_K * xb, short il,
                               thread half4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
    const float coeff = d_all * sc;
    const float ml = coeff * 32.f;
    const float dl0 = coeff;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = (half)(dl0 *  ((half)(q & 0xFF))       - ml);
        reg[i][1] = (half)(dl1 * ((float)(q & 0xFF00))     - ml);
        reg[i][2] = (half)(dl2 * ((float)(q & 0xFF0000))   - ml);
        reg[i][3] = (half)(dl3 * ((float)(q & 0xFF000000)) - ml);
    }
}
"""
