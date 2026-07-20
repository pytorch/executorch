#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""Shared GGUF **Q6_K** primitives for the MLX backend.

This module holds the pieces common to every Q6_K kernel (linear matmul/matvec
and the embedding gather), so format-specific op modules import from here rather
than from each other:

* ``QK_K`` / ``Q6K_BLOCK_BYTES`` and the per-super-block byte layout constants.
* ``_Q6K_HEADER`` -- the Metal header (the ``block_q6_K`` struct plus the
  per-element and vectorized dequant helpers) shared by all Q6_K Metal kernels.

Q6_K layout

Q6_K layout (per 256-element super-block, 210 bytes, see llama.cpp
``block_q6_K`` in ``ggml-common.h``)::

    uint8  ql[128]    # quants, lower 4 bits
    uint8  qh[64]     # quants, upper 2 bits
    int8   scales[16] # per-16-element sub-block scales (8-bit)
    half   d          # super-block scale

The dequantized value for a 6-bit code ``q`` (0..63) in sub-block ``s`` is
``d * scales[s] * (q - 32)``.

Attribution
-----------
The Q6_K block layout and the Metal dequant helpers in ``_Q6K_HEADER`` follow
llama.cpp
(``ggml-common.h`` / ``ggml-metal.metal``: ``block_q6_K``, ``dequantize_q6_K``),
which is MIT-licensed (Copyright (c) 2023-2024 The ggml authors).
"""

from __future__ import annotations


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
# Shared Metal header
# ---------------------------------------------------------------------------

# The GGUF block_q6_K struct (matches llama.cpp ggml-common.h; sizeof == 210, no
# padding since max align is 2) plus dequant helpers for both per-element
# (embedding) and vectorized (matmul) use.
_Q6K_HEADER = """
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef matrix<bfloat, 2, 4> bfloat2x4;
template<typename T> struct vec2x4;
template<> struct vec2x4<float>   { using type = float2x4;  };
template<> struct vec2x4<half>    { using type = half2x4;   };
template<> struct vec2x4<bfloat>  { using type = bfloat2x4; };

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
