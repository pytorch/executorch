#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""Shared GGUF **Q4_K** primitives for the MLX backend.

This module holds the pieces common to every Q4_K kernel (linear matmul/matvec
and the embedding gather):

* ``QK_K`` / ``Q4K_BLOCK_BYTES`` and the per-super-block byte layout constants.
* ``_Q4K_HEADER`` -- the Metal header (the ``block_q4_K`` struct plus the
  per-element and vectorized dequant helpers) shared by all Q4_K Metal kernels.

Q4_K layout (per 256-element super-block, 144 bytes, see llama.cpp
``block_q4_K`` in ``ggml-common.h``)::

    half     d       # super-block scale for quantized scales
    half     dmin    # super-block scale for quantized mins
    uint8  scales[12]  # 6-bit packed scales + mins
    uint8  qs[128]     # 4-bit quants

The dequantized value for a 4-bit code ``q`` in sub-block ``s`` is
``d * scale[s] * q - dmin * min[s]`` (affine).

Attribution
-----------
The Q4_K block layout and the Metal dequant helpers in ``_Q4K_HEADER`` follow
llama.cpp
(``ggml-common.h`` / ``ggml-metal.metal``: ``block_q4_K``, ``dequantize_q4_K``,
``get_scale_min_k4``), which is MIT-licensed (Copyright (c) 2023-2024 The ggml
authors).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Q4_K constants
# ---------------------------------------------------------------------------

QK_K = 256
K_SCALE_SIZE = 12
_Q4K_D_BYTES = 2
_Q4K_DMIN_BYTES = 2
_Q4K_SCALES_BYTES = K_SCALE_SIZE
_Q4K_QS_BYTES = QK_K // 2  # 128
Q4K_BLOCK_BYTES = (
    _Q4K_D_BYTES + _Q4K_DMIN_BYTES + _Q4K_SCALES_BYTES + _Q4K_QS_BYTES
)  # 144


# ---------------------------------------------------------------------------
# Shared Metal header
# ---------------------------------------------------------------------------

# Ported from llama.cpp ggml-common.h (block_q4_K, get_scale_min_k4) and
# ggml-metal.metal (dequantize_q4_K). Struct field order matches GGUF bytes:
# d(0:2), dmin(2:4), scales(4:16), qs(16:144).
_Q4K_HEADER = """
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

// Unpack 6-bit scale and min for sub-block index j (0..7).
// Ported from llama.cpp get_scale_min_k4 (ggml-quants.c).
inline void get_scale_min_k4(int j, device const uint8_t * q,
                             thread uint8_t & sc, thread uint8_t & m) {
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Metal variant used by dequantize_q4_K_16 (matches ggml-metal.metal).
inline uchar2 get_scale_min_k4_just2(int j, int k, device const uint8_t * q) {
    return j < 4
        ? uchar2{uint8_t(q[j + 0 + k] & 63), uint8_t(q[j + 4 + k] & 63)}
        : uchar2{
              uint8_t((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] >> 6) << 4)),
              uint8_t((q[j + 4 + k] >> 4) | ((q[j + 0 + k] >> 6) << 4))};
}

// Dequantize a single element at within-block position p (0..255).
// Mirrors dequantize_row_q4_K (ggml-quants.c): 64-element chunks, 32 lows
// then 32 highs per chunk, each half using its own scale/min pair.
inline float dequant_q4k_elem(device const block_q4_K * blk, int p) {
    const int chunk = p >> 6;          // 0..3 (64-element groups)
    const int sub   = p & 63;          // 0..63 within chunk
    const int q_idx = (chunk << 5) + (sub & 31);
    device const uint8_t * q = blk->qs + q_idx;

    uint8_t sc, mn;
    get_scale_min_k4((chunk << 1) + (sub >= 32 ? 1 : 0), blk->scales, sc, mn);

    const float d  = (float) blk->d;
    const float dm = (float) blk->dmin;
    const float dl = d * (float) sc;
    const float ml = dm * (float) mn;

    const uint8_t nib = (sub < 32) ? (q[0] & 0xF) : (q[0] >> 4);
    return dl * (float) nib - ml;
}

// Vectorized Q4_K dequantize: decodes 16 values into half4x4.
// Ported from llama.cpp dequantize_q4_K (ggml-metal.metal).
// il = sub-block passed by the mat-mat kernel.
// mat-mat uses NL=16 (QK_K/16), same as the Q6_K kernel_mul_mm port.
inline void dequantize_q4_K_16(device const block_q4_K * xb, short il,
                               thread half4x4 & reg) {
    device const uint8_t * q = xb->qs;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;

    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d  = il < 2 ? (float) xb->d : (float) xb->d / 16.f;
    const float dm = (float) xb->dmin;
    const float dl = d * (float) sc[0];
    const float ml = dm * (float) sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)(dl * (float)(q[i] & mask) - ml);
    }
}
"""
