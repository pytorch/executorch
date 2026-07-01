#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""Shared GGUF **Q5_K** primitives for the MLX backend.

This module holds the pieces common to every Q5_K kernel (linear matmul/matvec
and the embedding gather), so format-specific op modules import from here:

* ``QK_K`` / ``K_SCALE_SIZE`` / ``Q5K_BLOCK_BYTES`` and the per-super-block byte
  layout constants.
* ``_Q5K_HEADER`` -- the Metal header (the ``block_q5_K`` struct plus the
  per-element and vectorized dequant helpers) shared by all Q5_K Metal kernels.

Q5_K layout (per 256-element super-block, 176 bytes, see llama.cpp
``block_q5_K`` in ``ggml-common.h``)::

    half     d                    # super-block scale for the quantized scales
    half     dmin                 # super-block scale for the quantized mins
    uint8    scales[12]           # 6-bit packed sub-block scales + mins (as Q4_K)
    uint8    qh[QK_K/8 = 32]      # quants, high bit (the 5th bit)
    uint8    qs[QK_K/2 = 128]     # quants, low 4 bits

Q5_K combines Q4_K's affine super-block (``d``/``dmin`` with 6-bit packed
scales/mins unpacked via ``get_scale_min_k4``) with a Q6_K-style high-bit array:
each weight is a 5-bit code ``q`` (0..31) whose low 4 bits come from ``qs`` and
whose 5th bit comes from ``qh``. The dequantized value for code ``q`` in
sub-block ``s`` is ``d * scale[s] * q - dmin * min[s]`` (affine, same shape as
Q4_K but with ``q`` in 0..31).

Attribution
-----------
The Q5_K block layout and the Metal dequant helpers in ``_Q5K_HEADER`` follow
llama.cpp (``ggml-common.h`` / ``ggml-metal.metal``: ``block_q5_K``,
``dequantize_q5_K``, ``dequantize_row_q5_K``, ``get_scale_min_k4``), which is
MIT-licensed (Copyright (c) 2023-2024 The ggml authors).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Q5_K constants
# ---------------------------------------------------------------------------

QK_K = 256
K_SCALE_SIZE = 12
_Q5K_D_BYTES = 2
_Q5K_DMIN_BYTES = 2
_Q5K_SCALES_BYTES = K_SCALE_SIZE
_Q5K_QH_BYTES = QK_K // 8  # 32
_Q5K_QS_BYTES = QK_K // 2  # 128
Q5K_BLOCK_BYTES = (
    _Q5K_D_BYTES + _Q5K_DMIN_BYTES + _Q5K_SCALES_BYTES + _Q5K_QH_BYTES + _Q5K_QS_BYTES
)  # 176


# ---------------------------------------------------------------------------
# Shared Metal header
# ---------------------------------------------------------------------------

# Ported from llama.cpp ggml-common.h (block_q5_K, get_scale_min_k4) and
# ggml-metal.metal (dequantize_q5_K). Struct field order matches GGUF bytes:
# d(0:2), dmin(2:4), scales(4:16), qh(16:48), qs(48:176).
_Q5K_HEADER = """
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    half    d;                  // super-block scale for quantized scales
    half    dmin;               // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // 6-bit packed scales + mins (same as Q4_K)
    uint8_t qh[QK_K/8];         // quants, high bit (the 5th bit)
    uint8_t qs[QK_K/2];         // quants, low 4 bits
} block_q5_K;                   // 176 bytes

// Unpack 6-bit scale and min for sub-block index j (0..7).
// Ported from llama.cpp get_scale_min_k4 (ggml-quants.c). Identical to Q4_K.
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

// Metal variant used by dequantize_q5_K_16 (matches ggml-metal.metal). Same as Q4_K.
inline uchar2 get_scale_min_k4_just2(int j, int k, device const uint8_t * q) {
    return j < 4
        ? uchar2{uint8_t(q[j + 0 + k] & 63), uint8_t(q[j + 4 + k] & 63)}
        : uchar2{
              uint8_t((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] >> 6) << 4)),
              uint8_t((q[j + 4 + k] >> 4) | ((q[j + 0 + k] >> 6) << 4))};
}

// Dequantize a single element at within-block position p (0..255).
// Mirrors dequantize_row_q5_K (ggml-quants.c): 64-element chunks, 32 lows then
// 32 highs per chunk; the 5th bit comes from qh at bit (2*chunk + half).
inline float dequant_q5k_elem(device const block_q5_K * blk, int p) {
    const int c    = p >> 6;        // 0..3 (64-element chunk)
    const int within = p & 63;      // 0..63 within chunk
    const int l    = within & 31;   // 0..31 (also the qh byte index)
    const int hi   = within >> 5;   // 0 low nibble, 1 high nibble
    const int sub  = 2 * c + hi;    // sub-block 0..7 (also the qh bit index)

    const uint8_t byte = blk->qs[c * 32 + l];
    const int nib = (hi == 0) ? (byte & 0xF) : (byte >> 4);
    const int hibit = (blk->qh[l] >> sub) & 1;
    const int q = nib + (hibit << 4);  // 0..31

    uint8_t sc, mn;
    get_scale_min_k4(sub, blk->scales, sc, mn);

    const float d  = (float) blk->d;
    const float dm = (float) blk->dmin;
    return d * (float) sc * (float) q - dm * (float) mn;
}

// Vectorized Q5_K dequantize: decodes 16 values into half4x4.
// Ported from llama.cpp dequantize_q5_K (ggml-metal.metal): Q4_K's vectorized
// decode plus the Q6_K-style 5th bit. mat-mat uses NL=16 (QK_K/16); for sub-block
// `il` the 16 outputs map to block positions [il*16 : il*16+16].
inline void dequantize_q5_K_16(device const block_q5_K * xb, short il,
                               thread half4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il / 4) * 2;
    q  = q + (il / 4) * 32 + 16 * (il & 1);
    qh = qh + 16 * (il & 1);
    const uint8_t ul = 1 << (il / 2);  // 5th-bit mask (uses original il)
    il = il & 3;

    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d  = il < 2 ? (float) xb->d : (float) xb->d / 16.f;
    const float dm = (float) xb->dmin;
    const float dl = d * (float) sc[0];
    const float ml = dm * (float) sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    // Low nibble (mask 0x0F): the 5th bit adds 16. High nibble (mask 0xF0) is the
    // value pre-shifted by 4 with d/16, so the 5th bit adds 16*16 = 256.
    const float qh_val = il < 2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        const float v = (float)(q[i] & mask) + ((qh[i] & ul) ? qh_val : 0.f);
        reg[i / 4][i % 4] = (half)(dl * v - ml);
    }
}
"""
