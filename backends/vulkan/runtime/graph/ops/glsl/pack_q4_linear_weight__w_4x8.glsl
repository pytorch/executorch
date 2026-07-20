/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define STORAGE ${STORAGE}

layout(std430) buffer;

// Output: W_4X8 block-packed nibble weight, written as 4K x 8N blocks. Each invocation
// produces one full 4K x 8N block at logical position (k4, n8) — equivalent to
// 2 consecutive ivec2 tiles at the SAME k4: n4 = 2*n8 (lower) and n4 = 2*n8+1
// (upper). The 4 ints making up the block are:
//     [0] = packed_x for (k4, n4 = 2*n8)        (rows N0, N1 of n4_a)
//     [1] = packed_y for (k4, n4 = 2*n8)        (rows N2, N3 of n4_a)
//     [2] = packed_x for (k4, n4 = 2*n8+1)      (rows N0, N1 of n4_b)
//     [3] = packed_y for (k4, n4 = 2*n8+1)      (rows N2, N3 of n4_b)
//
// Buffer (nc) form: stored as a flat ivec4 buffer; one block per ivec4 at
// index `k4 * N8 + n8`, where N8 = N4_padded/2 and N4_padded is the next-even
// N4. This is byte-identical to writing 4 consecutive ints at scalar index
// `4*(k4*N8 + n8)` (the legacy 2-tile layout).
//
// Buffer (kc dense) form: stored as a flat ivec4 buffer; one block per ivec4
// at index `n8 * K4 + k4`. Adjacent ivec4s along K cover adjacent k4 (kc-
// contiguous); adjacent n8 blocks are stride K4 apart.
//
// Texture2D (kc dense) form: stored as ivec4 texels. Each texel covers one
// block; image position is (k4, n8). Adjacent texels along x are adjacent k4
// (kc-contiguous), supplying the lane-stride reduction pattern of the coop
// GEMV.
//
// Texture2D (nc) form: stored as ivec4 texels. Each texel covers one block;
// image position is (n8, k4). Adjacent texels along x are adjacent n8 (nc-
// contiguous). Lets nc-walking consumers route weight reads through the
// texture cache.
//
// Interleaved (dp4a-style) byte-pair layout (same for all forms):
//   Each byte of .x holds one (N_even, N_odd) nibble pair at a fixed K.
//   .x byte b (b in {0,1,2,3}) = (N0, K=b) | (N1, K=b) << 4
//   .y byte b                  = (N2, K=b) | (N3, K=b) << 4
// The low nibble of each byte is the even-N row and the high nibble is the
// odd-N row.
${layout_declare_tensor(B, "w", "t_packed_weight", "int", STORAGE, is_scalar_array=False)}
// Input: raw [N, K/2] uint8 data read as uint32.
// Each uint32 holds 8 nibbles = 8 K-values for one N-row.
// Indexed as t_int4_weight[n * K8 + k8] where K8 = ceil(K/8).
${layout_declare_tensor(B, "r", "t_int4_weight", "uint", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ivec2 orig_sizes; // {K, N}
  // Unused — kept so both prepack call sites (buffer and texture2d) can share
  // an identical push-constant layout. The block-row stride is implicit in
  // ceil(N/8) on both paths: for buffer this matches N4_padded/2 (where
  // N4_padded = (N4+1)&~1); for texture2d this is the image's N8 dimension.
  int n4_pitch;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Returns the packed_x / packed_y uint pair for the (k4, n4) tile.
void compute_tile_packed(
    out uint packed_x,
    out uint packed_y,
    const int k4,
    const int n4,
    const int K8,
    const int N) {
  packed_x = 0u;
  packed_y = 0u;

  for (int ni = 0; ni < 4; ++ni) {
    const int n = n4 * 4 + ni;

    // k4 * 4 gives the starting K index. We need 4 consecutive K values.
    // The source has 8 K-nibbles per uint32 at t_int4_weight[n * K8 + k8].
    // k4 * 4 / 8 = k4 / 2 gives the uint32 index along K.
    const int k_start = k4 * 4;
    const int k8_idx = k_start / 8;
    const uint src_word =
        (n < N) ? t_int4_weight[n * K8 + k8_idx] : 0x88888888u;

    // Within this uint32, extract 4 nibbles starting at position (k_start % 8)
    const int nibble_offset = (k_start % 8);

    // Interleaved: byte b holds one nibble from even-N row (low) and one
    // nibble from odd-N row (high). The low/high selection is by ni parity.
    // Even-ni rows go into packed_x's or packed_y's low nibble of each byte,
    // odd-ni rows go into the high nibble.
    for (int ki = 0; ki < 4; ++ki) {
      const uint nibble = (src_word >> (4 * (nibble_offset + ki))) & 0xFu;
      const int bit_offset = 8 * ki + (ni & 1) * 4;
      if (ni < 2) {
        packed_x |= nibble << bit_offset;
      } else {
        packed_y |= nibble << bit_offset;
      }
    }
  }
}

void main() {
  // One invocation = one full 4K x 8N block at logical (k4, n8).
  const int k4 = int(gl_GlobalInvocationID.x);
  const int n8 = int(gl_GlobalInvocationID.y);

  const int K = orig_sizes.x;
  const int N = orig_sizes.y;
  const int K8 = (K + 7) / 8;
  const int K4 = K / 4;
  const int N4 = (N + 3) / 4;
  // N8 = ceil(N4/2). Both buffer (where N4_padded = (N4+1)&~1, so N4_padded/2
  // = (N4+1)/2) and texture2d paths use the same dispatch shape.
  const int N8 = (N4 + 1) / 2;

  if (k4 >= K4 || n8 >= N8) {
    return;
  }

  const int n4_a = 2 * n8;
  const int n4_b = n4_a + 1;

  uint packed_x_a = 0u;
  uint packed_y_a = 0u;
  uint packed_x_b = 0u;
  uint packed_y_b = 0u;

  // Lower tile (n4_a) — always materialized. compute_tile_packed handles
  // n >= N rows with the 0x88888888u (bias-zero) fallback per row.
  compute_tile_packed(packed_x_a, packed_y_a, k4, n4_a, K8, N);

  // Upper tile (n4_b). When n4_b >= N4 the entire tile is OOB along N — use
  // the bias-zero pattern directly so the GEMV/GEMM consumers can safely
  // read whole blocks even when N4 is odd.
  if (n4_b < N4) {
    compute_tile_packed(packed_x_b, packed_y_b, k4, n4_b, K8, N);
  } else {
    packed_x_b = 0x88888888u;
    packed_y_b = 0x88888888u;
  }

  const ivec4 texel = ivec4(
      int(packed_x_a),
      int(packed_y_a),
      int(packed_x_b),
      int(packed_y_b));

$if STORAGE == "texture2d":
  $if WEIGHT_KC == 1:
    // Texture2D (kc dense). Image position = (k4, n8); adjacent texels along
    // x cover adjacent k4 (kc-contiguous).
    imageStore(t_packed_weight, ivec2(k4, n8), texel);
  $else:
    // Texture2D (nc). Image position = (n8, k4); adjacent texels along x
    // cover adjacent n8 (nc-contiguous). Same byte-pair payload as nc-buffer
    // but stored as an ivec4 image2D so consumers route weight reads through
    // the texture cache while keeping the nc walking pattern.
    imageStore(t_packed_weight, ivec2(n8, k4), texel);
$elif WEIGHT_KC == 1:
  // Buffer (kc dense) form. One ivec4 per block at index `n8 * K4 + k4`.
  // Adjacent ivec4s along K cover adjacent k4 (kc-contiguous); adjacent n8
  // blocks are stride K4 apart. Mirrors the kc Tex2D layout so consumers can
  // A/B-test SSBO ivec4 reads vs texelFetch on the same byte-pair payload.
  t_packed_weight[n8 * K4 + k4] = texel;
$else:
  // Buffer (nc) form. One ivec4 per block at index `k4 * N8 + n8` — byte-
  // identical to the legacy 2-ivec2-tile / 4-scalar-int layout because
  // N4_padded = (N4 + 1) & ~1 is even, so 2 * (k4 * N4_padded + 2*n8)
  // = 4 * (k4 * N8 + n8).
  t_packed_weight[k4 * N8 + n8] = texel;
}
