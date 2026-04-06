/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define BUF_T ${buffer_scalar_type(BUF_DTYPE)}
#define VEC4_T ${texel_load_type(DTYPE, PACKED_STORAGE)}
#define T ${texel_load_component_type(DTYPE, PACKED_STORAGE)}

$if PACKED_STORAGE == "buffer":
  #define OUTPUT_BUFFER

#extension GL_EXT_control_flow_attributes : require

${define_required_extensions("buffer", BUF_DTYPE)}
$if PACKED_STORAGE != "buffer":
  ${define_required_extensions(PACKED_STORAGE, DTYPE)}

layout(std430) buffer;

#include "common.glslh"

$if PACKED_STORAGE == "buffer":
  ${layout_declare_tensor(B, "w", "t_weight_packed", DTYPE, "buffer", is_scalar_array=False)}
$else:
  ${layout_declare_tensor(B, "w", "t_weight_packed", DTYPE, PACKED_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_src", BUF_DTYPE, "buffer", is_scalar_array=True)}

layout(push_constant) uniform restrict Block {
  int N;
  int K;
  int B;
  int is_transposed;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Packs fp linear weight into 4OC x 4IC blocked layout.
//
// Source data is contiguous row-major with no per-row padding, so scalar reads
// are used to correctly handle dimensions that are not multiples of 4.
//
// When is_transposed != 0, source is [B, N, K] row-major (transposed weight):
//   Scalar at (b, n, k) = t_weight_src[b * N * K + n * K + k]
//   Each 4x4 block is transposed so that:
//     packed[dk] = {w[k4*4+dk][n4*4+0..3]}
//
// When is_transposed == 0, source is [B, K, N] row-major (non-transposed):
//   Scalar at (b, k, n) = t_weight_src[b * K * N + k * N + n]
//   Already in the desired column grouping, no transpose needed.
//
// Output: batch-stacked blocked layout indexed by (b, k4, n4, dk).

T load_scalar(const int idx) {
  return T(t_weight_src[idx]);
}

VEC4_T load_scalar_row(const int row_base, const int col, const int max_col) {
  return VEC4_T(
      load_scalar(row_base + col),
      (col + 1 < max_col) ? load_scalar(row_base + col + 1) : T(0),
      (col + 2 < max_col) ? load_scalar(row_base + col + 2) : T(0),
      (col + 3 < max_col) ? load_scalar(row_base + col + 3) : T(0));
}

void main() {
  const int n4 = int(gl_GlobalInvocationID.x);
  const int k4 = int(gl_GlobalInvocationID.y);
  const int b = int(gl_GlobalInvocationID.z);

  const int K4 = div_up_4(K);
  const int N4 = div_up_4(N);

  if (n4 >= N4 || k4 >= K4 || b >= B) {
    return;
  }

  if (is_transposed != 0) {
    // Source is [N, K] or [B, N, K].
    // Read 4 N-rows at the k4 column block, transpose into 4OC x 4IC block.
    const int batch_offset = b * N * K;
    VEC4_T src_rows[4];
    [[unroll]] for (int dn = 0; dn < 4; dn++) {
      int n = n4 * 4 + dn;
      if (n < N) {
        src_rows[dn] = load_scalar_row(batch_offset + n * K, k4 * 4, K);
      } else {
        src_rows[dn] = VEC4_T(0);
      }
    }
    [[unroll]] for (int dk = 0; dk < 4; dk++) {
      VEC4_T out_val = VEC4_T(
          src_rows[0][dk], src_rows[1][dk],
          src_rows[2][dk], src_rows[3][dk]);
#ifdef OUTPUT_BUFFER
      t_weight_packed[((b * K4 + k4) * N4 + n4) * 4 + dk] = out_val;
#else
      imageStore(t_weight_packed, ivec2(n4 * 4 + dk, b * K4 + k4), out_val);
#endif
    }
  } else {
    // Source is [K, N] or [B, K, N].
    // Read 4 K-rows at the n4 column block. No transpose needed.
    const int batch_offset = b * K * N;
    [[unroll]] for (int dk = 0; dk < 4; dk++) {
      int k = k4 * 4 + dk;
      VEC4_T out_val;
      if (k < K) {
        out_val = load_scalar_row(batch_offset + k * N, n4 * 4, N);
      } else {
        out_val = VEC4_T(0);
      }
#ifdef OUTPUT_BUFFER
      t_weight_packed[((b * K4 + k4) * N4 + n4) * 4 + dk] = out_val;
#else
      imageStore(t_weight_packed, ivec2(n4 * 4 + dk, b * K4 + k4), out_val);
#endif
    }
  }
}
