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

// Push constants are uploaded in 16-byte chunks (one ivec4 each) to comply
// with the per-entry size limit.
layout(push_constant) uniform restrict Block {
  ivec4 dims0; // (N=C_out, K=K_total, C_in, Cin_padded)
  ivec4 dims1; // (K_h, K_w, _unused, _unused)
};

#define N          dims0.x
#define K          dims0.y
#define C_IN       dims0.z
#define CIN_PADDED dims0.w
#define K_H        dims1.x
#define K_W        dims1.y

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Packs the ORIGINAL serialized conv2d weight [C_out, C_in, K_h, K_w]
// (PyTorch row-major contiguous) directly into the 4OC x 4IC blocked layout
// that conv2d_gemm.glsl loads via load_packed_weight_tile_with_checks, with no
// CPU-side repack of the serialized data.
//
// The GEMM treats the weight as [N=C_out, K=K_total] with the im2col K-axis
// layout
//   k = (ki * K_w + kj) * Cin_padded + ci
// so each 4-tile of K holds 4 consecutive ci for one (ki, kj). Lanes with
// ci >= C_in are zero (Cin padding).
//
// This produces a byte-identical packed tensor to running the generic
// pack_fp_linear_weight (is_transposed=1) over the CPU-flattened [C_out,
// K_total] weight: a 4x4 block is transposed so packed[dk] = {w_flat[n4*4 +
// 0..3][k4*4 + dk]}.

// Read the flattened weight scalar at logical (n, k) directly from the
// serialized [C_out, C_in, K_h, K_w] buffer, applying the im2col K decode and
// Cin padding. Returns 0 for out-of-range n / padding ci lanes.
T load_flat_weight_scalar(const int n, const int k) {
  if (n >= N || k >= K) {
    return T(0);
  }
  const int ci = k % CIN_PADDED;
  if (ci >= C_IN) {
    return T(0); // Cin padding lane
  }
  const int krow = k / CIN_PADDED; // ki * K_w + kj
  const int kj = krow % K_W;
  const int ki = krow / K_W;
  // Serialized [C_out, C_in, K_h, K_w] contiguous index.
  const int src_idx = ((n * C_IN + ci) * K_H + ki) * K_W + kj;
  return T(t_weight_src[src_idx]);
}

VEC4_T load_flat_weight_row(const int n, const int k_base) {
  return VEC4_T(
      load_flat_weight_scalar(n, k_base),
      load_flat_weight_scalar(n, k_base + 1),
      load_flat_weight_scalar(n, k_base + 2),
      load_flat_weight_scalar(n, k_base + 3));
}

void main() {
  const int n4 = int(gl_GlobalInvocationID.x);
  const int k4 = int(gl_GlobalInvocationID.y);

  const int K4 = div_up_4(K);
  const int N4 = div_up_4(N);

  if (n4 >= N4 || k4 >= K4) {
    return;
  }

  // Read 4 N-rows at the k4 column block, transpose into a 4OC x 4IC block.
  // Mirrors the is_transposed branch of pack_fp_linear_weight.
  VEC4_T src_rows[4];
  [[unroll]] for (int dn = 0; dn < 4; dn++) {
    src_rows[dn] = load_flat_weight_row(n4 * 4 + dn, k4 * 4);
  }
  [[unroll]] for (int dk = 0; dk < 4; dk++) {
    VEC4_T out_val = VEC4_T(
        src_rows[0][dk], src_rows[1][dk], src_rows[2][dk], src_rows[3][dk]);
#ifdef OUTPUT_BUFFER
    t_weight_packed[(k4 * N4 + n4) * 4 + dk] = out_val;
#else
    imageStore(t_weight_packed, ivec2(n4 * 4 + dk, k4), out_val);
#endif
  }
}
