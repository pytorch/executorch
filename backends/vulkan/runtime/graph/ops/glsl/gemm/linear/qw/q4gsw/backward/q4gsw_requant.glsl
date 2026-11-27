/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

${define_required_extensions(STORAGE, DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_packed", "int", "buffer", is_scalar_array=False, vec_size=4)}
${layout_declare_tensor(B, "r", "t_latent", DTYPE, STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_scales", DTYPE, "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "latent_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "group_size", "32")}

// STE re-quant of fp32 latent [N, K] into the W_4X8 block-packed 4-bit codes
// the forward reads (mirrors pack_q4_linear_weight__w_4x8.glsl). Each thread
// writes one 4K x 8N block (an ivec4) at (k4, n8).
uint quant_nibble(const int n, const int k, const int N, const int K) {
  const float s = float(t_scales[(k / group_size) * N + n]);
  // roundEven + clamp-in-float match torch.round (half-to-even) then clamp.
  float qf = 0.0;
  if (s != 0.0) {
    qf = clamp(roundEven(float(t_latent[n * K + k]) / s), -8.0, 7.0);
  }
  return uint(int(qf) + 8) & 0xFu;
}

// Pack a 4K x 4N tile into the (packed_x, packed_y) int pair. Byte b holds one
// (even-N low nibble, odd-N high nibble) pair at K = k4*4 + b; rows N0,N1 go to
// packed_x, rows N2,N3 to packed_y.
void pack_tile(
    out uint packed_x,
    out uint packed_y,
    const int k4,
    const int n4,
    const int N,
    const int K) {
  packed_x = 0u;
  packed_y = 0u;
  for (int ni = 0; ni < 4; ++ni) {
    const int n = n4 * 4 + ni;
    for (int b = 0; b < 4; ++b) {
      const uint code = quant_nibble(n, k4 * 4 + b, N, K);
      const int shift = 8 * b + (ni & 1) * 4;
      if (ni < 2) {
        packed_x |= code << shift;
      } else {
        packed_y |= code << shift;
      }
    }
  }
}

void main() {
  const int k4 = int(gl_GlobalInvocationID.x);
  const int n8 = int(gl_GlobalInvocationID.y);

  const int K = latent_sizes.x;
  const int N = latent_sizes.y;
  const int K4 = K >> 2;
  const int N4 = (N + 3) >> 2;
  const int N8 = (N4 + 1) >> 1;

  if (k4 >= K4 || n8 >= N8) {
    return;
  }

  const int n4_a = 2 * n8;
  const int n4_b = n4_a + 1;

  uint packed_x_a = 0u;
  uint packed_y_a = 0u;
  // OOB upper tile (odd N4 boundary) is the bias-zero pattern, matching the
  // forward's prepack padding so the whole block is always readable.
  uint packed_x_b = 0x88888888u;
  uint packed_y_b = 0x88888888u;

  pack_tile(packed_x_a, packed_y_a, k4, n4_a, N, K);
  if (n4_b < N4) {
    pack_tile(packed_x_b, packed_y_b, k4, n4_b, N, K);
  }

  t_packed[k4 * N8 + n8] = ivec4(
      int(packed_x_a), int(packed_y_a), int(packed_x_b), int(packed_y_b));
}
