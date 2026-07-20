/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Transpose + type-cast: [M, K] contiguous texture3D -> [K, ceil(M/4)]
// vectorized output where each element holds 4 consecutive values along M at a
// given K position.
//
// Output storage is configurable (buffer or texture2d):
//   buffer:    element at index [k * M4 + m4] is OUT_VEC4_T
//   texture2d: texel at (m4, k) is vec4 (width-packed layout [M4, K])
//
// Each thread writes a 4K x 4M tile (4 output vec4s). Texture3D input is
// [M, K] width-packed: texel at (k4, m, 0) holds K[k4*4..k4*4+3].
// Global WG: {K/4, ceil(M/4), 1}

#version 450 core

${define_required_extensions(IN_STORAGE, DTYPE)}
${define_required_extensions(OUT_STORAGE, OUT_DTYPE)}

#define PRECISION ${PRECISION}

#define OUT_VEC4_T ${texel_load_type(OUT_DTYPE, OUT_STORAGE)}

$if OUT_STORAGE == "buffer":
  #define OUTPUT_BUFFER

layout(std430) buffer;

$if OUT_STORAGE == "buffer":
  ${layout_declare_tensor(B, "w", "t_output", OUT_DTYPE, "buffer", is_scalar_array=False)}
$else:
  ${layout_declare_tensor(B, "w", "t_output", OUT_DTYPE, "texture2d")}
${layout_declare_tensor(B, "r", "t_input", DTYPE, IN_STORAGE)}

${layout_declare_ubo(B, "ivec4", "sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int K = sizes.x;
  const int M = sizes.y;
  const int M4 = (M + 3) >> 2;

  const int k4 = int(gl_GlobalInvocationID.x);
  const int m4 = int(gl_GlobalInvocationID.y);

  const int k = k4 * 4;
  const int m = m4 * 4;
  if (k >= K || m >= M) {
    return;
  }

  // Load 4 texels from 4 consecutive rows — each texel has 4 K-values
  vec4 row0 = texelFetch(t_input, ivec3(k4, m, 0), 0);
  vec4 row1 = (m + 1 < M) ? texelFetch(t_input, ivec3(k4, m + 1, 0), 0) : vec4(0.0);
  vec4 row2 = (m + 2 < M) ? texelFetch(t_input, ivec3(k4, m + 2, 0), 0) : vec4(0.0);
  vec4 row3 = (m + 3 < M) ? texelFetch(t_input, ivec3(k4, m + 3, 0), 0) : vec4(0.0);

  // Transpose: row[i][j] -> out[j] = vec4(row0[j], row1[j], row2[j], row3[j])
#ifdef OUTPUT_BUFFER
  t_output[k * M4 + m4] = OUT_VEC4_T(row0.x, row1.x, row2.x, row3.x);
  if (k + 1 < K) {
    t_output[(k + 1) * M4 + m4] = OUT_VEC4_T(row0.y, row1.y, row2.y, row3.y);
  }
  if (k + 2 < K) {
    t_output[(k + 2) * M4 + m4] = OUT_VEC4_T(row0.z, row1.z, row2.z, row3.z);
  }
  if (k + 3 < K) {
    t_output[(k + 3) * M4 + m4] = OUT_VEC4_T(row0.w, row1.w, row2.w, row3.w);
  }
#else
  imageStore(t_output, ivec2(m4, k), vec4(row0.x, row1.x, row2.x, row3.x));
  if (k + 1 < K) {
    imageStore(t_output, ivec2(m4, k + 1), vec4(row0.y, row1.y, row2.y, row3.y));
  }
  if (k + 2 < K) {
    imageStore(t_output, ivec2(m4, k + 2), vec4(row0.z, row1.z, row2.z, row3.z));
  }
  if (k + 3 < K) {
    imageStore(t_output, ivec2(m4, k + 3), vec4(row0.w, row1.w, row2.w, row3.w));
  }
#endif
}
