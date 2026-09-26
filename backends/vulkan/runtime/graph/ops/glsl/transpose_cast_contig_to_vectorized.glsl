/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Transpose + type-cast: [M, K] contiguous buffer -> [K, ceil(M/4)] vectorized
// output where each element holds 4 consecutive values along M at a given K
// position.
//
// Output storage is configurable (buffer or texture2d):
//   buffer:    element at index [k * M4 + m4] is OUT_VEC4_T
//   texture2d: texel at (m4, k) is vec4 (width-packed layout [M4, K])
//
// Each thread writes one output vec4 (4M at one K).
// Global WG: {K, ceil(M/4), 1}

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

  const int k = int(gl_GlobalInvocationID.x);
  const int m4 = int(gl_GlobalInvocationID.y);

  const int m = m4 * 4;
  if (m >= M || k >= K) {
    return;
  }

  float v0 = t_input[m * K + k];
  float v1 = (m + 1 < M) ? t_input[(m + 1) * K + k] : 0.0;
  float v2 = (m + 2 < M) ? t_input[(m + 2) * K + k] : 0.0;
  float v3 = (m + 3 < M) ? t_input[(m + 3) * K + k] : 0.0;

#ifdef OUTPUT_BUFFER
  t_output[k * M4 + m4] = OUT_VEC4_T(v0, v1, v2, v3);
#else
  imageStore(t_output, ivec2(m4, k), vec4(v0, v1, v2, v3));
#endif
}
