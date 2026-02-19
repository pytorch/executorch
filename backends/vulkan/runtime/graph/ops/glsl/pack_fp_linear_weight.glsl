/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(OUTPUT_STORAGE, DTYPE)}
${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, OUTPUT_STORAGE)}
#define T ${texel_load_component_type(DTYPE, OUTPUT_STORAGE)}

$if OUTPUT_STORAGE == "buffer":
  #define OUTPUT_BUFFER

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_packed_weight", DTYPE, OUTPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer", is_scalar_array=True)}

layout(push_constant) uniform restrict Block {
  // Original weight sizes: [N, K] (out_features, in_features)
  ivec2 orig_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "common.glslh"

void main() {
  // The source weight tensor has size [W=K, H=N] in WHCN format.
  // Each shader invocation processes one vec4 of the output.
  // The thread position (n4, k) corresponds to the output block index.
  //
  // Output layout: [K, N4] where each element is a vec4 containing 4
  // consecutive N values for the same K position.
  // This layout is optimized for tiled matrix multiplication where we
  // iterate over K and accumulate into N.
  //
  // w_tile.data[k][n4] = vec4(W[n4*4+0, k], W[n4*4+1, k], W[n4*4+2, k], W[n4*4+3, k])

  const int n4 = int(gl_GlobalInvocationID.x);
  const int k = int(gl_GlobalInvocationID.y);

  const int K = orig_sizes.x;  // in_features
  const int N = orig_sizes.y;  // out_features

  const int N4 = div_up_4(N);

  if (n4 >= N4 || k >= K) {
    return;
  }

  // Each output vec4 contains 4 consecutive N values for position k
  // Input layout is [N, K] row-major, so element [n, k] is at index n*K + k
  const int n_base = mul_4(n4);

  VEC4_T packed_data = VEC4_T(0);

  // Load 4 consecutive N values for position k
  for (int ni = 0; ni < 4; ++ni) {
    const int n = n_base + ni;
    if (n < N) {
      packed_data[ni] = T(t_weight[n * K + k]);
    }
  }

  // Write to output
  // Output is [K, N4] where each vec4 has 4 N values for one K position
#ifdef OUTPUT_BUFFER
  t_packed_weight[k * N4 + n4] = packed_data;
#else
  imageStore(t_packed_weight, ivec3(n4, k, 0), packed_data);
#endif
}
