/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_qmat2", "int", STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", "int", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 qmat2_sizes;
  ivec2 orig_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "common.glslh"
#include "linear_int8_weight_block.glslh"

void main() {
  // The size of the source weight tensor is [W=K, H=N]. Each shader invocation
  // processes a 4x4 block. The thread position corresponds to the block index.
  int n4 = int(gl_GlobalInvocationID.x);
  int k4 = int(gl_GlobalInvocationID.y);

  const int K = orig_sizes.x;
  const int N = orig_sizes.y;

  // Determine the total number of blocks and check bounds
  const int N4 = div_up_4(N);
  const int K4 = div_up_4(K);
  if (n4 >= N4 || k4 >= K4) {
    return;
  }

  // Each block is represented as an ivec4. Each int corresponds to a row i.e.
  // N dim of the weight tensor and contains data for 4 columns i.e. K dim.
  Int8WeightBlock block;
  const int n = mul_4(n4);
  if (N - n >= 4) {
    load_block_data_no_checks(block, k4, n, K4, N);
  } else {
    load_block_data_with_checks(block , k4, n, K4, N);
  }

  // The weight blocks are stored in a tranposed manner, such that weight blocks
  // are indexed like packed_weight[k4][n4]. This is to optimize memory
  // coalescing when computing tiled GEMM.
  write_weight_block(block, n4, k4, N4);
}
