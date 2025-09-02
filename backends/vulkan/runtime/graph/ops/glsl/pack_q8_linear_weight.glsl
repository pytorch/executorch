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
  int block_x = int(gl_GlobalInvocationID.x);
  int block_y = int(gl_GlobalInvocationID.y);

  const int N = orig_sizes.y;
  const int K = orig_sizes.x;

  // Each group of 4 8bit values are packed into each uint in the input tensor.
  const int N4 = div_up_4(N);
  const int K4 = div_up_4(K);

  // Check bounds
  if (block_x >= N4 || block_y >= K4) {
    return;
  }

  Int8WeightBlockSourceData src_data;
  const uint k = mul_4(block_y);
  if (K - k >= 4) {
    load_block_source_data_no_checks(src_data, block_x, mul_4(block_y), N4, K);
  } else {
    load_block_source_data_with_checks(src_data, block_x, mul_4(block_y), N4, K);
  }

  Int8WeightBlockPacked packed_block;
  create_packed_block(packed_block, src_data);

  write_packed_block(
      packed_block,
      block_x,
      block_y,
      N4);
}
