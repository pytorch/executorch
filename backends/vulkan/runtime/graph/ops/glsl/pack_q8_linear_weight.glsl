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

#extension GL_EXT_debug_printf : enable

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_qmat2", "int", STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", "int", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 qmat2_sizes;
  ivec2 orig_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define DEBUG_MODE
#include "common.glslh"
#include "linear_int8_weight_block.glslh"

void main() {
  uint block_x = gl_GlobalInvocationID.x;
  uint block_y = gl_GlobalInvocationID.y;

  bool should_print = (block_x == 0) && (block_y == 0);

  const int N = orig_sizes.y;
  const int K = orig_sizes.x;

  // Each group of 4 8bit values are packed into each uint in the input tensor.
  const int N4 = div_up_4(N);
  const int K4 = div_up_4(K);

  if (should_print) {
    debugPrintfEXT("N: %d, K: %d, N4: %d, K4: %d", N, K, N4, K4);
  }

  // Check bounds
  if (block_x >= N4 || block_y >= K4) {
    return;
  }

  Int8WeightBlockSourceData src_data;
  load_block_source_data(src_data, block_x, mul_4(block_y), N4);

  Int8WeightBlockPacked packed_block;
  create_packed_block(packed_block, src_data);

  if (false && should_print) {
    debugPrintfEXT("block_x: %d, block_y: %d\\n", block_x, block_y);
    printInt8WeightBlockSourceData(src_data);
  }

  write_packed_block(
      packed_block,
      block_x,
      block_y,
      N4);
}
