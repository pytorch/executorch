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

${layout_declare_tensor(B, "w", "t_packed_int4_weight", "int", STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int4_weight", "uint", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 qmat2_sizes;
  ivec2 orig_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define DEBUG_MODE
#include "common.glslh"
#include "linear_int4_weight_block.glslh"

void main() {
  const int k8 = int(gl_GlobalInvocationID.x);
  const int n8 = int(gl_GlobalInvocationID.y);

  const int K = orig_sizes.x;
  const int N = orig_sizes.y;

  // Each shader invocation processes a 4x8 block of the input data.
  const int K4 = div_up_4(K);
  const int K8 = div_up_8(K);
  const int N8 = div_up_8(N);

  // Check bounds
  if (n8 >= N8 || k8 >= K8) {
    return;
  }

  bool should_print = n8 == 0 && k8 == 1;
  should_print = false;

  Int4Weight2xBlockSourceData src_data;
  const int n = mul_8(n8);
  if (N - n >= 8) {
    load_block_source_data_no_checks(src_data, k8, n, K8, N);
  } else {
    load_block_source_data_with_checks(src_data, k8, n, K8, N);
  }

  // A 8Kx8K block of the weight matrix is loaded into memory. This will be
  // split into two blocks each holding 4Kx8N worth of data.
  // The first block contains data for k + (0, 1, 2, 3) i.e. the first 4 columns
  // of the loaded weight block.
  Int4WeightBlockPacked packed_block_1;
  // The second block contains data for k + (4, 5, 6, 7) i.e. the second 4 cols
  // of the loaded weight block
  Int4WeightBlockPacked packed_block_2;
  create_packed_blocks(packed_block_1, packed_block_2, src_data);

  if (should_print) {
    printInt4Weight2xBlockSourceData(src_data);
    printInt4WeightBlockPacked(packed_block_1);
  }


  const int k4 = mul_2(k8);
  write_packed_block(packed_block_1, k4, n8, K4);
  write_packed_block(packed_block_2, k4 + 1, n8, K4);
}
