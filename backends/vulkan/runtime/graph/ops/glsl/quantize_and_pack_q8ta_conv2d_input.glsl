/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, INPUT_STORAGE)}
#define T ${texel_load_component_type(DTYPE, INPUT_STORAGE)}

$if OUTPUT_STORAGE == "buffer":
  #define OUTPUT_BUFFER
$if INPUT_STORAGE == "buffer":
  #define INPUT_BUFFER

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_input", "int", OUTPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_fp_input", DTYPE, INPUT_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(push_constant) uniform restrict Block {
  float inv_scale;
  int zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "conv2d_fp_input_block_load.glslh"
#include "conv2d_int8_input_block_store.glslh"

void main() {
  Conv2dBlockIndex block_idx;
  block_idx.data = ivec3(gl_GlobalInvocationID);

  Conv2dBlockExtents block_extents = make_block_extents(input_sizes);
  if (block_idx_out_of_bounds(block_idx, block_extents)) {
    return;
  }

  FPActivationBlock fp_input_block;
  load_fp_input_block(fp_input_block, block_idx);

  Int8ActivationBlock int8_input_block;
  quantize_fp_activation_block(int8_input_block, fp_input_block, inv_scale, zp);

  store_packed_int8_input_block(block_idx, block_extents, int8_input_block);
}
