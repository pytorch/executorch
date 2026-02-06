/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", "int8")}

#define PRECISION ${PRECISION}

${define_active_storage_type(STORAGE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_packed_int8_weight", "int", STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int8_weight", "int8", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 qmat2_sizes;
  ivec4 orig_sizes; // [OC, K_h, K_w, IC]
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "common.glslh"

void main() {
  const int block_x = int(gl_GlobalInvocationID.x);
  const int block_y = int(gl_GlobalInvocationID.y);

  const int kx = block_x % orig_sizes.z;
  const int oc4 = block_x / orig_sizes.z;

  const int OC4 = div_up_4(orig_sizes.x);
  const int IC4 = div_up_4(orig_sizes.w);

  const int nblocks_x = orig_sizes.z * OC4;
  const int nblocks_y = IC4 * orig_sizes.y;

  const int ic4 = block_y % IC4;
  const int ky = block_y / IC4;

  if (block_x >= nblocks_x || block_y >= nblocks_y) {
    return;
  }

  const int oc = mul_4(oc4);
  const int ic = mul_4(ic4);

  const int oc_stride = align_up_4(orig_sizes.y * orig_sizes.z * orig_sizes.w);
  const int oc_offset = oc * oc_stride;
  const int ky_offset = ky * (orig_sizes.z * orig_sizes.w);
  const int kx_offset = kx * orig_sizes.w;
  int buf_idx = oc_offset + ky_offset + kx_offset + ic;

  ivec4 packed_block = ivec4(0);
  for (int row = 0; row < 4; row++) {
    if (oc + row < orig_sizes.x) {
      ivec4 weight_vals = ivec4(0);
      for (int col = 0; col < 4; col++) {
        if (ic + col < orig_sizes.w) {
          weight_vals[col] = int(t_int8_weight[buf_idx + col]);
        }
      }
      packed_block[row] = pack_into_int32(weight_vals);
    }
    buf_idx += oc_stride;
  }

#ifdef USING_BUFFER
  const int out_buf_idx = block_y * (nblocks_x) + block_x;
  t_packed_int8_weight[out_buf_idx] = packed_block;
#else
  imageStore(t_packed_int8_weight, ivec2(block_x, block_y), packed_block);
#endif
}
