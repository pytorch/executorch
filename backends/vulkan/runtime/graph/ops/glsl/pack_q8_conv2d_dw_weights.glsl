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

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_packed_int8_weight", "int", STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int8_weight", "int", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 qmat2_sizes;
  ivec3 orig_sizes; // [K_h, aligned_K_w, OC]
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "common.glslh"

void main() {
  // The size of the source weight tensor is [K_h, aligned_K_w, OC] for depthwise conv.
  // Each shader invocation processes a 4x4 block of weights for a group of output channels.
  const int oc4 = int(gl_GlobalInvocationID.x);
  const int k4 = int(gl_GlobalInvocationID.y);
  const int k = mul_4(k4);

  const int H = orig_sizes.x;
  const int orig_W = orig_sizes.y;
  const int W4 = div_up_4(orig_W);
  const int OC = orig_sizes.z;

  const int h = k4 / W4;
  const int w4 = k4 % W4;
  const int w = mul_4(w4);

  // Determine the total number of blocks and check bounds
  const int OC4 = div_up_4(OC);
  const int K4 = H * W4;

  if (oc4 >= OC4 || k4 >= K4) {
    return;
  }

  ivec4 packed_block;

  int buf_idx = (h * orig_W + w) * OC4 + oc4;
  int r_limit = min(4, orig_W - w);
  [[unroll]] for (int r = 0; r < r_limit; r++) {
    packed_block[r] = t_int8_weight[buf_idx];
    buf_idx += OC4;
  }
  [[unroll]] for (int r = r_limit; r < 4; r++) {
    packed_block[r] = 0;
  }

#ifdef USING_BUFFER
  t_packed_int8_weight[k4 * OC4 + oc4] = packed_block;
#else
  imageStore(t_packed_int8_weight, ivec2(oc4, k4), packed_block);
#endif
}
