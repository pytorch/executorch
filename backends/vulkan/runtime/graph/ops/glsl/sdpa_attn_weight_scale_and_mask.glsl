/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type(STORAGE)}
${define_required_extensions(DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "rw", "attn_weight", DTYPE, STORAGE)}

$if STORAGE == "buffer":
  ${layout_declare_ubo(B, "ivec4", "attn_weight_sizes")}
  ${layout_declare_ubo(B, "ivec4", "attn_weight_strides")}
  ${layout_declare_ubo(B, "ivec4", "q_projected_sizes")}
  ${layout_declare_ubo(B, "int", "input_pos")}
$else:
  ${layout_declare_ubo(B, "ivec3", "attn_weight_limits")}
  ${layout_declare_ubo(B, "ivec4", "q_projected_sizes")}
  ${layout_declare_ubo(B, "int", "input_pos")}

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#ifdef USING_BUFFER

/***************************
 ** Buffer Implementation **
 ***************************/

void main() {
  const ivec4 attn_weight_idx = ivec4(
      gl_GlobalInvocationID.x,
      gl_GlobalInvocationID.y,
      gl_GlobalInvocationID.z,
      0);

  if (any(greaterThanEqual(attn_weight_idx, attn_weight_sizes))) {
    return;
  }

  const T scale = T(1.0 / sqrt(float(q_projected_sizes.x)));

  const int attn_weight_id = tidx_to_bufi(attn_weight_idx, attn_weight_strides);
  if (attn_weight_idx.x <= attn_weight_idx.y + input_pos) {
    attn_weight[attn_weight_id] = attn_weight[attn_weight_id] * scale;
  } else {
    // No keyword for -inf. Must "create" it by division by 0.
    attn_weight[attn_weight_id] = T(-1.0 / 0.0);
  }
}

#else

/****************************
 ** Texture Implementation **
 ****************************/

/*
 * This implementation assumes that the packed dim of the attn_weight is 0.
 */
void main() {
  const ivec3 attn_weight_pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(attn_weight_pos, attn_weight_limits))) {
    return;
  }

  const float scale = float(1.0 / sqrt(float(q_projected_sizes.x)));

  vec4 outtex = imageLoad(attn_weight, attn_weight_pos) * scale;

  // Mask out the upper triangular of attn_weight to -inf
  [[unroll]] for (int i = 0; i < 4; ++i) {
    if (attn_weight_pos.x * 4 + i > attn_weight_pos.y + input_pos) {
      outtex[i] = float(-1.0 / 0.0);
    }
  }

  write_texel(attn_weight, attn_weight_pos, outtex);
}

#endif // USING_BUFFER
