/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define IN_T ${buffer_scalar_type(IN_DTYPE)}
#define OUT_T ${buffer_scalar_type(OUT_DTYPE)}

${define_active_storage_type("buffer")}
${define_required_extensions(IN_DTYPE)}
${define_required_extensions(OUT_DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "r", "t_in", IN_DTYPE, "buffer")}
${layout_declare_tensor(B, "w", "t_out", OUT_DTYPE, "buffer")}

$if MODE == "per_tensor":
  layout(push_constant) uniform restrict Block {
    float scale;
    int zero_point;
    int quant_min;
    int quant_max;
  };
$else:
  ${layout_declare_tensor(B, "r", "t_scale", "float", "buffer")}
  ${layout_declare_tensor(B, "r", "t_zero_point", "int", "buffer")}

  layout(push_constant) uniform restrict Block {
    int num_tokens;
    int quant_min;
    int quant_max;
  };

${layout_declare_ubo(B, "ivec4", "t_in_sizes")}
${layout_declare_ubo(B, "ivec4", "t_in_strides")}
${layout_declare_ubo(B, "ivec4", "t_out_sizes")}
${layout_declare_ubo(B, "ivec4", "t_out_strides")}

#include "indexing_utils.h"
#include "dequantize.glslh"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
$if MODE == "per_tensor":
  const ivec4 pos = ivec4(
      gl_GlobalInvocationID.x,
      gl_GlobalInvocationID.y,
      gl_GlobalInvocationID.z,
      0);

  const int t_in_idx = tidx_to_bufi(pos, t_in_strides);
  const int t_out_idx = tidx_to_bufi(pos, t_out_strides);

  IN_T qvalue = t_in[t_in_idx];
  OUT_T value;

  value = dequantize_val(qvalue, scale, zero_point);

  t_out[t_out_idx] = value;

$if MODE == "per_token":
  const ivec4 pos = ivec4(
      gl_GlobalInvocationID.x,
      gl_GlobalInvocationID.y,
      gl_GlobalInvocationID.z,
      0);

  const int t_in_idx = tidx_to_bufi(pos, t_in_strides);
  const int t_out_idx = tidx_to_bufi(pos, t_out_strides);

  // Skip if out of bounds
  if (t_in_idx >= t_in_sizes.x * t_in_sizes.y * t_in_sizes.z * t_in_sizes.w) {
    return;
  }

  IN_T qvalue = t_in[t_in_idx];
  OUT_T value;

  // Calculate logical position from linear index and strides
  ivec4 logical_pos;
  int remaining = t_in_idx;

  logical_pos.x = remaining % t_in_sizes.x;
  remaining /= t_in_sizes.x;

  logical_pos.y = remaining % t_in_sizes.y;
  remaining /= t_in_sizes.y;

  logical_pos.z = remaining % t_in_sizes.z;
  remaining /= t_in_sizes.z;

  logical_pos.w = remaining;

  // Calculate token index based on logical position
  int token_idx = 0;

  // Check dimensions to determine how to calculate token_idx
  if (t_in_sizes.w > 1) {
    // 4D tensor
    token_idx = logical_pos.w * (t_in_sizes.z * t_in_sizes.y) + logical_pos.z * t_in_sizes.y + logical_pos.y;
  } else if (t_in_sizes.z > 1) {
    // 3D tensor
    token_idx = logical_pos.z * t_in_sizes.y + logical_pos.y;
  } else if (t_in_sizes.y > 1) {
    // 2D tensor
    token_idx = logical_pos.y;
  }
  // For 1D tensor, token_idx remains 0

  // Make sure token_idx is within bounds
  token_idx = min(token_idx, num_tokens - 1);

  value = dequantize_val(qvalue, t_scale[token_idx], t_zero_point[token_idx]);

  t_out[t_out_idx] = value;
}
