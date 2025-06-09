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
#define IVEC4_T ${texel_load_type(IN_DTYPE, STORAGE)}

#define OUT_T ${buffer_scalar_type(OUT_DTYPE)}
#define FVEC4_T ${texel_load_type(OUT_DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}
${define_required_extensions(IN_DTYPE)}
${define_required_extensions(OUT_DTYPE)}

// Need this in order to properly handle overflow for dequantize_val
// since there is an inconsistency between the cpu logic
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "r", "t_in", IN_DTYPE, STORAGE)}
${layout_declare_tensor(B, "w", "t_out", OUT_DTYPE, STORAGE)}

$if MODE == "per_tensor":
  layout(push_constant) uniform restrict Block {
    float scale;
    int zero_point;
    int quant_min;
    int quant_max;
  };
$else:
  ${layout_declare_tensor(B, "r", "t_scale", "float", STORAGE)}
  ${layout_declare_tensor(B, "r", "t_zero_point", "int", STORAGE)}

  layout(push_constant) uniform restrict Block {
    int num_tokens;
    int quant_min;
    int quant_max;
  };

$if STORAGE == "buffer":
  ${layout_declare_ubo(B, "ivec4", "t_in_sizes")}
  ${layout_declare_ubo(B, "ivec4", "t_in_strides")}
  ${layout_declare_ubo(B, "ivec4", "t_out_sizes")}
  ${layout_declare_ubo(B, "ivec4", "t_out_strides")}
$else:
  ${layout_declare_ubo(B, "ivec3", "t_in_limits")}
  ${layout_declare_ubo(B, "ivec3", "t_out_limits")}

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

OUT_T dequantize_val(IN_T qvalue, float scale_val, int zero_point_val) {
  $if MODE == "per_tensor":
    // out_data_ptr[i] = static_cast<OUT_CTYPE>((input_data_ptr[i] - static_cast<int32_t>(zero_point)) * scale);
    //                                           -2147483648 - 100 = 2147483548 * 0.0001 > cast to float32
    OUT_T value = OUT_T(float(int(qvalue) - zero_point_val) * scale_val);

  $if MODE == "per_token":
    // out_data_ptr[i] = static_cast<OUT_CTYPE>((input_data_ptr[i] - zero_point) * scale);
    //                                           -2147483648 - 100 = -2147483748 * 0.0001 > cast to float32
    OUT_T value = OUT_T(float(int(qvalue) - int64_t(zero_point_val)) * scale_val);

  return value;
}

#ifdef USING_BUFFER

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

#else

void main() {
$if MODE == "per_tensor":
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // Skip if out of bounds
  if (any(greaterThanEqual(pos, t_in_limits))) {
    return;
  }

  IVEC4_T intex = load_texel(t_in, pos);
  FVEC4_T outtex;

  [[unroll]] for (int i = 0; i < 4; ++i) {
    IN_T qvalue = IN_T(intex[i]);
    OUT_T value = dequantize_val(qvalue, scale, zero_point);
    outtex[i] = value;
  }
  write_texel(t_out, pos, outtex);

$if MODE == "per_token":
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // Skip if out of bounds
  if (any(greaterThanEqual(pos, t_in_limits))) {
    return;
  }

  IVEC4_T intex = load_texel(t_in, pos);

  int token_idx = 0;
  ivec3 dims = t_in_limits;

  if (dims.z > 1) {
    // 3D tensor
    token_idx = pos.z * dims.y + pos.y;
  } else if (dims.y > 1) {
    // 2D tensor
    token_idx = pos.y;
  }
  // For 1D tensor, token_idx remains 0

  // Make sure token_idx is within bounds
  token_idx = min(token_idx, num_tokens - 1);

  // For texture storage, we need to calculate the texel position and component index
  int texel_idx = token_idx / 4;
  int comp_idx = token_idx % 4;

  vec4 scale_vals = load_texel(t_scale, ivec3(texel_idx, 0, 0));
  ivec4 zp_vals = load_texel(t_zero_point, ivec3(texel_idx, 0, 0));

  float scale_val = scale_vals[comp_idx];
  int zero_point_val = zp_vals[comp_idx];

  FVEC4_T outtex;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    IN_T qvalue = IN_T(intex[i]);
    OUT_T value = dequantize_val(qvalue, scale_val, zero_point_val);
    outtex[i] = value;
  }

  write_texel(t_out, pos, outtex);

}

#endif
