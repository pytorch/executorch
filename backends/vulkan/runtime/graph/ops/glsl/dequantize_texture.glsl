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
#define IVEC4_T ${texel_load_type(IN_DTYPE, "texture3d")}

#define OUT_T ${buffer_scalar_type(OUT_DTYPE)}
#define FVEC4_T ${texel_load_type(OUT_DTYPE, "texture3d")}

#define ${MODE}

${define_active_storage_type("texture3d")}
${define_required_extensions(IN_DTYPE)}
${define_required_extensions(OUT_DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", OUT_DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", IN_DTYPE, "texture3d")}

$if MODE == "per_tensor":
  layout(push_constant) uniform restrict Block {
    float scale;
    int zero_point;
    int quant_min;
    int quant_max;
  };
$if MODE == "per_token":
  ${layout_declare_tensor(B, "r", "t_scale", "float", "buffer")}
  ${layout_declare_tensor(B, "r", "t_zero_point", "int", "buffer")}

  layout(push_constant) uniform restrict Block {
    int num_tokens;
    int quant_min;
    int quant_max;
  };

${layout_declare_ubo(B, "ivec3", "t_in_limits")}
${layout_declare_ubo(B, "ivec3", "t_out_limits")}

#include "indexing_utils.h"
#include "dequantize.glslh"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#ifdef per_tensor

void dequantize_per_tensor() {
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
    $if OUT_DTYPE == "double":
      outtex[i] = float(value);
    $else:
      outtex[i] = value;
  }
  write_texel(t_out, pos, outtex);
}

#else

void dequantize_per_token() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

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

  token_idx = min(token_idx, num_tokens - 1);

  // Scale and zero_point are prepacked as buffers, so direct access
  float scale_val = t_scale[token_idx];
  int zero_point_val = t_zero_point[token_idx];

  FVEC4_T outtex;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    IN_T qvalue = IN_T(intex[i]);
    OUT_T value = dequantize_val(qvalue, scale_val, zero_point_val);
    $if OUT_DTYPE == "double":
      outtex[i] = float(value);
    $else:
      outtex[i] = value;
  }

  write_texel(t_out, pos, outtex);
}

#endif

void main() {
  dequantize_${MODE}();
}
