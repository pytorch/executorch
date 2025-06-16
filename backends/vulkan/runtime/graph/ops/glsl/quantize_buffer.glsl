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

#define ${MODE}

${define_active_storage_type("buffer")}
${define_required_extensions(IN_DTYPE)}
${define_required_extensions(OUT_DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "t_out", OUT_DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_in", IN_DTYPE, "buffer")}

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

${layout_declare_ubo(B, "int", "out_numel")}
${layout_declare_ubo(B, "ivec4", "t_in_sizes")}
${layout_declare_ubo(B, "ivec4", "t_in_strides")}
${layout_declare_ubo(B, "ivec4", "t_out_sizes")}
${layout_declare_ubo(B, "ivec4", "t_out_strides")}

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}

#include "quantize.glslh"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const lowp ivec4 out_dim_order = unhash_dim_order(out_layout);
const lowp ivec4 in_dim_order = unhash_dim_order(in_layout);

#ifdef per_tensor

void quantize_per_tensor() {
  const int out_bufi = int(gl_GlobalInvocationID.x);

  if (out_bufi >= out_numel) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, t_out_strides, out_dim_order);
  const int in_bufi = tidx_to_bufi(out_tidx, t_in_strides);

  IN_T value = t_in[in_bufi];
  OUT_T qvalue = quantize_val(value, scale, zero_point);

  t_out[out_bufi] = qvalue;
}

#else

void quantize_per_token() {
  const int out_bufi = int(gl_GlobalInvocationID.x);

  if (out_bufi >= out_numel) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, t_out_strides, out_dim_order);
  const int in_bufi = tidx_to_bufi(out_tidx, t_in_strides);

  IN_T value = t_in[in_bufi];

  int token_idx = 0;

  if (t_out_sizes.w > 1) {
    // 4D tensor
    token_idx = out_tidx.w * (t_out_sizes.z * t_out_sizes.y) + out_tidx.z * t_out_sizes.y + out_tidx.y;
  } else if (t_out_sizes.z > 1) {
    // 3D tensor
    token_idx = out_tidx.z * t_out_sizes.y + out_tidx.y;
  } else if (t_out_sizes.y > 1) {
    // 2D tensor
    token_idx = out_tidx.y;
  }
  // For 1D tensor, token_idx remains 0

  token_idx = min(token_idx, num_tokens - 1);

  OUT_T qvalue = quantize_val(value, t_scale[token_idx], t_zero_point[token_idx]);

  t_out[out_bufi] = qvalue;
}

#endif

void main() {
  quantize_${MODE}();
}
