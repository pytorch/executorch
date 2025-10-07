/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, "buffer")}
#define T ${texel_load_component_type(DTYPE, "buffer")}

$if IO_STORAGE == "buffer":
  #define PACKED_INT8_OUTPUT_BUFFER
  #define PACKED_INT8_INPUT_BUFFER
$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

#define MAX_WINDOW_WIDTH 12
#define MAX_KERNEL_WIDTH 5

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_output", "int", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_input", "int", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}
${layout_declare_ubo(B, "Conv2DParams", "conv2d_params")}

layout(push_constant) uniform restrict Block {
  float input_scale;
  int input_zp;
  float output_inv_scale;
  int output_zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "1")}

#include "conv2d_dw_q8_utils.glslh"

void main() {
  const int tid = int(gl_GlobalInvocationID.x);
  Conv2dBlockExtents out_block_extents = make_block_extents(output_sizes);

  Conv2dBlockIndex out_block_idx = linear_idx_to_block_idx(
      tid, out_block_extents);

  if (block_idx_out_of_bounds(out_block_idx, out_block_extents)) {
    return;
  }

  const int out_w = mul_4(out_block_idx.data.x);
  const int w_start =
      (out_w * conv2d_params.stride.x) - conv2d_params.padding.x;
  const int w_end = ((out_w + 3) * conv2d_params.stride.x) -
      conv2d_params.padding.x +
      (conv2d_params.kernel_size.x - 1) * conv2d_params.dilation.x;

  Conv2dBlockExtents in_block_extents = make_block_extents(input_sizes);

  const ivec4 input_zps = ivec4(pack_into_int32(ivec4(input_zp)));
  const vec4 weight_scales = vec4(t_weight_scales[out_block_idx.data.z]);

  const int Kw4 = div_up_4(conv2d_params.kernel_size.x);

  FPOutBlock out_block;
  for (int ky = 0; ky < conv2d_params.kernel_size.y; ky++) {
    const int out_h = out_block_idx.data.y;
    const int h = out_h * conv2d_params.stride.y - conv2d_params.padding.y +
        ky * conv2d_params.dilation.y;

    InputWindow1D input_window = load_input_window(
        w_start,
        w_end,
        h,
        out_block_idx.data.z,
        in_block_extents,
        input_scale,
        input_zp,
        input_zps);

    WeightRow weight_row = load_weight_row(
        out_block_idx.data.z,
        ky,
        out_block_extents.data.z,
        conv2d_params.kernel_size.x,
        Kw4,
        weight_scales);

    perform_conv1d(out_block, input_window, weight_row);
  }

  if (apply_bias > 0) {
    const vec4 bias = vec4(t_bias[out_block_idx.data.z]);
    for (int row = 0; row < 4; row++) {
      out_block.data[row] += bias;
    }
  }

  const ivec4 packed_out_block = quantize_and_pack(
      out_block, output_inv_scale, output_zp);

#ifdef PACKED_INT8_OUTPUT_BUFFER
  t_packed_int8_output[tid] = packed_out_block;
#else
  imageStore(t_packed_int8_output, out_block_idx.data, packed_out_block);
#endif
}
