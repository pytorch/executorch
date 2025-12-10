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
${layout_declare_tensor(B, "r", "t_packed_int8_input", "int", IO_STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_packed_int8_weight", "int", WEIGHT_STORAGE, is_scalar_array=True)}
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

  const int out_h = out_block_idx.data.y;
  const int out_w = mul_4(out_block_idx.data.x);

  Conv2dBlockExtents in_block_extents = make_block_extents(input_sizes);

  const int Kw4 = div_up_4(conv2d_params.kernel_size.x);

  // Compute 4 channels for 4 output elements.
  ivec4 acc[4];
  [[unroll]] for (int i = 0; i < 4; ++i) {
    acc[i] = ivec4(0);
  }

  for (int ky = 0; ky < conv2d_params.kernel_size.y; ky++) {
    const int h = out_h * conv2d_params.stride.y - conv2d_params.padding.y +
        ky * conv2d_params.dilation.y;

    for (int kx = 0; kx < conv2d_params.kernel_size.x; kx++) {
      const int w = out_w * conv2d_params.stride.x - conv2d_params.padding.x +
          kx * conv2d_params.dilation.x;

      // Load and unpack weights.
      const int packed_weight_4c = load_weight_1w4c(
          kx,
          ky,
          out_block_idx.data.z,
          Kw4,
          out_block_extents.data.z
      );

      const ivec4 weight_4c = unpack_int8x4(packed_weight_4c);

      [[unroll]] for (int subtile_w = 0; subtile_w < 4; ++subtile_w) {
          ivec4 input_texel = unpack_int8x4(load_input_1w4c(
              w + conv2d_params.stride.x * subtile_w,
              h,
              out_block_idx.data.z,
              out_block_extents.data.z,
              in_block_extents));
          acc[subtile_w] += weight_4c * input_texel;
      }
    }
  }

  // Apply input zero point as weight_sum * input_zp.
  vec4 weight_sums = vec4(t_weight_sums[out_block_idx.data.z]);
  const vec4 weight_scales = vec4(t_weight_scales[out_block_idx.data.z]);

  vec4 facc[4];
  [[unroll]] for (int subtile_w = 0; subtile_w < 4; ++subtile_w) {
    facc[subtile_w] = vec4(acc[subtile_w]);
    facc[subtile_w] -= weight_sums * input_zp;
    facc[subtile_w] *= weight_scales * input_scale;
  }

  if (apply_bias > 0) {
    const vec4 bias = vec4(t_bias[out_block_idx.data.z]);
    [[unroll]] for (int subtile_w = 0; subtile_w < 4; ++subtile_w) {
      facc[subtile_w] += bias;
    }
  }

  ivec4 packed_out;
  [[unroll]] for (int subtile_w = 0; subtile_w < 4; ++subtile_w) {
    packed_out[subtile_w] = pack_into_int32(quantize(facc[subtile_w], output_inv_scale, output_zp));
  }

#ifdef PACKED_INT8_OUTPUT_BUFFER
  t_packed_int8_output[tid] = packed_out;
#else
  imageStore(t_packed_int8_output, out_block_idx.data, packed_out);
#endif
}
