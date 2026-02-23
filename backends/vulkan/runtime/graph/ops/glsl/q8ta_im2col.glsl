/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define PACKED_INT8_OUTPUT_BUFFER

layout(std430) buffer;

#include "indexing.glslh"
#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_output", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_input", "int", "buffer", is_scalar_array=True)}

// Metadata for im2col output and input tensors (layout-agnostic)
${layout_declare_ubo(B, "BufferMetadata", "im2col_outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}
${layout_declare_ubo(B, "Conv2DParams", "conv2d_params")}

${layout_declare_spec_const(C, "int", "apply_bias", "1")}

// Layout specialization constants
${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}

layout(push_constant) uniform restrict Block {
  int zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int out_buf_idx = int(gl_GlobalInvocationID.x);

  // Extract sizes from BufferMetadata
  const ivec4 im2col_sizes = ivec4(im2col_outp.sizes[0]);
  const ivec4 input_sizes = ivec4(inp.sizes[0]);

  // im2col block extents
  const int im2col_W4 = div_up_4(im2col_sizes.x);
  const int im2col_H = im2col_sizes.y;
  const int im2col_Z4 = div_up_4(im2col_sizes.z);

  // im2col block index from linear output buffer index
  const int c4_idx = out_buf_idx % im2col_Z4;
  const int row = out_buf_idx / im2col_Z4;
  const int w4_idx = row % im2col_W4;
  const int h_idx = row / im2col_W4;

  // out of bounds check
  if (w4_idx >= im2col_W4 || h_idx >= im2col_H || c4_idx >= im2col_Z4) {
    return;
  }

  const int im2col_w = mul_4(w4_idx);
  const int im2col_h = h_idx;
  const int im2col_k = mul_4(c4_idx);

  const int group_idx = im2col_k / conv2d_params.K_per_group;
  const int k_in_group = im2col_k % conv2d_params.K_per_group;

  const int c_in_group = k_in_group % conv2d_params.in_channels_per_group;
  const int krow = k_in_group / conv2d_params.in_channels_per_group;
  const int kernel_x = krow % conv2d_params.kernel_size.x;
  const int kernel_y = krow / conv2d_params.kernel_size.x;

  // Base input position
  const int input_x_base =
      (im2col_w * conv2d_params.stride.x) - conv2d_params.padding.x +
      (kernel_x * conv2d_params.dilation.x);
  const int input_y =
      (im2col_h * conv2d_params.stride.y) - conv2d_params.padding.y +
      (kernel_y * conv2d_params.dilation.y);
  const int input_z =
      group_idx * conv2d_params.in_channels_per_group + c_in_group;

  // Input tensor extents
  const int input_W = input_sizes.x;
  const int input_H = input_sizes.y;
  const int input_Z4 = div_up_4(input_sizes.z);

  const int zp_packed = pack_into_int32(ivec4(zp));
  const int z4 = div_4(input_z);

  // Check if y and z are in bounds (constant for all 4 width elements)
  const bool y_z_in_bounds =
      (input_y >= 0 && input_y < input_H && z4 >= 0 && z4 < input_Z4);

  // Load 4 elements from input, one for each output width position.
  // Each loaded int contains 4 packed int8 channel values.
  ivec4 im2col_block;
  for (int i = 0; i < 4; i++) {
    const int x = input_x_base + i * conv2d_params.stride.x;
    if (!y_z_in_bounds || x < 0 || x >= input_W) {
      im2col_block[i] = zp_packed;
    } else {
      const int x4 = div_4(x);
      const int x_mod = mod_4(x);
      int scalar_idx;
      if (get_outer_packed_dim_block_size(inp_layout) == 1) {
        scalar_idx = input_y * int(inp.strides[0][1])
                     + x * int(inp.strides[0][0])
                     + z4 * int(inp.strides[0][2]);
      } else {
        scalar_idx = mul_4(
            input_y * int(inp.strides[0][1])
            + x4 * int(inp.strides[0][0])
            + z4) + x_mod;
      }
      im2col_block[i] = t_packed_int8_input[scalar_idx];
    }
  }

  // store_packed_int8_output_tile (with TILE_M4=1, TILE_N4=1)
  const int buffer_idx = h_idx * int(im2col_outp.strides[0][1])
                         + w4_idx * int(im2col_outp.strides[0][0])
                         + c4_idx;

  if (w4_idx < im2col_W4 && c4_idx < im2col_Z4) {
    t_packed_int8_output[buffer_idx] = im2col_block;
  }
}
