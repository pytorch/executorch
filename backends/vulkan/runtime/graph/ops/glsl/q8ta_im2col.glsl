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

#define TILE_M4 1
#define TILE_N4 1
#define TILE_K4 1

#define TILE_M 4
#define TILE_N 4
#define TILE_K 4

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
${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "im2col_outp_layout", "CONTIG_LAYOUT_INT")}

layout(push_constant) uniform restrict Block {
  int zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "conv2d_int8_output_tile_store.glslh"

// Compute input tensor index from im2col coordinates
TensorIndex4D get_input_tidx(
    const int im2col_w,
    const int im2col_h,
    const int k_in_group,
    const int group_idx) {
  TensorIndex4D tidx;
  tidx.data.w = 0;

  const int c_in_group = k_in_group % conv2d_params.in_channels_per_group;
  const int row = k_in_group / conv2d_params.in_channels_per_group;
  const int kernel_x = row % conv2d_params.kernel_size.x;
  const int kernel_y = row / conv2d_params.kernel_size.x;

  tidx.data.z = group_idx * conv2d_params.in_channels_per_group + c_in_group;

  tidx.data.x = (im2col_w * conv2d_params.stride.x) - conv2d_params.padding.x +
      (kernel_x * conv2d_params.dilation.x);
  tidx.data.y = (im2col_h * conv2d_params.stride.y) - conv2d_params.padding.y +
      (kernel_y * conv2d_params.dilation.y);

  return tidx;
}

// Load a single int8 value from the input tensor using layout-agnostic indexing
int load_input_element(const TensorIndex4D tidx, const int input_zp) {
  // Bounds checking
  if (any(lessThan(tidx.data, ivec4(0))) ||
      any(greaterThanEqual(tidx.data, ivec4(inp.sizes[0])))) {
    return input_zp;
  }

  // Use layout-agnostic indexing to get buffer position
  int texel_idx;
  if (get_outer_packed_dim_block_size(inp_layout) == 1) {
    // For 4C or 4C1W layouts: use tensor4d_idx_to_texel_idx
    texel_idx = tensor4d_idx_to_texel_idx(inp, tidx, inp_layout);
  } else {
    // For 4W4C layout: compute index directly
    const int w4 = div_4(tidx.data[0]);
    const int c4 = div_4(tidx.data[2]);
    const int h_stride = int(inp.strides[0][1]);
    const int w_stride = int(inp.strides[0][0]);
    texel_idx = (tidx.data[1] * h_stride + w4 * w_stride + c4) * 4 + mod_4(tidx.data[0]);
  }

  // Load packed int32 containing 4 int8 values
  const int packed_input = t_packed_int8_input[texel_idx];

  // Extract the appropriate int8 value based on channel offset within texel
  const int c_offset = mod_4(tidx.data[2]);
  return extract_8bit_from_packed_int_le(packed_input, c_offset);
}

// Load a 4x4 im2col block (4 widths × 4 channels)
ivec4 load_im2col_block(
    const int im2col_w_start,
    const int im2col_h,
    const int k_in_group_start,
    const int group_idx) {
  ivec4 im2col_block;

  for (int r = 0; r < 4; r++) {
    const int im2col_w = im2col_w_start + r;
    ivec4 row_values;
    for (int c = 0; c < 4; c++) {
      const int k_in_group = k_in_group_start + c;

      if (k_in_group >= conv2d_params.logical_K_per_group) {
        row_values[c] = zp;
        continue;
      }

      TensorIndex4D input_tidx =
          get_input_tidx(im2col_w, im2col_h, k_in_group, group_idx);

      row_values[c] = load_input_element(input_tidx, zp);
    }

    im2col_block[r] = pack_into_int32(row_values);
  }
  return im2col_block;
}

void main() {
  const int out_buf_idx = int(gl_GlobalInvocationID.x);

  const ivec4 im2col_sizes = ivec4(im2col_outp.sizes[0]);
  Conv2dBlockExtents im2col_block_extents = make_block_extents(im2col_sizes);

  Conv2dBlockIndex im2col_block_idx = linear_idx_to_block_idx(
      out_buf_idx, im2col_block_extents);

  if (block_idx_out_of_bounds(im2col_block_idx, im2col_block_extents)) {
    return;
  }

  // Convert block index to im2col coordinates
  const int im2col_w = mul_4(im2col_block_idx.data.x);
  const int im2col_h = im2col_block_idx.data.y;
  const int im2col_k = mul_4(im2col_block_idx.data.z);

  // Compute group and k offset within group
  const int group_idx = im2col_k / conv2d_params.K_per_group;
  const int k_in_group = im2col_k % conv2d_params.K_per_group;

  // Load the im2col block using layout-agnostic input access
  Int8OutTile int8_im2col_tile;
  int8_im2col_tile.data[0][0] = load_im2col_block(
      im2col_w, im2col_h, k_in_group, group_idx);

  // Store to output (4W4C format)
  store_packed_int8_output_tile(
      int8_im2col_tile, im2col_block_idx, im2col_block_extents);
}
