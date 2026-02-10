/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, "buffer")}
#define T ${texel_load_component_type(DTYPE, "buffer")}

${define_active_storage_type("buffer")}

layout(std430) buffer;

#include "indexing.glslh"
#include "common.glslh"
#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_output", "int", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_packed_int8_input", "int", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_packed_int8_weight", "int", "texture2d", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

// Metadata for input/output tensors (memory layout agnostic)
${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}
${layout_declare_ubo(B, "Conv2DParams", "conv2d_params")}

layout(push_constant) uniform restrict Block {
  float input_scale;
  int input_zp;
  float output_inv_scale;
  int output_zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "1")}

// Layout specialization constants
${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}

#include "block_indexing.glslh"

// Load a 4xint8 block of weights.
// Weights are stored in 4W4C format: [kH, kW/4, C/4, 4, 4] where the first 4 is
// the outer (kW) dimension and the second 4 is the inner (channel) dimension.
// Returns packed int32 containing 4 int8 values for channels c to c+3.
int load_weight(int kw, int kh, int c4, int KW4, int C4) {
  // Find the packed block index (4W4C tiling)
  const int kw4 = kw / 4;  // W block
  const int block_x_offset = kw % 4;
  // Texture layout: x = c4, y = kh * KW4 + kw4
  return texelFetch(t_packed_int8_weight, ivec2(c4, kh * KW4 + kw4), 0)[block_x_offset];
}

ivec4 quantize(const vec4 texel, const float inv_scale, const int zp) {
  vec4 quantized = round(texel * inv_scale) + zp;
  return clamp(ivec4(quantized), -128, 127);
}

void main() {
  const int c4 = int(gl_GlobalInvocationID.z);

  // Initialize output tensor index (WHCN order)
  // Each thread handles 4 adjacent widths starting at base_out_w
  TensorIndex4D outp_tidx;
  outp_tidx.data[0] = int(gl_GlobalInvocationID.x) * 4;
  outp_tidx.data[1] = int(gl_GlobalInvocationID.y);
  outp_tidx.data[2] = c4 * 4;
  outp_tidx.data[3] = 0;

  const int W = int(outp.sizes[0][0]);
  const int C4 = int(div_up_4(outp.sizes[0][2]));

  // Bounds check
  if (any(greaterThanEqual(outp_tidx.data, ivec4(outp.sizes[0])))) {
    return;
  }

  // Compute weight addressing constants
  const int KW4 = int(div_up_4(conv2d_params.kernel_size.x));

  // Get strides for width and height dimensions (in texel space)
  const int w_stride = int(inp.strides[0][0]);
  const int h_stride = int(inp.strides[0][1]);

  // Pre-compute step sizes for efficient indexing
  const int w_texel_step = conv2d_params.dilation.x * w_stride;
  const int h_texel_step = conv2d_params.dilation.y * h_stride;
  // Step between adjacent output width positions in input texel space
  const int subtile_w_step = conv2d_params.stride.x * w_stride;

  // Compute base input position for subtile_w=0
  TensorIndex4D inp_tidx;
  inp_tidx.data[0] = outp_tidx.data[0] * conv2d_params.stride.x - conv2d_params.padding.x;
  inp_tidx.data[1] = outp_tidx.data[1] * conv2d_params.stride.y - conv2d_params.padding.y;
  inp_tidx.data[2] = outp_tidx.data[2];
  inp_tidx.data[3] = 0;  // batch = 0 since N == 1

  int base_inp_texel_idx;
  if (get_outer_packed_dim_block_size(inp_layout) == 1) {
    base_inp_texel_idx = tensor4d_idx_to_texel_idx(inp, inp_tidx, inp_layout);
  }

  // Store the base width position to reset the index position at the beginning
  // of each loop
  const int base_inp_w = inp_tidx.data[0];

  // Initialize accumulators for 4 width positions × 4 channels each
  ivec4 acc[4];
  [[unroll]] for (int i = 0; i < 4; ++i) {
    acc[i] = ivec4(0);
  }

  // Input dimensions for bounds checking
  const int inp_W = int(inp.sizes[0][0]);
  const int inp_H = int(inp.sizes[0][1]);

  // Perform depthwise convolution
  for (int ky = 0; ky < conv2d_params.kernel_size.y; ky++) {
    const bool h_in_bounds = (inp_tidx.data[1] >= 0 && inp_tidx.data[1] < inp_H);

    // Reset width coordinate at start of each kernel row
    inp_tidx.data[0] = base_inp_w;

    for (int kx = 0; kx < conv2d_params.kernel_size.x; kx++) {
      // Load weight once, reuse for all 4 width positions
      const int packed_weight = load_weight(kx, ky, c4, KW4, C4);
      const ivec4 weight_4c = unpack_int8x4(packed_weight);

      // Process 4 adjacent width positions using stride offsets
      [[unroll]] for (int subtile_w = 0; subtile_w < 4; ++subtile_w) {
        ivec4 input_4c = ivec4(input_zp);
        if (h_in_bounds && inp_tidx.data[0] >= 0 && inp_tidx.data[0] < inp_W) {
          // Compute texel index: base + kernel offset + subtile offset
          int inp_texel_idx;
          if (get_outer_packed_dim_block_size(inp_layout) == 1) {
            inp_texel_idx = base_inp_texel_idx + kx * w_texel_step + subtile_w * subtile_w_step;
          } else {
            // const int w_offset = kx * conv2d_params.dilation.x + subtile_w * conv2d_params.stride.x;
            // inp_texel_idx = base_inp_texel_idx + div_4(w_offset) * w_stride + mod_4(w_offset);
            // inp_texel_idx = tensor4d_idx_to_texel_idx(inp, inp_tidx, inp_layout);
            const int w4 = div_4(inp_tidx.data[0]);
            inp_texel_idx = (inp_tidx.data[1] * h_stride + w4 * w_stride + c4) * 4 + mod_4(inp_tidx.data[0]);
          }
          const int packed_input = t_packed_int8_input[inp_texel_idx];
          input_4c = unpack_int8x4(packed_input);
        }

        // Accumulate: element-wise multiply for depthwise conv
        acc[subtile_w] += weight_4c * input_4c;

        // Advance to next output position's input coordinate
        inp_tidx.data[0] += conv2d_params.stride.x;
      }

      // We advanced by 4*stride.x during subtile loop; adjust for net dilation step
      inp_tidx.data[0] += conv2d_params.dilation.x - 4 * conv2d_params.stride.x;
    }

    // Advance height by dilation for next kernel row
    inp_tidx.data[1] += conv2d_params.dilation.y;

    if (get_outer_packed_dim_block_size(inp_layout) == 1) {
      // Advance base index by height step for next kernel row
      base_inp_texel_idx += h_texel_step;
    }
  }

  // Apply input zero point as weight_sum * input_zp
  const vec4 weight_sums = vec4(t_weight_sums[c4]);
  const vec4 weight_scales = vec4(t_weight_scales[c4]);

  // Convert to float, apply dequantization, and optionally add bias
  vec4 facc[4];
  [[unroll]] for (int subtile_w = 0; subtile_w < 4; ++subtile_w) {
    facc[subtile_w] = vec4(acc[subtile_w]);
    facc[subtile_w] -= weight_sums * input_zp;
    facc[subtile_w] *= weight_scales * input_scale;
  }

  // Apply bias if enabled
  if (apply_bias > 0) {
    const vec4 bias = vec4(t_bias[c4]);
    [[unroll]] for (int subtile_w = 0; subtile_w < 4; ++subtile_w) {
      facc[subtile_w] += bias;
    }
  }

  // Compute base output texel index (for subtile_w=0)
  const int base_outp_texel_idx = tensor4d_idx_to_texel_idx(outp, outp_tidx, outp_layout);
  const int out_w_stride = int(outp.strides[0][0]);

  // Quantize and store outputs using stride offsets
  [[unroll]] for (int subtile_w = 0; subtile_w < 4; ++subtile_w) {
    // Skip out-of-bounds width positions
    if (outp_tidx.data[0] >= W) {
      continue;
    }

    const ivec4 quantized_out = quantize(facc[subtile_w], output_inv_scale, output_zp);
    const int packed_out = pack_into_int32(quantized_out);

    // Store using stride offset from base
    int outp_texel_idx;
    if (get_outer_packed_dim_block_size(outp_layout) == 1) {
      outp_texel_idx = base_outp_texel_idx + subtile_w * out_w_stride;
    } else {
      // outp_texel_idx = tensor4d_idx_to_texel_idx(outp, outp_tidx, outp_layout);
      outp_texel_idx = base_outp_texel_idx + subtile_w;
    }

    t_packed_int8_output[outp_texel_idx] = packed_out;

    outp_tidx.data[0] += 1;
  }
}
