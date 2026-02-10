/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", DTYPE)}

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product : require

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

// Load weight block for a given (ic4, kx, ky, oc4) position.
// Weight texture layout (from pack_q8_conv2d_weights.glsl):
//   block_x = oc4 * K_w + kx
//   block_y = ky * IC4 + ic4
// Each texel ivec4 has 4 components (4 output channels), each component is
// a packed int32 containing 4 int8 values for 4 consecutive input channels.
ivec4 load_weight_block(int ic4, int kx, int ky, int oc4, int IC4, int KW) {
  const int block_x = oc4 * KW + kx;
  const int block_y = ky * IC4 + ic4;
  return texelFetch(t_packed_int8_weight, ivec2(block_x, block_y), 0);
}

ivec4 quantize(const vec4 texel, const float inv_scale, const int zp) {
  vec4 quantized = round(texel * inv_scale) + zp;
  return clamp(ivec4(quantized), -128, 127);
}

void main() {
  // Thread mapping
  int oc4 = int(gl_GlobalInvocationID.z);
  int w4 = int(gl_GlobalInvocationID.x);

  // Initialize output tensor index (WHCN order)
  // Each thread handles 4 adjacent widths starting at base_out_w
  TensorIndex4D outp_tidx;
  outp_tidx.data[0] = w4 * 4;
  outp_tidx.data[1] = int(gl_GlobalInvocationID.y);
  outp_tidx.data[2] = oc4 * 4;
  outp_tidx.data[3] = 0;

  const int W = int(outp.sizes[0][0]);
  const int OC = int(outp.sizes[0][2]);
  const int OC4 = int(div_up_4(OC));

  // Bounds check
  if (any(greaterThanEqual(outp_tidx.data, ivec4(outp.sizes[0])))) {
    return;
  }

  // Input dimensions
  const int inp_W = int(inp.sizes[0][0]);
  const int inp_H = int(inp.sizes[0][1]);
  const int IC = int(inp.sizes[0][2]);

  // Compute channels per group
  const int OC_per_group = OC / conv2d_params.groups;
  const int IC_per_group = IC / conv2d_params.groups;
  const int IC4_per_group = div_up_4(IC_per_group);

  // Determine which group this output channel block belongs to
  const int group_idx = outp_tidx.data[2] / OC_per_group;
  const int ic_group_start = group_idx * IC_per_group;

  // Get strides for efficient indexing
  const int inp_w_stride = int(inp.strides[0][0]);
  const int inp_h_stride = int(inp.strides[0][1]);
  const int inp_c_stride = int(inp.strides[0][2]);
  const int w_texel_step = conv2d_params.dilation.x * inp_w_stride;
  const int h_texel_step = conv2d_params.dilation.y * inp_h_stride;
  const int subtile_w_step = conv2d_params.stride.x * inp_w_stride;

  // Compute base input position (for subtile_w=0, ic4=0)
  TensorIndex4D inp_tidx;
  inp_tidx.data[0] = outp_tidx.data[0] * conv2d_params.stride.x - conv2d_params.padding.x;
  inp_tidx.data[1] = outp_tidx.data[1] * conv2d_params.stride.y - conv2d_params.padding.y;
  inp_tidx.data[2] = ic_group_start;
  inp_tidx.data[3] = 0;

  int base_inp_texel_idx;
  if (get_outer_packed_dim_block_size(inp_layout) == 1) {
    base_inp_texel_idx = tensor4d_idx_to_texel_idx(inp, inp_tidx, inp_layout);
  }

  // Store base width to reset at beginning of each loop
  const int base_inp_w = inp_tidx.data[0];

  // Create packed input zero point (4 copies of input_zp packed into int32)
  const int input_zp_packed = pack_into_int32(ivec4(input_zp));

  // Initialize accumulators for 4 width positions × 4 output channels each
  ivec4 acc[4];
  [[unroll]] for (int i = 0; i < 4; ++i) {
    acc[i] = ivec4(0);
  }

  // Perform convolution using packed int8 dot products
  for (int ky = 0; ky < conv2d_params.kernel_size.y; ky++) {
    const bool h_in_bounds = (inp_tidx.data[1] >= 0 && inp_tidx.data[1] < inp_H);

    // Process input channels in blocks of 4
    for (int ic4 = 0; ic4 < IC4_per_group; ic4++) {
      // Input channel index for this block (base channel of the 4-channel block)
      inp_tidx.data[2] = ic_group_start + ic4 * 4;

      // Reset width coordinate at start of each ic4 iteration
      inp_tidx.data[0] = base_inp_w;

      for (int kx = 0; kx < conv2d_params.kernel_size.x; kx++) {
        // Load weight block: 4 output channels × 4 input channels
        // weight_block[oc] contains packed weights for ic4*4 to ic4*4+3 -> oc
        const ivec4 weight_block = load_weight_block(ic4, kx, ky, oc4, IC4_per_group, conv2d_params.kernel_size.x);

        // Process 4 adjacent width positions
        [[unroll]] for (int subtile_w = 0; subtile_w < 4; ++subtile_w) {
          // Load packed input (4 consecutive channels packed into one int32)
          // Use input_zp_packed for out-of-bounds positions
          int packed_input = input_zp_packed;
          if (h_in_bounds && inp_tidx.data[0] >= 0 && inp_tidx.data[0] < inp_W) {
            // Compute input texel index using base + offsets
            int inp_texel_idx;
            if (get_outer_packed_dim_block_size(inp_layout) == 1) {
              inp_texel_idx = base_inp_texel_idx + ic4 * inp_c_stride + kx * w_texel_step + subtile_w * subtile_w_step;
            } else {
              // inp_texel_idx = tensor4d_idx_to_texel_idx(inp, inp_tidx, inp_layout);
              const int w4 = div_4(inp_tidx.data[0]);
              const int inp_c4 = div_4(inp_tidx.data[2]);
              inp_texel_idx = (inp_tidx.data[1] * inp_h_stride + w4 * inp_w_stride + inp_c4) * 4 + mod_4(inp_tidx.data[0]);
            }
            packed_input = t_packed_int8_input[inp_texel_idx];
          }

          // Accumulate using packed int8 dot product for each output channel
          // dotPacked4x8AccSatEXT computes: acc + dot(unpack(a), unpack(b))
          [[unroll]] for (int oc_offset = 0; oc_offset < 4; ++oc_offset) {
            acc[subtile_w][oc_offset] = dotPacked4x8AccSatEXT(
                packed_input,
                weight_block[oc_offset],
                acc[subtile_w][oc_offset]);
          }

          // Advance to next output position's input coordinate
          inp_tidx.data[0] += conv2d_params.stride.x;
        }

        // Adjust for net dilation step
        inp_tidx.data[0] += conv2d_params.dilation.x - 4 * conv2d_params.stride.x;
      }
    }

    // Advance height by dilation for next kernel row
    inp_tidx.data[1] += conv2d_params.dilation.y;

    if (get_outer_packed_dim_block_size(inp_layout) == 1) {
      // Advance base index by height step for next kernel row
      base_inp_texel_idx += h_texel_step;
    }
  }

  // Apply input zero point correction via weight_sums
  const vec4 weight_sums = vec4(t_weight_sums[oc4]);
  const vec4 weight_scales = vec4(t_weight_scales[oc4]);

  // Convert to float, apply dequantization, and optionally add bias
  vec4 facc[4];
  [[unroll]] for (int subtile_w = 0; subtile_w < 4; ++subtile_w) {
    facc[subtile_w] = vec4(acc[subtile_w]);
    facc[subtile_w] -= weight_sums * input_zp;
    facc[subtile_w] *= weight_scales * input_scale;
  }

  // Apply bias if enabled
  if (apply_bias > 0) {
    const vec4 bias = vec4(t_bias[oc4]);
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
      outp_texel_idx = base_outp_texel_idx + subtile_w;
    }

    t_packed_int8_output[outp_texel_idx] = packed_out;

    outp_tidx.data[0] += 1;
  }
}
