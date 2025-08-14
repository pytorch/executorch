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
#define SCALE_T ${buffer_scalar_type(SCALE_DTYPE)}
#define ZP_T ${buffer_scalar_type(ZP_DTYPE)}

#define ${MODE}

${define_active_storage_type("buffer")}
${define_required_extensions(IN_DTYPE)}
${define_required_extensions(OUT_DTYPE)}
${define_required_extensions(SCALE_DTYPE)}
${define_required_extensions(ZP_DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "t_out", OUT_DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_in", IN_DTYPE, "buffer")}

$if MODE == "per_tensor":
  ${layout_declare_tensor(B, "r", "t_scale", SCALE_DTYPE, "buffer")}
  ${layout_declare_tensor(B, "r", "t_zero_point", ZP_DTYPE, "buffer")}

  layout(push_constant) uniform restrict Block {
    int quant_min;
    int quant_max;
  };
$if MODE == "per_token":
  ${layout_declare_tensor(B, "r", "t_scale", SCALE_DTYPE, "buffer")}
  ${layout_declare_tensor(B, "r", "t_zero_point", ZP_DTYPE, "buffer")}

  layout(push_constant) uniform restrict Block {
    int num_tokens;
    int quant_min;
    int quant_max;
  };
$if MODE == "per_channel":
  ${layout_declare_tensor(B, "r", "t_scale", SCALE_DTYPE, "buffer")}
  ${layout_declare_tensor(B, "r", "t_zero_point", ZP_DTYPE, "buffer")}

  layout(push_constant) uniform restrict Block {
    int axis;
    int num_channels;
    int quant_min;
    int quant_max;
  };
$if MODE == "block_wise":
  ${layout_declare_tensor(B, "r", "t_scale", SCALE_DTYPE, "buffer")}
  ${layout_declare_tensor(B, "r", "t_zero_point", ZP_DTYPE, "buffer")}

  layout(push_constant) uniform restrict Block {
    ivec4 blockSize;     // bW, bH, bC, bN
    ivec4 numBlocks;     // tW/bW, tH/bH, tC/bC, tN/bN
    ivec4 blockStride;   // pre-computed linear strides for the block grid
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

#include "dequantize.glslh"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const lowp ivec4 out_dim_order = unhash_dim_order(out_layout);
const lowp ivec4 in_dim_order = unhash_dim_order(in_layout);

/*
  Dequantization Shader (Buffer Storage)
    This shader converts n-bit integer tensor values back to floating-point representations
    using pre-computed quantization parameters (scale and zero_point). The dequantization
    reconstructs the original floating-point values from their discrete integer representations
    with minimal precision loss.

  Important Considerations:
    (+) All input tensors are assumed to be WIDTH_PACKED (i.e., contiguous in the last dimension)
    (+) The axis map layout is assumed to be a standard layout for scales and zero_points
    (++) The scale and zero_point tensors must be implemented as buffers

  Workgroup Configuration:
  - dequantize_per_tensor
      This mode reverses the uniform quantization applied across the entire tensor by using the
      single scale and zero_point values to convert quantized integer values back to their original
      floating-point representation.

    (*) global_wg_size: default
    (*) local_wg_size: default

  - dequantize_per_token
      This mode reverses the quantization applied individually to each token (or element) in the
      input by using separate scale and zero_point values for each token. For a tensor of shape
      [B, S, H], it applies the inverse transformation token-wise across the B*S tokens, converting
      quantized values back to their original floating-point representation for each group of H
      elements independently.

    (*) global_wg_size: default
    (*) local_wg_size: default

  - dequantize_per_channel
      This mode reverses the quantization applied separately to each channel of the input tensor
      by using distinct scale and zero_point values for each channel. For a tensor of shape
      [B, C, H, W] with axis = 1, it applies the inverse transformation channel-wise across the C
      channels, converting quantized values back to their original floating-point representation
      independently for each channel.

    (*) global_wg_size: default
    (*) local_wg_size: default

  - dequantize_block_wise
      This mode reverses the block-wise quantization applied to groups of elements by using separate
      scale and zero_point values for each block. Equivalent to dequantize_affine, it applies the
      inverse affine transformation per block to convert quantized values back to their original
      floating-point representation. For example, if the tensor shape is [6, 9, 4] and
      blockSize = [3, 3, 2], the tensor is divided into 12 blocks, each containing 18 elements,
      and dequantization is performed independently on each block.

    (*) global_wg_size: default
    (*) local_wg_size: default

  Dequantization Formula:
    value = (qvalue - zero_point) * scale
*/

#ifdef per_tensor

void dequantize_per_tensor() {
  const int out_bufi = int(gl_GlobalInvocationID.x);

  if (out_bufi >= out_numel) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, t_out_strides, out_dim_order);
  const int in_bufi = tidx_to_bufi(out_tidx, t_in_strides);

  IN_T qvalue = t_in[in_bufi];
  OUT_T value = dequantize_val(qvalue, float(t_scale[0]), int(t_zero_point[0]));

  t_out[out_bufi] = value;
}

#elif defined(per_token)

void dequantize_per_token() {
  const int out_bufi = int(gl_GlobalInvocationID.x);

  if (out_bufi >= out_numel) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, t_out_strides, out_dim_order);
  const int in_bufi = tidx_to_bufi(out_tidx, t_in_strides);

  IN_T qvalue = t_in[in_bufi];

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

  OUT_T value = dequantize_val(qvalue, float(t_scale[token_idx]), int(t_zero_point[token_idx]));

  t_out[out_bufi] = value;
}

#elif defined(per_channel)

void dequantize_per_channel() {
  const int out_bufi = int(gl_GlobalInvocationID.x);

  if (out_bufi >= out_numel) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, t_out_strides, out_dim_order);
  const int in_bufi = tidx_to_bufi(out_tidx, t_in_strides);

  IN_T qvalue = t_in[in_bufi];

  // Calculate channel index based on the dequantization axis (already converted to WHCN)
  // The axis parameter is now in WHCN coordinate system:
  // axis 0 -> W dimension (tidx.x)
  // axis 1 -> H dimension (tidx.y)
  // axis 2 -> C dimension (tidx.z)
  // axis 3 -> N dimension (tidx.w)
  int channel_idx = 0;

  if (axis == 0) {
    channel_idx = out_tidx.x;
  } else if (axis == 1) {
    channel_idx = out_tidx.y;
  } else if (axis == 2) {
    channel_idx = out_tidx.z;
  } else if (axis == 3) {
    channel_idx = out_tidx.w;
  }

  channel_idx = min(channel_idx, num_channels - 1);

  OUT_T value = dequantize_val(qvalue, float(t_scale[channel_idx]), int(t_zero_point[channel_idx]));

  t_out[out_bufi] = value;
}

#else // block_wise

void dequantize_block_wise() {
  const int out_bufi = int(gl_GlobalInvocationID.x);

  if (out_bufi >= out_numel) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, t_out_strides, out_dim_order);
  const int in_bufi = tidx_to_bufi(out_tidx, t_in_strides);

  IN_T qvalue = t_in[in_bufi];

  const ivec4 bcoord = out_tidx / blockSize;

  const int block_id = bcoord.x * blockStride.x + bcoord.y * blockStride.y + bcoord.z * blockStride.z + bcoord.w * blockStride.w;

  const OUT_T value = dequantize_val(qvalue, float(t_scale[block_id]), int(t_zero_point[block_id]));

  t_out[out_bufi] = value;
}

#endif

void main() {
  dequantize_${MODE}();
}
