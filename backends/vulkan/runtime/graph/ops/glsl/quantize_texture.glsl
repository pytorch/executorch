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
#define FVEC4_T ${texel_load_type(IN_DTYPE, "texture3d")}

#define OUT_T ${buffer_scalar_type(OUT_DTYPE)}
#define IVEC4_T ${texel_load_type(OUT_DTYPE, "texture3d")}
#define SCALE_T ${buffer_scalar_type(SCALE_DTYPE)}
#define ZP_T ${buffer_scalar_type(ZP_DTYPE)}

#define ${MODE}

${define_active_storage_type("texture3d")}
${define_required_extensions(IN_DTYPE)}
${define_required_extensions(OUT_DTYPE)}
${define_required_extensions(SCALE_DTYPE)}
${define_required_extensions(ZP_DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "t_out", OUT_DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", IN_DTYPE, "texture3d")}

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

  layout(push_constant) uniform restrict BlockPC {
    ivec4 blockSize;        // WHCN
    ivec4 numBlocks;        // (#W,#H,#C,#N)
    ivec4 blockStride;      // {1, #W, #W * #H, #W * #H * #C}
    int   quant_min;
    int   quant_max;
  };

${layout_declare_ubo(B, "ivec3", "t_in_limits")}
${layout_declare_ubo(B, "ivec3", "t_out_limits")}

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}

#include "quantize.glslh"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
  Quantization Shader (Texture Storage)
    This shader converts floating-point tensor values to n-bit integer representations
    using pre-computed quantization parameters (scale and zero_point). The quantization
    maps floating-point values to a discrete integer range while preserving the original
    data distribution as much as possible.

  Important Considerations:
    (+) All input tensors are assumed to be WIDTH_PACKED (i.e., contiguous in the last dimension)
    (+) The axis map layout is assumed to be a standard layout for scales and zero_points
    (++) The scale and zero_point tensors must be implemented as buffers

  Workgroup Configuration:
  - quantize_per_tensor
      This mode applies uniform quantization across the entire tensor using a single scale
      and zero_point value.

    (*) global_wg_size: default
    (*) local_wg_size: default

  - quantize_per_token
      This mode applies quantization individually to each token (or element) in the input,
      using separate scale and zero_point values for each token. For instance if we have
      a tensor of shape [B, S, H] then we have B*S tokens (and s+zp pairs) of H elements each.

    (*) global_wg_size: default
    (*) local_wg_size: default

  - quantize_per_channel
      This mode applies quantization separately to each channel of the input tensor, using
      distinct scale and zero_point values for each channel. For example, if the tensor shape
      is [B, C, H, W] and axis = 1, quantization parameters are computed per channel C, allowing
      each channel to be quantized independently.

    (*) global_wg_size: default
    (*) local_wg_size: Default with special handling for batch dimension. When quantizing along
        the batch axis, Z dimension is set to 1 to ensure correct workgroup dispatching. Otherwise,
        uses standard workgroup size derived from global workgroup dimensions.

  - quantize_block_wise
      This mode applies quantization in blocks or groups of elements, allowing different scale
      and zero_point values for each block. It is equivalent to quantize_affine, where quantization
      parameters are affine transformations applied per block. For example, if the tensor shape
      is [6, 9, 4] and blockSize = [3, 3, 2], then we have 12 blocks each with 18 elements.

    (*) global_wg_size: default
    (*) local_wg_size: Default with special handling for batch dimension. When quantizing along
        the batch axis, Z dimension is set to 1 to ensure correct workgroup dispatching. Otherwise,
        uses standard workgroup size derived from global workgroup dimensions.

  Quantization Formula:
    qvalue = clamp(round(value / scale) + zero_point, quant_min, quant_max).
*/

#ifdef per_tensor

void quantize_per_tensor() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, t_in_limits))) {
    return;
  }

  FVEC4_T intex = load_texel(t_in, pos);
  IVEC4_T outtex;

  [[unroll]] for (int i = 0; i < 4; ++i) {
    IN_T value = IN_T(intex[i]);
    OUT_T qvalue = quantize_val(value, float(t_scale[0]), int(t_zero_point[0]));
    outtex[i] = qvalue;
  }
  write_texel(t_out, pos, outtex);
}

#elif defined(per_token)

void quantize_per_token() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, t_in_limits))) {
    return;
  }

  FVEC4_T intex = load_texel(t_in, pos);

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
  float scale_val = float(t_scale[token_idx]);
  int zero_point_val = int(t_zero_point[token_idx]);

  IVEC4_T outtex;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    IN_T value = IN_T(intex[i]);
    OUT_T qvalue = quantize_val(value, scale_val, zero_point_val);
    outtex[i] = qvalue;
  }

  write_texel(t_out, pos, outtex);
}

#elif defined(per_channel)

void quantize_per_channel() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, t_in_limits))) {
    return;
  }

  FVEC4_T intex = load_texel(t_in, pos);
  IVEC4_T outtex;

  // Calculate channel index based on the quantization axis (already converted to WHCN)
  // The axis parameter is now in WHCN coordinate system:
  // axis 0 -> W dimension (pos.x for texture, but width-packed so pos.x * 4 + component)
  // axis 1 -> H dimension (pos.y)
  // axis 2 -> C dimension (pos.z / C), but for 4D tensors this includes batch-channel folding
  // axis 3 -> N dimension (pos.z / N), but for 4D tensors this includes batch-channel folding

  if (axis == 0) {
    // Width dimension - each texel component has different channel index
    [[unroll]] for (int i = 0; i < 4; ++i) {
      IN_T value = IN_T(intex[i]);
      int channel_idx = pos.x * 4 + i;
      channel_idx = min(channel_idx, num_channels - 1);

      float scale_val = float(t_scale[channel_idx]);
      int zero_point_val = int(t_zero_point[channel_idx]);
      OUT_T qvalue = quantize_val(value, scale_val, zero_point_val);
      outtex[i] = qvalue;
    }
  } else if (axis == 1) {
    // Height dimension - all texel components use same channel index
    int channel_idx = pos.y;
    channel_idx = min(channel_idx, num_channels - 1);
    float scale_val = float(t_scale[channel_idx]);
    int zero_point_val = int(t_zero_point[channel_idx]);

    [[unroll]] for (int i = 0; i < 4; ++i) {
      IN_T value = IN_T(intex[i]);
      OUT_T qvalue = quantize_val(value, scale_val, zero_point_val);
      outtex[i] = qvalue;
    }
  } else if (axis == 2) {
    // Channel dimension - for 4D tensors, need to account for batch-channel folding
    // The Z coordinate contains folded batch*channel information
    // We need to extract the actual channel index from the folded dimension
    int folded_idx = pos.z;
    int channel_idx = folded_idx % num_channels;

    float scale_val = float(t_scale[channel_idx]);
    int zero_point_val = int(t_zero_point[channel_idx]);

    [[unroll]] for (int i = 0; i < 4; ++i) {
      IN_T value = IN_T(intex[i]);
      OUT_T qvalue = quantize_val(value, scale_val, zero_point_val);
      outtex[i] = qvalue;
    }
  } else if (axis == 3) {
    // Batch dimension - for 4D tensors, need to account for batch-channel folding
    // The Z coordinate contains folded batch*channel information
    // We need to extract the actual batch index from the folded dimension
    int folded_idx = pos.z;
    int batch_idx = folded_idx / num_channels;

    float scale_val = float(t_scale[batch_idx]);
    int zero_point_val = int(t_zero_point[batch_idx]);

    [[unroll]] for (int i = 0; i < 4; ++i) {
      IN_T value = IN_T(intex[i]);
      OUT_T qvalue = quantize_val(value, scale_val, zero_point_val);
      outtex[i] = qvalue;
    }
  }

  write_texel(t_out, pos, outtex);
}

#else // block_wise

void quantize_block_wise() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, t_in_limits)))
    return;

  FVEC4_T intex = load_texel(t_in, pos);
  IVEC4_T outtex;

  ivec4 base_tidx = ivec4(pos.x * 4, pos.y, pos.z, 0);
  int foldedZ = pos.z;

  int C_total = numBlocks.z * blockSize.z;

  [[unroll]] for (int i = 0; i < 4; ++i) {
    ivec4 tidx = ivec4(base_tidx.x + i, base_tidx.y, (foldedZ % C_total), (foldedZ / C_total));

    ivec4 bcoord = tidx / blockSize;
    int block_id = bcoord.x * blockStride.x + bcoord.y * blockStride.y + bcoord.z * blockStride.z + bcoord.w * blockStride.w;

    IN_T value = IN_T(intex[i]);
    OUT_T qvalue = quantize_val(value, float(t_scale[block_id]), int(t_zero_point[block_id]));
    outtex[i] = qvalue;
  }

  write_texel(t_out, pos, outtex);
}

#endif

void main() {
  quantize_${MODE}();
}
