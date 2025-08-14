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

  layout(push_constant) uniform restrict Block {
    ivec4 blockSize;     // bW, bH, bC, bN
    ivec4 numBlocks;     // tW/bW, tH/bH, tC/bC, tN/bN
    ivec4 blockStride;   // pre-computed linear strides for the block grid
    int quant_min;
    int quant_max;
  };

${layout_declare_ubo(B, "ivec3", "t_in_limits")}
${layout_declare_ubo(B, "ivec3", "t_out_limits")}

#include "indexing_utils.h"
#include "dequantize.glslh"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * DEQUANTIZATION SHADER (TEXTURE STORAGE)
 *
 * This shader converts n-bit integer tensor values back to floating-point representations
 * using pre-computed quantization parameters (scale and zero_point). The dequantization
 * reconstructs the original floating-point values from their discrete integer representations
 * with minimal precision loss.
 *
 * ALGORITHM:
 * 1. Load quantized integer texel (4 values) from 3D texture
 * 2. Apply dequantization formula to each component: value = (qvalue - zero_point) * scale
 * 3. Store reconstructed floating-point texel to output texture
 *
 * WORKGROUP CONFIGURATION:
 * - Per-Tensor Mode:
 *   - Global WG Size: {W, H, C/4} for input size (W, H, C) with width-packing
 *   - Local WG Size: Default (typically {8, 8, 1} or based on global WG size)
 * - Per-Token Mode:
 *   - Global WG Size: {W, H, C/4} for input size (W, H, C) with width-packing
 *   - Local WG Size: Default (typically {8, 8, 1} or based on global WG size)
 *
 * SUPPORTED CONFIGURATIONS:
 * - Texture Storage: Uses 3D texture indexing with texel-based processing
 * - Assumes width-packed layout (packed_dim = 0) for input/output textures
 * - Handles texel padding for non-multiple-of-4 tensor dimensions
 * - For per-token mode: scale/zero_point tensors must use buffer storage
 * - Input/output textures: Must use standard axis mapping for per-token mode
 *
 * DEQUANTIZATION FORMULA VISUALIZATION:
 * For integer range [quant_min, quant_max] mapped back to [min_val, max_val]:
 *
 * Integer Domain:           Floating Point Domain:
 * quant_min ──────────────► min_val
 *    │                         │
 *    │    scale = (max_val - min_val) / (quant_max - quant_min)
 *    │    zero_point = quant_min - round(min_val / scale)
 *    │                         │
 * quant_max ──────────────► max_val
 *
 * Texel Dequantization Process:
 * Input Texel: [-103, -128, -123, -96] (int4)
 * Per-component dequantization with scale=0.1, zero_point=-128:
 * Component 0: (-103 - (-128)) * 0.1 = 25 * 0.1 = 2.5
 * Component 1: (-128 - (-128)) * 0.1 = 0 * 0.1 = 0.0
 * Component 2: (-123 - (-128)) * 0.1 = 5 * 0.1 = 0.5
 * Component 3: (-96 - (-128)) * 0.1 = 32 * 0.1 = 3.2
 * Output Texel: [2.5, 0.0, 0.5, 3.2] (float4)
 *
 * PER-TENSOR DEQUANTIZATION:
 * - Single scale and zero_point values for entire tensor
 * - All texel components use same dequantization parameters
 * - Parameters passed as push constants for efficiency
 * - Each thread processes one texel (4 elements) independently
 * - Formula: value[i] = (qvalue[i] - zero_point) * scale
 *
 * PER-TOKEN DEQUANTIZATION:
 * - Separate scale and zero_point for each token
 * - Token = all elements except last dimension (e.g., for [B,S,H]: B*S tokens of H elements)
 * - Parameters stored in buffer arrays indexed by token_id
 * - Each thread calculates token_id from its 3D texture position
 * - Scale/zero_point buffers accessed directly (not as textures)
 * - Formula: value[i] = (qvalue[i] - zero_point[token_id]) * scale[token_id]
 *
 * Token ID calculation for texel at position (x, y, z):
 * - 3D tensor: token_id = z * texture_height + y
 * - 2D tensor: token_id = y
 * - 1D tensor: token_id = 0
 */

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
    OUT_T value = dequantize_val(qvalue, float(t_scale[0]), int(t_zero_point[0]));

    $if OUT_DTYPE == "double":
      outtex[i] = float(value);
    $else:
      outtex[i] = value;
  }
  write_texel(t_out, pos, outtex);
}

#elif defined(per_token)

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
  float scale_val = float(t_scale[token_idx]);
  int zero_point_val = int(t_zero_point[token_idx]);

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

#elif defined(per_channel)

void dequantize_per_channel() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, t_in_limits))) {
    return;
  }

  IVEC4_T intex = load_texel(t_in, pos);
  FVEC4_T outtex;

  // Calculate channel index based on the dequantization axis (already converted to WHCN)
  // The axis parameter is now in WHCN coordinate system:
  // axis 0 -> W dimension (pos.x)
  // axis 1 -> H dimension (pos.y)
  // axis 2 -> C dimension (pos.z)
  // axis 3 -> N dimension (batch folding in texture storage)

  if (axis == 0) {
    // Width dimension - each texel component has different channel index
    [[unroll]] for (int i = 0; i < 4; ++i) {
      IN_T qvalue = IN_T(intex[i]);
      int channel_idx = pos.x * 4 + i;
      channel_idx = min(channel_idx, num_channels - 1);

      float scale_val = float(t_scale[channel_idx]);
      int zero_point_val = int(t_zero_point[channel_idx]);
      OUT_T value = dequantize_val(qvalue, scale_val, zero_point_val);
      $if OUT_DTYPE == "double":
        outtex[i] = float(value);
      $else:
        outtex[i] = value;
    }
  } else if (axis == 1) {
    int channel_idx = pos.y;
    channel_idx = min(channel_idx, num_channels - 1);
    float scale_val = float(t_scale[channel_idx]);
    int zero_point_val = int(t_zero_point[channel_idx]);

    [[unroll]] for (int i = 0; i < 4; ++i) {
      IN_T qvalue = IN_T(intex[i]);
      OUT_T value = dequantize_val(qvalue, scale_val, zero_point_val);
      $if OUT_DTYPE == "double":
        outtex[i] = float(value);
      $else:
        outtex[i] = value;
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
      IN_T qvalue = IN_T(intex[i]);
      OUT_T value = dequantize_val(qvalue, scale_val, zero_point_val);
      $if OUT_DTYPE == "double":
        outtex[i] = float(value);
      $else:
        outtex[i] = value;
    }
  } else if (axis == 3) {
    // Batch dimension - for 4D tensors, need to account for batch-channel folding
    // The Z coordinate contains folded batch*channel information
    // We need to extract the actual channel index from the folded dimension
    int folded_idx = pos.z;
    // In this case num_channels actually corresponds to the number of channels
    // the C dimension N(C)HW
    int channel_idx = folded_idx / num_channels;

    float scale_val = float(t_scale[channel_idx]);
    int zero_point_val = int(t_zero_point[channel_idx]);

    [[unroll]] for (int i = 0; i < 4; ++i) {
      IN_T qvalue = IN_T(intex[i]);
      OUT_T value = dequantize_val(qvalue, scale_val, zero_point_val);
      $if OUT_DTYPE == "double":
        outtex[i] = float(value);
      $else:
        outtex[i] = value;
    }
  }

  write_texel(t_out, pos, outtex);
}

#else // block_wise

void dequantize_block_wise() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, t_in_limits)))
    return;

  IVEC4_T intex = load_texel(t_in, pos);
  FVEC4_T outtex;

  ivec4 base_tidx = ivec4(pos.x * 4, pos.y, pos.z, 0);
  int foldedZ = pos.z;

  int C_total = numBlocks.z * blockSize.z;

  [[unroll]] for (int i = 0; i < 4; ++i) {
    ivec4 tidx = ivec4(base_tidx.x + i, base_tidx.y, (foldedZ % C_total), (foldedZ / C_total));

    ivec4 bcoord = tidx / blockSize;
    int block_id = bcoord.x * blockStride.x + bcoord.y * blockStride.y + bcoord.z * blockStride.z + bcoord.w * blockStride.w;

    IN_T qvalue = IN_T(intex[i]);
    OUT_T value = dequantize_val(qvalue, float(t_scale[block_id]), int(t_zero_point[block_id]));
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
