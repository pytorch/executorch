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

#define ${MODE}

${define_active_storage_type("texture3d")}
${define_required_extensions(IN_DTYPE)}
${define_required_extensions(OUT_DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", OUT_DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", IN_DTYPE, "texture3d")}

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

${layout_declare_ubo(B, "ivec3", "t_in_limits")}
${layout_declare_ubo(B, "ivec3", "t_out_limits")}

#include "indexing_utils.h"
#include "quantize.glslh"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * QUANTIZATION SHADER (TEXTURE STORAGE)
 *
 * This shader converts floating-point tensor values to n-bit integer representations
 * using pre-computed quantization parameters (scale and zero_point). The quantization
 * maps floating-point values to a discrete integer range while preserving the
 * original data distribution as much as possible.
 *
 * ALGORITHM:
 * 1. Load floating-point texel (4 values) from 3D texture
 * 2. Apply quantization formula to each component: qvalue = round(value / scale) + zero_point
 * 3. Clamp each result to [quant_min, quant_max] range
 * 4. Store quantized integer texel to output texture
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
 * - Assumes width-packed layout (packed_dim = 0) in current implementation
 * - Handles texel padding for non-multiple-of-4 tensor dimensions
 * - For per-token mode: scale/zero_point tensors must use buffer storage
 *
 * QUANTIZATION FORMULA VISUALIZATION:
 * For input range [min_val, max_val] mapped to integer range [quant_min, quant_max]:
 *
 * Floating Point Domain:    Integer Domain:
 * min_val ────────────────► quant_min
 *    │                         │
 *    │    scale = (max_val - min_val) / (quant_max - quant_min)
 *    │    zero_point = quant_min - round(min_val / scale)
 *    │                         │
 * max_val ────────────────► quant_max
 *
 * Texel Quantization Process:
 * Input Texel: [2.5, -1.0, 0.5, 3.2] (float4)
 * Per-component quantization with scale=0.1, zero_point=-128:
 * Component 0: round(2.5 / 0.1) + (-128) = 25 + (-128) = -103
 * Component 1: round(-1.0 / 0.1) + (-128) = -10 + (-128) = -138 → clamp to -128
 * Component 2: round(0.5 / 0.1) + (-128) = 5 + (-128) = -123
 * Component 3: round(3.2 / 0.1) + (-128) = 32 + (-128) = -96
 * Output Texel: [-103, -128, -123, -96] (int4)
 *
 * PER-TENSOR QUANTIZATION:
 * - Single scale and zero_point values for entire tensor
 * - All texel components use same quantization parameters
 * - Parameters passed as push constants for efficiency
 * - Each thread processes one texel (4 elements) independently
 * - Formula: qvalue[i] = clamp(round(value[i] / scale) + zero_point, quant_min, quant_max)
 *
 * PER-TOKEN QUANTIZATION:
 * - Separate scale and zero_point for each token
 * - Token = all elements except last dimension (e.g., for [B,S,H]: B*S tokens of H elements)
 * - Parameters stored in buffer arrays indexed by token_id
 * - Each thread calculates token_id from its 3D texture position
 * - Scale/zero_point buffers accessed directly (not as textures)
 * - Formula: qvalue[i] = clamp(round(value[i] / scale[token_id]) + zero_point[token_id], quant_min, quant_max)
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
    OUT_T qvalue = quantize_val(value, scale, zero_point);
    outtex[i] = qvalue;
  }
  write_texel(t_out, pos, outtex);
}

#else

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
  float scale_val = t_scale[token_idx];
  int zero_point_val = t_zero_point[token_idx];

  IVEC4_T outtex;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    IN_T value = IN_T(intex[i]);
    OUT_T qvalue = quantize_val(value, scale_val, zero_point_val);
    outtex[i] = qvalue;
  }

  write_texel(t_out, pos, outtex);
}

#endif

void main() {
  quantize_${MODE}();
}
