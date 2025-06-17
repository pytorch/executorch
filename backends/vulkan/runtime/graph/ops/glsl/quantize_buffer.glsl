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

#define ${MODE}

${define_active_storage_type("buffer")}
${define_required_extensions(IN_DTYPE)}
${define_required_extensions(OUT_DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "t_out", OUT_DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_in", IN_DTYPE, "buffer")}

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

${layout_declare_ubo(B, "int", "out_numel")}
${layout_declare_ubo(B, "ivec4", "t_in_sizes")}
${layout_declare_ubo(B, "ivec4", "t_in_strides")}
${layout_declare_ubo(B, "ivec4", "t_out_sizes")}
${layout_declare_ubo(B, "ivec4", "t_out_strides")}

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}

#include "quantize.glslh"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const lowp ivec4 out_dim_order = unhash_dim_order(out_layout);
const lowp ivec4 in_dim_order = unhash_dim_order(in_layout);

/*
 * QUANTIZATION SHADER (BUFFER STORAGE)
 *
 * This shader converts floating-point tensor values to n-bit integer representations
 * using pre-computed quantization parameters (scale and zero_point). The quantization
 * maps floating-point values to a discrete integer range while preserving the
 * original data distribution as much as possible.
 *
 * ALGORITHM:
 * 1. Load floating-point input value from buffer
 * 2. Apply quantization formula: qvalue = round(value / scale) + zero_point
 * 3. Clamp result to [quant_min, quant_max] range
 * 4. Store quantized integer value to output buffer
 *
 * WORKGROUP CONFIGURATION:
 * - Per-Tensor Mode:
 *   - Global WG Size: {num_elements, 1, 1} (one thread per tensor element)
 *   - Local WG Size: Default (typically {64, 1, 1} or based on global WG size)
 * - Per-Token Mode:
 *   - Global WG Size: {num_elements, 1, 1} (one thread per tensor element)
 *   - Local WG Size: Default (typically {64, 1, 1} or based on global WG size)
 *
 * SUPPORTED CONFIGURATIONS:
 * - Per-Tensor Config: Uses linear buffer indexing with stride-based tensor access
 * - and supports any tensor layout through stride calculations and dimension ordering
 * - Per-Token Config: Assumes width-packed layout (packed_dim = 0)
 * - since that is how token index is calculated
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
 * Quantization Process:
 * Input: 2.5 (float)
 * Step 1: value / scale = 2.5 / 0.1 = 25.0
 * Step 2: round(25.0) + zero_point = 25 + (-128) = -103
 * Step 3: clamp(-103, -128, 127) = -103
 * Output: -103 (int8)
 *
 * PER-TENSOR QUANTIZATION:
 * - Single scale and zero_point values for entire tensor
 * - All elements use same quantization parameters
 * - Parameters passed as push constants for efficiency
 * - Formula: qvalue = clamp(round(value / scale) + zero_point, quant_min, quant_max)
 *
 * PER-TOKEN QUANTIZATION:
 * - Separate scale and zero_point for each token
 * - Token = all elements except last dimension (e.g., for [B,S,H]: B*S tokens of H elements)
 * - Parameters stored in buffer arrays indexed by token_id
 * - Each thread calculates its token_id from tensor coordinates
 * - Formula: qvalue = clamp(round(value / scale[token_id]) + zero_point[token_id], quant_min, quant_max)
 */

#ifdef per_tensor

void quantize_per_tensor() {
  const int out_bufi = int(gl_GlobalInvocationID.x);

  if (out_bufi >= out_numel) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, t_out_strides, out_dim_order);
  const int in_bufi = tidx_to_bufi(out_tidx, t_in_strides);

  IN_T value = t_in[in_bufi];
  OUT_T qvalue = quantize_val(value, scale, zero_point);

  t_out[out_bufi] = qvalue;
}

#else

void quantize_per_token() {
  const int out_bufi = int(gl_GlobalInvocationID.x);

  if (out_bufi >= out_numel) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, t_out_strides, out_dim_order);
  const int in_bufi = tidx_to_bufi(out_tidx, t_in_strides);

  IN_T value = t_in[in_bufi];

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

  OUT_T qvalue = quantize_val(value, t_scale[token_idx], t_zero_point[token_idx]);

  t_out[out_bufi] = qvalue;
}

#endif

void main() {
  quantize_${MODE}();
}
