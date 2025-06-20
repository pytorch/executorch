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

#include "dequantize.glslh"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const lowp ivec4 out_dim_order = unhash_dim_order(out_layout);
const lowp ivec4 in_dim_order = unhash_dim_order(in_layout);

/*
 * DEQUANTIZATION SHADER (BUFFER STORAGE)
 *
 * This shader converts n-bit integer tensor values back to floating-point representations
 * using pre-computed quantization parameters (scale and zero_point). The dequantization
 * reconstructs the original floating-point values from their discrete integer representations
 * with minimal precision loss.
 *
 * ALGORITHM:
 * 1. Load quantized integer value from buffer
 * 2. Apply dequantization formula: value = (qvalue - zero_point) * scale
 * 3. Store reconstructed floating-point value to output buffer
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
 * - Buffer Storage: Uses linear buffer indexing with stride-based tensor access
 * - Per-Tensor: Supports any tensor layout through stride calculations and dimension ordering
 * - Per-Token: Supports only width packed tensors (packed_dim = 0) and standard axis mapping
 * - Scale/zero_point tensors: Must use buffer storage with width packing (packed_dim = 0)
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
 * Dequantization Process:
 * Input: -103 (int8)
 * Step 1: qvalue - zero_point = -103 - (-128) = 25
 * Step 2: result * scale = 25 * 0.1 = 2.5
 * Output: 2.5 (float)
 *
 * PER-TENSOR DEQUANTIZATION:
 * - Single scale and zero_point values for entire tensor
 * - All elements use same dequantization parameters
 * - Parameters passed as push constants for efficiency
 * - Formula: value = (qvalue - zero_point) * scale
 *
 * PER-TOKEN DEQUANTIZATION:
 * - Separate scale and zero_point for each token
 * - Token = all elements except last dimension (e.g., for [B,S,H]: B*S tokens of H elements)
 * - Parameters stored in buffer arrays indexed by token_id
 * - Each thread calculates its token_id from tensor coordinates
 * - Formula: value = (qvalue - zero_point[token_id]) * scale[token_id]
 *
 * Token ID calculation for element at tensor index (w, z, y, x):
 * - 4D tensor: token_id = w * (sizes.z * sizes.y) + z * sizes.y + y
 * - 3D tensor: token_id = z * sizes.y + y
 * - 2D tensor: token_id = y
 * - 1D tensor: token_id = 0
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
  OUT_T value = dequantize_val(qvalue, scale, zero_point);

  t_out[out_bufi] = value;
}

#else

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

  OUT_T value = dequantize_val(qvalue, t_scale[token_idx], t_zero_point[token_idx]);

  t_out[out_bufi] = value;
}

#endif

void main() {
  dequantize_${MODE}();
}
