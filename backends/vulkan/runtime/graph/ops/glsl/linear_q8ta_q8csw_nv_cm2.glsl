/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Quantized int8 linear shader using GL_NV_cooperative_matrix2 extension.
 *
 * Uses float16 cooperative matrices with dequantization during load.
 *
 * Computes: output = dequant(input) @ dequant(weight)^T + bias
 * where dequant(input) = (input - zero_point) * scale
 * and dequant(weight) = weight * weight_scale[channel]
 */

#version 450 core

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_cooperative_matrix : enable
#extension GL_NV_cooperative_matrix2 : enable
#extension GL_EXT_buffer_reference : enable

${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${buffer_gvec_type(DTYPE, 4)}

// Block sizes for cooperative matrix - 16x16x16
#define BM 16
#define BN 16
#define BK 16

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_output", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input", "int8", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight", "int8", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight_sums", "int", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=True)}

${layout_declare_spec_const(C, "int", "apply_bias", "0")}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(push_constant) uniform restrict Block {
  float input_scale;
  int input_zp;
};

// Workgroup size: 32 threads (1 subgroup for Subgroup scope)
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// Matrix types using float16 with float16 accumulator
#define MAT_TYPE float16_t
#define ACC_TYPE float16_t

// Input block size: 16 int8 values packed as ivec4 (4 ints × 4 int8 per int)
#define INPUT_BLOCK_K 16

// Buffer reference type for input dequantization decode functor
// Matches the packed structure from q8ta_quantize.glsl: ivec4 with 4 int8 per int
layout(buffer_reference, std430, buffer_reference_align = 4) buffer decodeInputBuf {
  ivec4 packed;  // 4 ints, each containing 4 packed int8 values = 16 int8 total
};

// Input decode: unpack int8 from ivec4, dequantize as (val - zp) * scale
MAT_TYPE decodeInputFunc(const in decodeInputBuf bl, const in uint blockCoords[2], const in uint coordInBlock[2]) {
  uint idx = coordInBlock[1];
  int packed_int = bl.packed[idx >> 2];
  int8_t val = int8_t((packed_int >> (int(idx & 3) * 8)) & 0xFF);
  return MAT_TYPE((float(val) - float(input_zp)) * input_scale);
}

// Weight decode: dequantize as val * per-channel scale
layout(buffer_reference, std430, buffer_reference_align = 1) buffer decodeWeightBuf {
  int8_t v;
};

MAT_TYPE decodeWeightFunc(const in decodeWeightBuf bl, const in uint blockCoords[2], const in uint coordInBlock[2]) {
  uint out_channel = blockCoords[0] + coordInBlock[0];
  return MAT_TYPE(float(bl.v) * float(t_weight_scales[out_channel]));
}

void main() {
  // Get dimensions
  const uint M = uint(output_sizes.y);  // batch/rows
  const uint N = uint(output_sizes.x);  // output features
  const uint K = uint(input_sizes.x);   // input features

  // Each workgroup handles one BM x BN tile
  const uint ir = gl_WorkGroupID.x;  // row tile index
  const uint ic = gl_WorkGroupID.y;  // column tile index

  // Early exit if out of bounds
  if (ir * BM >= M || ic * BN >= N) {
    return;
  }

  // Create tensor layouts with clamping for boundary handling
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tensorLayoutA =
      createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tensorLayoutB =
      createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tensorLayoutD =
      createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);

  // Create transpose view for loading weights
  tensorViewNV<2, false, 1, 0> tensorViewTranspose = createTensorViewNV(2, false, 1, 0);

  // Set dimensions and strides for input (A matrix)
  // Block size is 16 int8 values packed as ivec4, so stride is in blocks (K / INPUT_BLOCK_K)
  tensorLayoutA = setTensorLayoutDimensionNV(tensorLayoutA, M, K);
  tensorLayoutA = setTensorLayoutStrideNV(tensorLayoutA, K / INPUT_BLOCK_K, 1);
  tensorLayoutA = setTensorLayoutBlockSizeNV(tensorLayoutA, 1, INPUT_BLOCK_K);

  tensorLayoutB = setTensorLayoutDimensionNV(tensorLayoutB, N, K);
  tensorLayoutB = setTensorLayoutStrideNV(tensorLayoutB, K, 1);

  tensorLayoutD = setTensorLayoutDimensionNV(tensorLayoutD, M, N);
  tensorLayoutD = setTensorLayoutStrideNV(tensorLayoutD, N, 1);

  // Initialize accumulator with bias (broadcast across rows) or zeros
  coopmat<ACC_TYPE, gl_ScopeSubgroup, BM, BN, gl_MatrixUseAccumulator> sum;

  if (apply_bias == 1) {
    // Bias layout: stride 0 in row dim broadcasts same bias values across rows
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tensorLayoutBias =
        createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tensorLayoutBias = setTensorLayoutDimensionNV(tensorLayoutBias, BM, N);
    tensorLayoutBias = setTensorLayoutStrideNV(tensorLayoutBias, 0, 1);

    // Load as T first then convert to ACC_TYPE (buffer stores T, not ACC_TYPE)
    coopmat<T, gl_ScopeSubgroup, BM, BN, gl_MatrixUseAccumulator> bias_tmp;
    coopMatLoadTensorNV(bias_tmp, t_bias, 0,
        sliceTensorLayoutNV(tensorLayoutBias, 0, BM, ic * BN, BN));
    sum = coopmat<ACC_TYPE, gl_ScopeSubgroup, BM, BN, gl_MatrixUseAccumulator>(bias_tmp);
  } else {
    sum = coopmat<ACC_TYPE, gl_ScopeSubgroup, BM, BN, gl_MatrixUseAccumulator>(ACC_TYPE(0.0f));
  }

  // Loop over K dimension
  const uint k_iters = (K + BK - 1) / BK;

  [[dont_unroll]]
  for (uint block_k = 0, i = 0; i < k_iters; block_k += BK, ++i) {
    // Cooperative matrices for A and B
    coopmat<MAT_TYPE, gl_ScopeSubgroup, BM, BK, gl_MatrixUseA> mat_a;
    coopmat<MAT_TYPE, gl_ScopeSubgroup, BK, BN, gl_MatrixUseB> mat_b;

    // Load A tile with input dequantization
    coopMatLoadTensorNV(mat_a, t_input, 0,
        sliceTensorLayoutNV(tensorLayoutA, ir * BM, BM, block_k, BK), decodeInputFunc);

    // Load B tile with transpose and weight dequantization
    coopMatLoadTensorNV(mat_b, t_weight, 0,
        sliceTensorLayoutNV(tensorLayoutB, ic * BN, BN, block_k, BK), tensorViewTranspose, decodeWeightFunc);

    // Multiply and accumulate
    sum = coopMatMulAdd(mat_a, mat_b, sum);
  }

  // Convert accumulator to output type
  coopmat<T, gl_ScopeSubgroup, BM, BN, gl_MatrixUseAccumulator> result =
      coopmat<T, gl_ScopeSubgroup, BM, BN, gl_MatrixUseAccumulator>(sum);

  // Store result
  coopMatStoreTensorNV(result, t_output, 0,
      sliceTensorLayoutNV(tensorLayoutD, ir * BM, BM, ic * BN, BN));
}
