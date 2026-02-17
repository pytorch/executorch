/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Floating-point matrix multiplication shader using GL_NV_cooperative_matrix2
 * extension for optimized performance on NVIDIA GPUs with tensor cores.
 *
 * Based on ggml's mul_mm_cm2.comp implementation.
 *
 * RTX 4080 supported configuration:
 *   - Scope: Subgroup (NOT Workgroup!)
 *   - A, B, C types: all float16
 *   - Tile sizes: M=16, N=16, K=16
 *
 * Computes: output = input @ weight^T + bias
 */

#version 450 core

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_cooperative_matrix : enable
#extension GL_NV_cooperative_matrix2 : enable

${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${buffer_gvec_type(DTYPE, 4)}

#define TILE_ROWS ${TILE_ROWS}
#define TILE_COLS ${TILE_COLS}

// Block sizes for cooperative matrix - matching RTX 4080 supported config
#define BM 16
#define BN 16
#define BK 16

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_output", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

// Workgroup size: 1 threads (subgroup/warp size for Subgroup scope)
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

${layout_declare_spec_const(C, "int", "apply_bias", "0")}

// Matrix type for float16 computation with SUBGROUP scope (RTX 4080 supported)

#define MAT_TYPE float16_t
#define ACC_TYPE float16_t

void main() {
  // Get tile indices
  const uint M = uint(output_sizes.y);  // batch/rows
  const uint N = uint(output_sizes.x);  // output features
  const uint K = uint(input_sizes.x);   // input features

  // Calculate N4 for prepacked weight stride
  const uint N4 = (N + 3) / 4;
  const uint weight_stride = N4 * 4;

  const uint blocks_m = (M + BM - 1) / BM;
  const uint ir = gl_WorkGroupID.x % blocks_m;  // row tile index
  const uint ic = gl_WorkGroupID.y;             // column tile index

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

  // Set dimensions and strides
  // Input A: M x K, row-major (stride = K)
  tensorLayoutA = setTensorLayoutDimensionNV(tensorLayoutA, M, K);
  tensorLayoutA = setTensorLayoutStrideNV(tensorLayoutA, K, 1);

  // Weight B: K x N (prepacked), row-major
  // Each row k has N elements: weight[k, 0:N]
  tensorLayoutB = setTensorLayoutDimensionNV(tensorLayoutB, K, N);
  tensorLayoutB = setTensorLayoutStrideNV(tensorLayoutB, N, 1);

  // Output D: M x N, row-major (stride = N)
  // Output layout matches expected: [batch, out_features]
  tensorLayoutD = setTensorLayoutDimensionNV(tensorLayoutD, M, N);
  tensorLayoutD = setTensorLayoutStrideNV(tensorLayoutD, N, 1);

  // Transpose view for B matrix (weight is stored transposed)
  tensorViewNV<2, false, 1, 0> tensorViewTranspose = createTensorViewNV(2, false, 1, 0);

  // Initialize accumulator - either with zeros or with bias (broadcast across rows)
  coopmat<ACC_TYPE, gl_ScopeSubgroup, BM, BN, gl_MatrixUseAccumulator> sum;

  if (apply_bias != 0) {
    // Create tensor layout for bias with broadcast (stride 0 in row dimension)
    // This makes all rows read the same bias values
    // Bias is 1D array of size N, we want to load it as BM x BN matrix
    // where each row has the same values: bias[ic*BN], bias[ic*BN+1], ..., bias[ic*BN+BN-1]
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tensorLayoutBias =
        createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);

    // Dimension: BM rows x N columns (full bias width)
    // Stride: 0 for rows (broadcast same values), 1 for columns
    tensorLayoutBias = setTensorLayoutDimensionNV(tensorLayoutBias, BM, N);
    tensorLayoutBias = setTensorLayoutStrideNV(tensorLayoutBias, 0, 1);  // stride 0 = broadcast rows

    // Load bias into accumulator (slice the column range for this tile)
    coopMatLoadTensorNV(sum, t_bias, 0,
        sliceTensorLayoutNV(tensorLayoutBias, 0, BM, ic * BN, BN));
  } else {
    // Initialize to zeros
    sum = coopmat<ACC_TYPE, gl_ScopeSubgroup, BM, BN, gl_MatrixUseAccumulator>(ACC_TYPE(0.0));
  }

  // Loop over K dimension
  const uint k_iters = (K + BK - 1) / BK;

  [[dont_unroll]]
  for (uint block_k = 0, i = 0; i < k_iters; block_k += BK, ++i) {
    // Use SUBGROUP scope for cooperative matrices (RTX 4080 supported)
    coopmat<MAT_TYPE, gl_ScopeSubgroup, BM, BK, gl_MatrixUseA> mat_a;
    coopmat<MAT_TYPE, gl_ScopeSubgroup, BK, BN, gl_MatrixUseB> mat_b;

    // Load A tile: input[ir*BM : ir*BM+BM, block_k : block_k+BK]
    coopMatLoadTensorNV(mat_a, t_input, 0,
        sliceTensorLayoutNV(tensorLayoutA, ir * BM, BM, block_k, BK));

    // Load B tile: weight[block_k : block_k+BK, ic*BN : ic*BN+BN]
    // Weight is prepacked in [K, N] layout, load directly without transpose
    coopMatLoadTensorNV(mat_b, t_weight, 0,
        sliceTensorLayoutNV(tensorLayoutB, block_k, BK, ic * BN, BN));

    // Multiply and accumulate
    sum = coopMatMulAdd(mat_a, mat_b, sum);
  }

  // Store result directly without transpose (row-major output)
  coopMatStoreTensorNV(sum, t_output, 0,
      sliceTensorLayoutNV(tensorLayoutD, ir * BM, BM, ic * BN, BN));
}
