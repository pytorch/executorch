/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include "test_utils.h"

#include <cassert>

at::Tensor unpack_weights_4x2(const at::Tensor& weights_4x2) {
  std::vector<int64_t> weights_shape(weights_4x2.sizes().vec());
  weights_shape[1] *= 2;

  at::Tensor weights_unpacked =
      at::empty(weights_shape, at::device(at::kCPU).dtype(at::kInt));

  const int64_t N = weights_unpacked.size(0);
  const int64_t K = weights_unpacked.size(1);

  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k += 2) {
      const uint8_t packed_val = weights_4x2[n][k / 2].item().to<uint8_t>();
      const uint8_t second_val = packed_val & 0x0F;
      const uint8_t first_val = (packed_val & 0xF0) >> 4;

      weights_unpacked[n][k] = int(first_val);
      weights_unpacked[n][k + 1] = int(second_val);
    }
  }

  return weights_unpacked;
}

at::Tensor linear_qta8a_qga4w_qta8o_4bit_dequant_impl(
    const at::Tensor& quantized_input,
    const at::Tensor& input_scale,
    const at::Tensor& input_zero_point,
    const at::Tensor& weights_4x2,
    const int64_t group_size,
    const at::Tensor& weight_scales_and_zeros,
    const at::Tensor& output_scale,
    const at::Tensor& output_zero_point) {
  // Calculate number of input tokens
  int64_t input_num_tokens = 1;
  for (size_t i = 0; i < quantized_input.sizes().size() - 1; i++) {
    input_num_tokens *= quantized_input.size(i);
  }

  // Manually dequantize the char tensor using per-token quantization
  at::Tensor x_float = at::zeros_like(quantized_input, at::kFloat);

  // Apply per-token dequantization
  auto input_accessor = quantized_input.accessor<int8_t, 3>();
  auto output_accessor = x_float.accessor<float, 3>();

  for (int64_t token_idx = 0; token_idx < input_num_tokens; token_idx++) {
    float scale_val = input_scale[token_idx].item<float>();
    int zero_point_val = input_zero_point[token_idx].item<int>();

    // Calculate batch and sequence indices for this token
    int64_t b = token_idx / quantized_input.size(1);
    int64_t m = token_idx % quantized_input.size(1);

    // Apply dequantization for all features in this token
    for (int64_t k = 0; k < quantized_input.size(-1); k++) {
      float dequant_val =
          (input_accessor[b][m][k] - zero_point_val) * scale_val;
      output_accessor[b][m][k] = dequant_val;
    }
  }

  std::vector<int64_t> weights_shape(weights_4x2.sizes().vec());
  weights_shape[1] *= 2;

  at::Tensor weights_dequantized =
      at::empty(weights_shape, at::device(at::kCPU).dtype(at::kFloat));

  const int64_t N = weights_dequantized.size(0);
  const int64_t K = weights_dequantized.size(1);

  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k += 2) {
      const int group_idx = k / group_size;
      const uint8_t packed_val = weights_4x2[n][k / 2].item().to<uint8_t>();
      const uint8_t second_val = packed_val & 0x0F;
      const uint8_t first_val = (packed_val & 0xF0) >> 4;

      const float scale =
          weight_scales_and_zeros[group_idx][n][0].item().to<float>();
      const float zero =
          weight_scales_and_zeros[group_idx][n][1].item().to<float>();

      weights_dequantized[n][k] = (float(first_val) - 8.0) * scale + zero;
      weights_dequantized[n][k + 1] = (float(second_val) - 8.0) * scale + zero;
    }
  }

  at::Tensor linear_result = at::linear(x_float, weights_dequantized);

  // Calculate number of output tokens
  int64_t output_num_tokens = 1;
  for (size_t i = 0; i < linear_result.sizes().size() - 1; i++) {
    output_num_tokens *= linear_result.size(i);
  }

  // Quantize the result manually using per-token quantization
  at::Tensor quantized_result = at::zeros_like(linear_result, at::kChar);

  // Apply per-token quantization
  auto linear_accessor = linear_result.accessor<float, 3>();
  auto quant_accessor = quantized_result.accessor<int8_t, 3>();

  for (int64_t token_idx = 0; token_idx < output_num_tokens; token_idx++) {
    float scale_val = output_scale[token_idx].item<float>();
    int zero_point_val = output_zero_point[token_idx].item<int>();

    // Calculate batch and sequence indices for this token
    int64_t b = token_idx / linear_result.size(1);
    int64_t m = token_idx % linear_result.size(1);

    // Apply quantization for all features in this token
    for (int64_t n = 0; n < linear_result.size(-1); n++) {
      float quant_val =
          std::round(linear_accessor[b][m][n] / scale_val) + zero_point_val;
      quant_val = std::clamp(quant_val, -128.0f, 127.0f);
      quant_accessor[b][m][n] = static_cast<int8_t>(quant_val);
    }
  }

  return quantized_result;
}
