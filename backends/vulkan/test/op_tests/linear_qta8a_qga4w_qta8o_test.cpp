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

void test_vulkan_linear_qta8a_qga4w_qta8o_impl(
    const int B,
    const int M,
    const int K,
    const int N,
    const int group_size = 8,
    const vkcompute::utils::StorageType in_storage =
        vkcompute::utils::kTexture3D,
    const vkcompute::utils::StorageType out_storage =
        vkcompute::utils::kTexture3D) {
  assert(K % group_size == 0);

  // Create per-token quantization parameters for input
  const int64_t input_num_tokens = B * M;
  at::Tensor input_scale =
      at::rand({input_num_tokens}, at::device(at::kCPU).dtype(at::kFloat)) *
          0.1f +
      0.05f; // Range [0.05, 0.15]
  at::Tensor input_zero_point = at::randint(
      -10, 10, {input_num_tokens}, at::device(at::kCPU).dtype(at::kInt));

  at::Tensor float_x =
      at::rand({B, M, K}, at::device(at::kCPU).dtype(at::kFloat));

  // Create a reference quantized tensor using per-token quantization
  // Mimic per-token quantization using at::quantize_per_channel by reshaping to
  // [num_tokens, features]
  at::Tensor float_x_reshaped = float_x.view({input_num_tokens, K});
  at::Tensor qx_ref_reshaped = at::quantize_per_channel(
      float_x_reshaped,
      input_scale.to(at::kDouble),
      input_zero_point.to(at::kLong),
      0, // axis 0 for per-token (first dimension after reshape)
      c10::ScalarType::QInt8);

  // Convert back to int8 tensor and reshape to original shape
  at::Tensor x =
      at::int_repr(qx_ref_reshaped).view(float_x.sizes()).to(at::kChar);

  at::Tensor weights_4x2 =
      at::randint(0, 256, {N, K / 2}, at::device(at::kCPU).dtype(at::kByte));

  const int k_groups = K / group_size;
  at::Tensor scales_and_zeros =
      at::rand({k_groups, N, 2}, at::device(at::kCPU).dtype(at::kFloat));

  // Create per-token quantization parameters for output
  const int64_t output_num_tokens = B * M;
  at::Tensor output_scale =
      at::rand({output_num_tokens}, at::device(at::kCPU).dtype(at::kFloat)) *
          0.1f +
      0.1f; // Range [0.1, 0.2]
  at::Tensor output_zero_point = at::randint(
      -10, 10, {output_num_tokens}, at::device(at::kCPU).dtype(at::kInt));

  at::Tensor out_ref = linear_qta8a_qga4w_qta8o_4bit_dequant_impl(
      x,
      input_scale,
      input_zero_point,
      weights_4x2,
      group_size,
      scales_and_zeros,
      output_scale,
      output_zero_point);

  // Build Vulkan graph
  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

#define MAKE_TENSORREF_FOR(x)              \
  ValueRef r_##x = graph.add_tensorref(    \
      x.sizes().vec(),                     \
      from_at_scalartype(x.scalar_type()), \
      x.const_data_ptr());

  MAKE_TENSORREF_FOR(weights_4x2);
  MAKE_TENSORREF_FOR(scales_and_zeros);

  IOValueRef r_x = graph.add_input_tensor(
      x.sizes().vec(), from_at_scalartype(x.scalar_type()), in_storage);

  IOValueRef r_input_scale = graph.add_input_tensor(
      input_scale.sizes().vec(),
      from_at_scalartype(input_scale.scalar_type()),
      utils::kBuffer);

  IOValueRef r_input_zero_point = graph.add_input_tensor(
      input_zero_point.sizes().vec(),
      from_at_scalartype(input_zero_point.scalar_type()),
      utils::kBuffer);

  IOValueRef r_output_scale = graph.add_input_tensor(
      output_scale.sizes().vec(),
      from_at_scalartype(output_scale.scalar_type()),
      utils::kBuffer);

  IOValueRef r_output_zero_point = graph.add_input_tensor(
      output_zero_point.sizes().vec(),
      from_at_scalartype(output_zero_point.scalar_type()),
      utils::kBuffer);

  const ValueRef r_out = graph.add_tensor(
      out_ref.sizes().vec(),
      from_at_scalartype(out_ref.scalar_type()),
      out_storage);

  VK_GET_OP_FN("et_vk.linear_qta8a_qga4w_qta8o.default")
  (graph,
   {r_x.value,
    r_input_scale.value,
    r_input_zero_point.value,
    r_weights_4x2,
    graph.add_scalar<int64_t>(group_size),
    r_scales_and_zeros,
    r_output_scale.value,
    r_output_zero_point.value,
    r_out});

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.encode_prepack();
  graph.prepack();
  graph.encode_execute();

  //
  // Run model
  //

  graph.propagate_resize();
  graph.copy_into_staging(r_x.staging, x.const_data_ptr(), x.numel());
  graph.copy_into_staging(
      r_input_scale.staging, input_scale.const_data_ptr(), input_scale.numel());
  graph.copy_into_staging(
      r_input_zero_point.staging,
      input_zero_point.const_data_ptr(),
      input_zero_point.numel());
  graph.copy_into_staging(
      r_output_scale.staging,
      output_scale.const_data_ptr(),
      output_scale.numel());
  graph.copy_into_staging(
      r_output_zero_point.staging,
      output_zero_point.const_data_ptr(),
      output_zero_point.numel());

  graph.execute();

  at::Tensor vk_out = at::empty_like(out_ref);
  graph.copy_from_staging(
      staging_out, vk_out.mutable_data_ptr(), vk_out.numel());

  // For quantized int8 operations, allow for 1-unit differences due to rounding
  bool is_close = at::allclose(vk_out, out_ref, 1.0, 1.0);

  at::Tensor weights_int = unpack_weights_4x2(weights_4x2);

  if (!is_close) {
    std::cout << "out_ref: \n" << out_ref << std::endl;
    std::cout << "vk_out: \n" << vk_out << std::endl;
  }

  ASSERT_TRUE(is_close);
}

void test_vulkan_linear_qta8a_qga4w_qta8o(
    const int B,
    const int M,
    const int K,
    const int N,
    const int group_size = 32) {
  test_vulkan_linear_qta8a_qga4w_qta8o_impl(
      B,
      M,
      K,
      N,
      group_size,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  test_vulkan_linear_qta8a_qga4w_qta8o_impl(
      B,
      M,
      K,
      N,
      group_size,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

TEST(
    VulkanLinearQTA8AQGA4WQTA8OTest,
    test_vulkan_linear_quant_gemm_custom_groupsize_one) {
  test_vulkan_linear_qta8a_qga4w_qta8o(
      /*B = */ 1,
      /*M = */ 2,
      /*K = */ 8,
      /*N = */ 8,
      /*group_size = */ 8);
}

TEST(
    VulkanLinearQTA8AQGA4WQTA8OTest,
    test_vulkan_linear_quant_gemm_custom_groupsize_two) {
  test_vulkan_linear_qta8a_qga4w_qta8o(
      /*B = */ 1,
      /*M = */ 2,
      /*K = */ 16,
      /*N = */ 8,
      /*group_size = */ 8);
}

TEST(VulkanLinearQTA8AQGA4WQTA8OTest, test_vulkan_linear_quant_gemm_case_one) {
  test_vulkan_linear_qta8a_qga4w_qta8o(
      /*B = */ 1,
      /*M = */ 4,
      /*K = */ 64,
      /*N = */ 32);
}

TEST(VulkanLinearQTA8AQGA4WQTA8OTest, test_vulkan_linear_quant_gemm_case_two) {
  test_vulkan_linear_qta8a_qga4w_qta8o(
      /*B = */ 1,
      /*M = */ 4,
      /*K = */ 128,
      /*N = */ 32);
}

TEST(
    VulkanLinearQTA8AQGA4WQTA8OTest,
    test_vulkan_linear_quant_gemm_case_three) {
  test_vulkan_linear_qta8a_qga4w_qta8o(
      /*B = */ 1,
      /*M = */ 8,
      /*K = */ 64,
      /*N = */ 16);
}

TEST(VulkanLinearQTA8AQGA4WQTA8OTest, test_vulkan_linear_quant_gemm_case_four) {
  test_vulkan_linear_qta8a_qga4w_qta8o(
      /*B = */ 1,
      /*M = */ 256,
      /*K = */ 256,
      /*N = */ 256);
}

TEST(
    VulkanLinearQTA8AQGA4WQTA8OTest,
    test_vulkan_linear_quant_gemv_custom_groupsize_one) {
  test_vulkan_linear_qta8a_qga4w_qta8o(
      /*B = */ 1,
      /*M = */ 2,
      /*K = */ 16,
      /*N = */ 8,
      /*group_size = */ 8);
}

TEST(VulkanLinearQTA8AQGA4WQTA8OTest, test_vulkan_linear_quant_gemv_case_one) {
  test_vulkan_linear_qta8a_qga4w_qta8o(
      /*B = */ 1,
      /*M = */ 1,
      /*K = */ 256,
      /*N = */ 256);
}

TEST(VulkanLinearQTA8AQGA4WQTA8OTest, test_vulkan_linear_quant_gemv_case_two) {
  test_vulkan_linear_qta8a_qga4w_qta8o(
      /*B = */ 1,
      /*M = */ 1,
      /*K = */ 32,
      /*N = */ 32);
}

TEST(
    VulkanLinearQTA8AQGA4WQTA8OTest,
    test_vulkan_linear_quant_gemv_case_three) {
  test_vulkan_linear_qta8a_qga4w_qta8o(
      /*B = */ 1,
      /*M = */ 1,
      /*K = */ 64,
      /*N = */ 16);
}
