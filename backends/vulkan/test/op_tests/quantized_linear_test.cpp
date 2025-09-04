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

class VulkanLinearQCS4WTest : public ::testing::Test {
 public:
  void SetUp() override {
    if (!vkcompute::api::context()
             ->adapter_ptr()
             ->supports_int16_shader_types()) {
      GTEST_SKIP();
    }
  }

  void TearDown() override {
    // Clean up any resources if needed
  }
};

class VulkanLinearQTA8AQGA4WTest : public ::testing::Test {
 public:
  void SetUp() override {
    if (!vkcompute::api::context()
             ->adapter_ptr()
             ->has_full_int8_buffers_support()) {
      GTEST_SKIP();
    }
  }

  void TearDown() override {
    // Clean up any resources if needed
  }
};

//
// Reference Implementations
//

at::Tensor linear_qga4w_reference_impl(
    const at::Tensor& x,
    const at::Tensor& weights_4x2,
    const int64_t groupsize,
    const at::Tensor& scales_and_zeros,
    const int64_t inner_k_tiles) {
  const std::vector<int64_t> original_x_size(x.sizes().vec());
  const size_t ndim = original_x_size.size();
  const int64_t out_features = weights_4x2.size(0);
  const at::Tensor x_flattened = x.reshape({-1, original_x_size[ndim - 1]});
  at::Tensor out = at::_weight_int4pack_mm_for_cpu(
      x_flattened, weights_4x2, groupsize, scales_and_zeros);
  std::vector<int64_t> out_shape(
      original_x_size.begin(), original_x_size.end());
  out_shape.at(ndim - 1) = out_features;
  return out.reshape(out_shape);
}

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

at::Tensor dequantize_and_linear_qga4w(
    const at::Tensor& x,
    const at::Tensor& weights_4x2,
    const int64_t groupsize,
    const at::Tensor& scales_and_zeros,
    const int64_t inner_k_tiles) {
  std::vector<int64_t> weights_shape(weights_4x2.sizes().vec());
  weights_shape[1] *= 2;

  at::Tensor weights_dequantized =
      at::empty(weights_shape, at::device(at::kCPU).dtype(at::kFloat));

  const int64_t N = weights_dequantized.size(0);
  const int64_t K = weights_dequantized.size(1);

  const int k_groups = K / groupsize;
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k += 2) {
      const int group_idx = k / groupsize;
      // const int scale_idx = k_groups * n + group_idx;
      const uint8_t packed_val = weights_4x2[n][k / 2].item().to<uint8_t>();
      const uint8_t second_val = packed_val & 0x0F;
      const uint8_t first_val = (packed_val & 0xF0) >> 4;

      const float scale = scales_and_zeros[group_idx][n][0].item().to<float>();
      const float zero = scales_and_zeros[group_idx][n][1].item().to<float>();

      weights_dequantized[n][k] = (float(first_val) - 8.0) * scale + zero;
      weights_dequantized[n][k + 1] = (float(second_val) - 8.0) * scale + zero;
    }
  }

  return at::linear(x, weights_dequantized);
}

at::Tensor dequantize_and_linear_qcs4w(
    const at::Tensor& x,
    const at::Tensor& weights_4x2,
    const at::Tensor& scales) {
  std::vector<int64_t> weights_shape(weights_4x2.sizes().vec());
  weights_shape[1] *= 2;

  at::Tensor weights_dequantized =
      at::empty(weights_shape, at::device(at::kCPU).dtype(at::kFloat));

  const int64_t N = weights_dequantized.size(0);
  const int64_t K = weights_dequantized.size(1);

  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k += 2) {
      // const int scale_idx = k_groups * n + group_idx;
      const uint8_t packed_val = weights_4x2[n][k / 2].item().to<uint8_t>();
      const uint8_t second_val = packed_val & 0x0F;
      const uint8_t first_val = (packed_val & 0xF0) >> 4;

      const float scale = scales[n].item().to<float>();

      weights_dequantized[n][k] = (float(first_val) - 8.0) * scale;
      weights_dequantized[n][k + 1] = (float(second_val) - 8.0) * scale;
    }
  }

  return at::linear(x, weights_dequantized);
}

at::Tensor linear_qcs4w_reference_impl(
    const at::Tensor& x,
    const at::Tensor& weights_4x2,
    const at::Tensor& scales) {
  const std::vector<int64_t> original_x_size(x.sizes().vec());
  const size_t ndim = original_x_size.size();
  const int64_t out_features = weights_4x2.size(0);
  const at::Tensor x_flattened = x.reshape({-1, original_x_size[ndim - 1]});

  const at::Tensor weights_unpacked =
      (unpack_weights_4x2(weights_4x2) - 8).to(at::kChar);
  at::Tensor out =
      at::_weight_int8pack_mm(x_flattened, weights_unpacked, scales);

  std::vector<int64_t> out_shape(
      original_x_size.begin(), original_x_size.end());
  out_shape.at(ndim - 1) = out_features;
  return out.reshape(out_shape);
}

at::Tensor linear_qta8a_qga4w_quantized_matmul(
    const at::Tensor& quantized_input, // [B, M, K] int8 quantized input
    const at::Tensor& input_scale, // [B*M] per-token input scales
    const at::Tensor& input_zero_point, // [B*M] per-token input zero points
    const at::Tensor& weights_4x2, // [N, K/2] 4-bit packed weights
    const int64_t group_size, // Group size for weight quantization
    const at::Tensor& weight_scales, // [K/group_size, N] weight scales
    const at::Tensor& weight_zeros) { // [K/group_size, N] weight zeros

  const int64_t B = quantized_input.size(0);
  const int64_t M = quantized_input.size(1);
  const int64_t K = quantized_input.size(2);
  const int64_t N = weights_4x2.size(0);

  // Create output tensor for floating point results
  at::Tensor float_output =
      at::zeros({B, M, N}, at::device(at::kCPU).dtype(at::kFloat));

  // Accessors for efficient access
  auto input_accessor = quantized_input.accessor<int8_t, 3>();
  auto output_accessor = float_output.accessor<float, 3>();
  auto weights_accessor = weights_4x2.accessor<uint8_t, 2>();
  auto weight_scales_accessor = weight_scales.accessor<float, 2>();
  auto weight_zeros_accessor = weight_zeros.accessor<int32_t, 2>();
  auto input_scale_accessor = input_scale.accessor<float, 1>();
  auto input_zero_accessor = input_zero_point.accessor<int32_t, 1>();

  // Perform quantized matrix multiplication following quantization.md equation
  // (5): result_real_value = lhs_scale * rhs_scale * Sum_over_k(
  //   (lhs_quantized_value[k] - lhs_zero_point) *
  //   (rhs_quantized_value[k] - rhs_zero_point)
  // )
  for (int64_t b = 0; b < B; b++) {
    for (int64_t m = 0; m < M; m++) {
      const int64_t token_idx = b * M + m;
      const float lhs_scale =
          input_scale_accessor[token_idx]; // Per-token input scale
      const int32_t lhs_zero_point =
          input_zero_accessor[token_idx]; // Per-token input zero point

      for (int64_t n = 0; n < N; n++) {
        float result_real_value = 0.0f;

        for (int64_t k = 0; k < K; k++) {
          // Get per-group weight quantization parameters
          const int64_t group_idx = k / group_size;
          const float rhs_scale =
              weight_scales_accessor[group_idx][n]; // Per-group weight scale
          const int32_t rhs_zero_point =
              weight_zeros_accessor[group_idx]
                                   [n]; // Per-group weight zero point

          // Unpack the 4-bit weight for this position
          const uint8_t packed_val = weights_accessor[n][k / 2];
          uint8_t weight_4bit;
          if (k % 2 == 0) {
            weight_4bit = (packed_val & 0xF0) >> 4; // First weight in pair
          } else {
            weight_4bit = packed_val & 0x0F; // Second weight in pair
          }

          // Get quantized values
          const int32_t lhs_quantized_value =
              static_cast<int32_t>(input_accessor[b][m][k]);
          // Convert 4-bit weight to signed: subtract 8 to get range [-8, 7]
          const int32_t rhs_quantized_value =
              static_cast<int32_t>(weight_4bit) - 8;

          // Apply proper quantization paradigm from quantization.md equation
          // (3): real_value = scale * (quantized_value - zero_point) Following
          // equation (5): result = lhs_scale * rhs_scale *
          //   (lhs_quantized - lhs_zero) * (rhs_quantized - rhs_zero)
          const float lhs_diff =
              static_cast<float>(lhs_quantized_value - lhs_zero_point);
          const float rhs_diff =
              static_cast<float>(rhs_quantized_value - rhs_zero_point);

          result_real_value += lhs_scale * rhs_scale * lhs_diff * rhs_diff;
        }

        output_accessor[b][m][n] = result_real_value;
      }
    }
  }

  return float_output;
}

at::Tensor linear_qta8a_qga4w_4bit_dequant_impl(
    const at::Tensor& quantized_input,
    const at::Tensor& input_scale,
    const at::Tensor& input_zero_point,
    const at::Tensor& weights_4x2,
    const int64_t group_size,
    const at::Tensor& weight_scales,
    const at::Tensor& weight_zeros) {
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

      const float scale = weight_scales[group_idx][n].item().to<float>();
      const int zero = weight_zeros[group_idx][n].item().to<int>();

      weights_dequantized[n][k] =
          ((float(first_val) - 8.0) - float(zero)) * scale;
      weights_dequantized[n][k + 1] =
          ((float(second_val) - 8.0) - float(zero)) * scale;
    }
  }

  at::Tensor linear_result = at::linear(x_float, weights_dequantized);

  return linear_result;
}

//
// Test functions
//

void test_reference_linear_qga4w(
    const int B,
    const int M,
    const int K,
    const int N,
    const int group_size = 32,
    const int inner_k_tiles = 8) {
  assert(K % group_size == 0);

  at::Tensor x = at::rand({B, M, K}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weights_4x2 =
      at::randint(0, 256, {N, K / 2}, at::device(at::kCPU).dtype(at::kByte));
  at::Tensor weights_int = unpack_weights_4x2(weights_4x2);

  const int k_groups = K / group_size;
  at::Tensor scales_and_zeros =
      at::rand({k_groups, N, 2}, at::device(at::kCPU).dtype(at::kFloat));

  at::Tensor out = linear_qga4w_reference_impl(
      x,
      at::_convert_weight_to_int4pack_for_cpu(weights_int, group_size),
      group_size,
      scales_and_zeros,
      inner_k_tiles);

  at::Tensor out_ref = dequantize_and_linear_qga4w(
      x, weights_4x2, group_size, scales_and_zeros, inner_k_tiles);

  ASSERT_TRUE(at::allclose(out, out_ref));
}

void test_reference_linear_qcs4w(
    const int B,
    const int M,
    const int K,
    const int N) {
  at::Tensor x = at::rand({B, M, K}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weights_4x2 =
      at::randint(0, 256, {N, K / 2}, at::device(at::kCPU).dtype(at::kByte));
  at::Tensor weights_int = unpack_weights_4x2(weights_4x2);

  at::Tensor scales = at::rand({N}, at::device(at::kCPU).dtype(at::kFloat));

  at::Tensor out = linear_qcs4w_reference_impl(x, weights_4x2, scales);

  at::Tensor out_ref = dequantize_and_linear_qcs4w(x, weights_4x2, scales);

  ASSERT_TRUE(at::allclose(out, out_ref));
}

void test_vulkan_linear_qga4w_impl(
    const int B,
    const int M,
    const int K,
    const int N,
    const int group_size = 32,
    const int inner_k_tiles = 8,
    const vkcompute::utils::StorageType in_storage =
        vkcompute::utils::kTexture3D,
    const vkcompute::utils::StorageType out_storage =
        vkcompute::utils::kTexture3D) {
  assert(K % group_size == 0);

  at::Tensor x = at::rand({B, M, K}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weights_4x2 =
      at::randint(0, 256, {N, K / 2}, at::device(at::kCPU).dtype(at::kByte));

  const int k_groups = K / group_size;
  at::Tensor scales_and_zeros =
      at::rand({k_groups, N, 2}, at::device(at::kCPU).dtype(at::kFloat));

  at::Tensor weights_int = unpack_weights_4x2(weights_4x2);
  at::Tensor out_ref = linear_qga4w_reference_impl(
      x,
      at::_convert_weight_to_int4pack_for_cpu(weights_int, group_size),
      group_size,
      scales_and_zeros,
      inner_k_tiles);

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

  const ValueRef r_out = graph.add_tensor(
      out_ref.sizes().vec(),
      from_at_scalartype(out_ref.scalar_type()),
      out_storage);

  VK_GET_OP_FN("et_vk.linear_weight_int4.default")
  (graph,
   {r_x.value,
    r_weights_4x2,
    graph.add_scalar<int64_t>(group_size),
    r_scales_and_zeros,
    kDummyValueRef,
    r_out});

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();

  graph.prepack();

  //
  // Run model
  //

  graph.propagate_resize();
  graph.copy_into_staging(r_x.staging, x.const_data_ptr(), x.numel());

  graph.execute();

  at::Tensor vk_out = at::empty_like(out_ref);
  graph.copy_from_staging(
      staging_out, vk_out.mutable_data_ptr(), vk_out.numel());

  ASSERT_TRUE(at::allclose(vk_out, out_ref, 1e-4, 1e-4));
}

void test_vulkan_linear_qga4w(
    const int B,
    const int M,
    const int K,
    const int N,
    const int group_size = 32,
    const int inner_k_tiles = 8) {
  test_vulkan_linear_qga4w_impl(
      B,
      M,
      K,
      N,
      group_size,
      inner_k_tiles,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  test_vulkan_linear_qga4w_impl(
      B,
      M,
      K,
      N,
      group_size,
      inner_k_tiles,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

void test_vulkan_linear_qcs4w_impl(
    const int B,
    const int M,
    const int K,
    const int N,
    const vkcompute::utils::StorageType in_storage =
        vkcompute::utils::kTexture3D,
    const vkcompute::utils::StorageType out_storage =
        vkcompute::utils::kTexture3D) {
  at::Tensor x = at::rand({B, M, K}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weights_4x2 =
      at::randint(0, 256, {N, K / 2}, at::device(at::kCPU).dtype(at::kByte));

  at::Tensor scales = at::rand({N}, at::device(at::kCPU).dtype(at::kFloat));

  at::Tensor out_ref = linear_qcs4w_reference_impl(x, weights_4x2, scales);

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
  MAKE_TENSORREF_FOR(scales);

  IOValueRef r_x = graph.add_input_tensor(
      x.sizes().vec(), from_at_scalartype(x.scalar_type()), in_storage);

  const ValueRef r_out = graph.add_tensor(
      out_ref.sizes().vec(),
      from_at_scalartype(out_ref.scalar_type()),
      out_storage);

  VK_GET_OP_FN("et_vk.linear_qcs4w.default")
  (graph, {r_x.value, r_weights_4x2, r_scales, r_out});

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();

  graph.prepack();

  //
  // Run model
  //

  graph.propagate_resize();
  graph.copy_into_staging(r_x.staging, x.const_data_ptr(), x.numel());

  graph.execute();

  at::Tensor vk_out = at::empty_like(out_ref);
  graph.copy_from_staging(
      staging_out, vk_out.mutable_data_ptr(), vk_out.numel());

  ASSERT_TRUE(at::allclose(vk_out, out_ref, 1e-4, 1e-4));
}

void test_vulkan_linear_qcs4w(
    const int B,
    const int M,
    const int K,
    const int N) {
  test_vulkan_linear_qcs4w_impl(
      B, M, K, N, vkcompute::utils::kBuffer, vkcompute::utils::kBuffer);

  test_vulkan_linear_qcs4w_impl(
      B, M, K, N, vkcompute::utils::kTexture3D, vkcompute::utils::kTexture3D);
}

void test_vulkan_linear_qta8a_qga4w_impl(
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

  const int64_t input_num_tokens = B * M;
  const int k_groups = K / group_size;

  at::Tensor input_scale =
      at::rand({input_num_tokens}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor input_zero_point = at::randint(
      -10, 10, {input_num_tokens}, at::device(at::kCPU).dtype(at::kInt));

  at::Tensor float_x =
      at::rand({B, M, K}, at::device(at::kCPU).dtype(at::kFloat));

  // Create a reference quantized tensor using per-token quantization
  // Mimic per-token quantization using at::quantize_per_channel by reshaping
  // [num_tokens, features]
  at::Tensor float_x_reshaped = float_x.view({input_num_tokens, K});
  at::Tensor qx_ref_reshaped = at::quantize_per_channel(
      float_x_reshaped,
      input_scale.to(at::kDouble),
      input_zero_point.to(at::kLong),
      0, // axis 0 for per-token (first dimension after reshape)
      c10::ScalarType::QInt8);

  at::Tensor x =
      at::int_repr(qx_ref_reshaped).view(float_x.sizes()).to(at::kChar);

  at::Tensor weights_4x2 =
      at::randint(0, 256, {N, K / 2}, at::device(at::kCPU).dtype(at::kByte));
  at::Tensor weight_scales =
      at::rand({k_groups, N}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weight_zeros = at::randint(
      -128, 128, {k_groups, N}, at::device(at::kCPU).dtype(at::kInt));

  at::Tensor out_ref = linear_qta8a_qga4w_4bit_dequant_impl(
      x,
      input_scale,
      input_zero_point,
      weights_4x2,
      group_size,
      weight_scales,
      weight_zeros);

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
  MAKE_TENSORREF_FOR(weight_scales);
  MAKE_TENSORREF_FOR(weight_zeros);

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

  const ValueRef r_out = graph.add_tensor(
      out_ref.sizes().vec(),
      from_at_scalartype(out_ref.scalar_type()),
      out_storage);

  VK_GET_OP_FN("et_vk.linear_qta8a_qga4w.default")
  (graph,
   {r_x.value,
    r_input_scale.value,
    r_input_zero_point.value,
    r_weights_4x2,
    graph.add_scalar<int64_t>(group_size),
    r_weight_scales,
    r_weight_zeros,
    r_out});

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();

  graph.prepack();

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

  graph.execute();

  at::Tensor vk_out = at::empty_like(out_ref);
  graph.copy_from_staging(
      staging_out, vk_out.mutable_data_ptr(), vk_out.numel());

  // This is a reference implementation that uses the quantized
  // matmul paradigm. It should follow closely with how the vulkan
  // implementation works, and demonstrates reasonably close results.
  at::Tensor qmm_ref = linear_qta8a_qga4w_quantized_matmul(
      x,
      input_scale,
      input_zero_point,
      weights_4x2,
      group_size,
      weight_scales,
      weight_zeros);

  // For quantized int8 operations, allow for 1-unit differences due to rounding
  bool is_close = at::allclose(vk_out, out_ref, 5e-3, 5e-3);
  if (!is_close) {
    std::cout << "qmm_ref: \n" << qmm_ref << std::endl;
    std::cout << "out_ref: \n" << out_ref << std::endl;
    std::cout << "vk_out: \n" << vk_out << std::endl;
  }

  ASSERT_TRUE(is_close);
}

void test_vulkan_linear_qta8a_qga4w(
    const int B,
    const int M,
    const int K,
    const int N,
    const int group_size = 32) {
  test_vulkan_linear_qta8a_qga4w_impl(
      B,
      M,
      K,
      N,
      group_size,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  test_vulkan_linear_qta8a_qga4w_impl(
      B,
      M,
      K,
      N,
      group_size,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

// Test linear_qga4w operator

TEST(VulkanLinearQGA4WTest, test_reference_impl) {
  test_reference_linear_qga4w(
      /*B = */ 1,
      /*M = */ 4,
      /*K = */ 128,
      /*N = */ 32);
}

TEST(VulkanLinearQGA4WTest, test_vulkan_impl_small_m) {
  test_vulkan_linear_qga4w(
      /*B = */ 1,
      /*M = */ 4,
      /*K = */ 128,
      /*N = */ 32);

  test_vulkan_linear_qga4w(
      /*B = */ 1,
      /*M = */ 1,
      /*K = */ 256,
      /*N = */ 256);
}

TEST(VulkanLinearQGA4WTest, test_vulkan_impl_gemm) {
  test_vulkan_linear_qga4w(
      /*B = */ 1,
      /*M = */ 256,
      /*K = */ 256,
      /*N = */ 256);
}

// Test linear_qcs4w operator

TEST_F(VulkanLinearQCS4WTest, test_reference_impl) {
  test_reference_linear_qcs4w(
      /*B = */ 1,
      /*M = */ 4,
      /*K = */ 128,
      /*N = */ 32);
}

TEST_F(VulkanLinearQCS4WTest, test_vulkan_impl_small_m) {
  test_vulkan_linear_qcs4w(
      /*B = */ 1,
      /*M = */ 4,
      /*K = */ 128,
      /*N = */ 32);

  test_vulkan_linear_qcs4w(
      /*B = */ 1,
      /*M = */ 1,
      /*K = */ 256,
      /*N = */ 256);
}

TEST_F(VulkanLinearQCS4WTest, test_vulkan_impl_gemm) {
  test_vulkan_linear_qcs4w(
      /*B = */ 1,
      /*M = */ 32,
      /*K = */ 32,
      /*N = */ 32);
}

// Test linear_qta8a_qga4w operator

TEST_F(
    VulkanLinearQTA8AQGA4WTest,
    test_vulkan_linear_quant_gemm_custom_groupsize) {
  test_vulkan_linear_qta8a_qga4w(
      /*B = */ 1,
      /*M = */ 2,
      /*K = */ 8,
      /*N = */ 8,
      /*group_size = */ 8);

  test_vulkan_linear_qta8a_qga4w(
      /*B = */ 1,
      /*M = */ 2,
      /*K = */ 16,
      /*N = */ 8,
      /*group_size = */ 8);
}

TEST_F(VulkanLinearQTA8AQGA4WTest, test_vulkan_linear_quant_gemm) {
  test_vulkan_linear_qta8a_qga4w(
      /*B = */ 1,
      /*M = */ 4,
      /*K = */ 64,
      /*N = */ 32);

  test_vulkan_linear_qta8a_qga4w(
      /*B = */ 1,
      /*M = */ 4,
      /*K = */ 128,
      /*N = */ 32);

  test_vulkan_linear_qta8a_qga4w(
      /*B = */ 1,
      /*M = */ 8,
      /*K = */ 64,
      /*N = */ 16);

  test_vulkan_linear_qta8a_qga4w(
      /*B = */ 1,
      /*M = */ 256,
      /*K = */ 256,
      /*N = */ 256);
}

TEST_F(
    VulkanLinearQTA8AQGA4WTest,
    test_vulkan_linear_quant_gemv_custom_groupsize) {
  test_vulkan_linear_qta8a_qga4w(
      /*B = */ 1,
      /*M = */ 1,
      /*K = */ 8,
      /*N = */ 8,
      /*group_size = */ 8);

  test_vulkan_linear_qta8a_qga4w(
      /*B = */ 1,
      /*M = */ 1,
      /*K = */ 16,
      /*N = */ 8,
      /*group_size = */ 8);
}

TEST_F(VulkanLinearQTA8AQGA4WTest, test_vulkan_linear_quant_gemv) {
  test_vulkan_linear_qta8a_qga4w(
      /*B = */ 1,
      /*M = */ 1,
      /*K = */ 32,
      /*N = */ 32);

  test_vulkan_linear_qta8a_qga4w(
      /*B = */ 1,
      /*M = */ 1,
      /*K = */ 64,
      /*N = */ 16);

  test_vulkan_linear_qta8a_qga4w(
      /*B = */ 1,
      /*M = */ 1,
      /*K = */ 256,
      /*N = */ 256);
}
