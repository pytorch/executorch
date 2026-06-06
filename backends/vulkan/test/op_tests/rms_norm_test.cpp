/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <cmath>
#include <random>

using executorch::aten::Half;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

//
// Helpers
//

std::vector<float> rand_floats(size_t n, unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(n);
  std::generate(data.begin(), data.end(), [&]() { return dist(gen); });
  return data;
}

size_t numel(const std::vector<int64_t>& sizes) {
  size_t n = 1;
  for (auto s : sizes) {
    n *= static_cast<size_t>(s);
  }
  return n;
}

std::vector<int32_t> to_int32(const std::vector<int64_t>& v) {
  return std::vector<int32_t>(v.begin(), v.end());
}

//
// Reference Implementation (pure C++)
//

std::vector<float> rms_norm_ref(
    const std::vector<float>& x,
    const std::vector<float>& weight,
    const std::vector<int64_t>& shape,
    float eps) {
  const size_t hidden = static_cast<size_t>(shape.back());
  const size_t num_rows = x.size() / hidden;
  std::vector<float> out(x.size());

  for (size_t r = 0; r < num_rows; ++r) {
    const size_t off = r * hidden;
    float sq_sum = 0.0f;
    for (size_t i = 0; i < hidden; ++i) {
      sq_sum += x[off + i] * x[off + i];
    }
    float rstd = 1.0f / std::sqrt(sq_sum / static_cast<float>(hidden) + eps);
    for (size_t i = 0; i < hidden; ++i) {
      out[off + i] = x[off + i] * rstd * weight[i];
    }
  }
  return out;
}

//
// Test function
//

void test_rms_norm(
    const std::vector<int64_t>& input_shape,
    const float eps = 1e-5f,
    const vkcompute::vkapi::ScalarType dtype = vkcompute::vkapi::kFloat,
    const vkcompute::utils::StorageType storage_type =
        vkcompute::utils::kTexture3D) {
  const int64_t hidden_size = input_shape.back();
  const size_t input_numel = numel(input_shape);
  const size_t weight_numel = static_cast<size_t>(hidden_size);

  std::vector<float> x_data = rand_floats(input_numel, 42);
  std::vector<float> w_data = rand_floats(weight_numel, 123);

  // For fp16: round-trip through Half so the reference uses the same precision
  // as the GPU input.
  std::vector<Half> x_half, w_half;
  if (dtype == vkcompute::vkapi::kHalf) {
    x_half.resize(input_numel);
    w_half.resize(weight_numel);
    for (size_t i = 0; i < input_numel; ++i) {
      x_half[i] = static_cast<Half>(x_data[i]);
      x_data[i] = static_cast<float>(x_half[i]);
    }
    for (size_t i = 0; i < weight_numel; ++i) {
      w_half[i] = static_cast<Half>(w_data[i]);
      w_data[i] = static_cast<float>(w_half[i]);
    }
  }

  std::vector<float> ref_data = rms_norm_ref(x_data, w_data, input_shape, eps);

  // Build Vulkan graph
  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(storage_type);
  ComputeGraph graph(config);

  IOValueRef r_x = graph.add_input_tensor(input_shape, dtype);

  ValueRef r_weight = (dtype == vkapi::kHalf)
      ? graph.add_tensorref({hidden_size}, vkapi::kHalf, w_half.data())
      : graph.add_tensorref({hidden_size}, vkapi::kFloat, w_data.data());

  const ValueRef r_eps = graph.add_scalar<double>(static_cast<double>(eps));
  const ValueRef r_out = graph.add_tensor(input_shape, dtype);

  VK_GET_OP_FN("et_vk.rms_norm.default")
  (graph, {r_x.value, r_weight, r_eps, r_out});

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.prepack();
  graph.propagate_resize();

  if (dtype == vkapi::kHalf) {
    graph.maybe_cast_and_copy_into_staging(
        r_x.staging, x_half.data(), input_numel, vkapi::kHalf);
  } else {
    graph.maybe_cast_and_copy_into_staging(
        r_x.staging, x_data.data(), input_numel, vkapi::kFloat);
  }

  graph.execute();

  // Read output — fp16 staging returns Half, convert to float for comparison.
  std::vector<float> vk_data(input_numel);
  if (dtype == vkapi::kHalf) {
    std::vector<Half> vk_half(input_numel);
    graph.maybe_cast_and_copy_from_staging(
        staging_out, vk_half.data(), input_numel, vkapi::kHalf);
    for (size_t i = 0; i < input_numel; ++i) {
      vk_data[i] = static_cast<float>(vk_half[i]);
    }
  } else {
    graph.maybe_cast_and_copy_from_staging(
        staging_out, vk_data.data(), input_numel, vkapi::kFloat);
  }

  TensorFactory<ScalarType::Float> tf;
  Tensor ref_tensor = tf.make(to_int32(input_shape), ref_data);
  Tensor vk_tensor = tf.make(to_int32(input_shape), vk_data);

  const double rtol = (dtype == vkapi::kHalf) ? 1e-2 : 1e-3;
  const double atol = (dtype == vkapi::kHalf) ? 1e-2 : 1e-3;
  EXPECT_TENSOR_CLOSE_WITH_TOL(ref_tensor, vk_tensor, rtol, atol);
}

//
// Texture storage tests
//

TEST(VulkanRmsNormTest, basic_small_texture) {
  test_rms_norm({1, 1, 1, 64});
}

TEST(VulkanRmsNormTest, llm_hidden_size_texture) {
  test_rms_norm({1, 1, 1, 896}, 1e-6f);
}

TEST(VulkanRmsNormTest, fp16_texture) {
  test_rms_norm({1, 1, 1, 896}, 1e-6f, vkcompute::vkapi::kHalf);
}

TEST(VulkanRmsNormTest, multi_row_texture) {
  test_rms_norm({1, 1, 7, 896});
}

TEST(VulkanRmsNormTest, multi_z_slice_texture) {
  // C=7 maps to multiple texture Z slices, exercising the y/z decomposition
  test_rms_norm({1, 7, 1, 896});
}

TEST(VulkanRmsNormTest, 4d_multi_z_slice_texture) {
  // 4D shape similar to model's QK norm with multiple Z slices
  test_rms_norm({1, 5, 4, 128});
}

//
// Buffer storage tests
//

TEST(VulkanRmsNormTest, basic_small_buffer) {
  test_rms_norm(
      {1, 1, 1, 64},
      1e-5f,
      vkcompute::vkapi::kFloat,
      vkcompute::utils::kBuffer);
}

TEST(VulkanRmsNormTest, fp16_buffer) {
  test_rms_norm(
      {1, 1, 1, 896},
      1e-6f,
      vkcompute::vkapi::kHalf,
      vkcompute::utils::kBuffer);
}

//
// Dynamic resize test
//

TEST(VulkanRmsNormTest, dynamic_resize_texture) {
  const int64_t hidden_size = 896;
  const float eps = 1e-6f;
  const int prefill_seq_len = 7;

  std::vector<float> w_data = rand_floats(static_cast<size_t>(hidden_size), 99);

  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

  IOValueRef r_x = graph.add_input_tensor(
      {1, 1, prefill_seq_len, hidden_size}, vkapi::kFloat);
  ValueRef r_weight =
      graph.add_tensorref({hidden_size}, vkapi::kFloat, w_data.data());

  const ValueRef r_eps = graph.add_scalar<double>(static_cast<double>(eps));
  const ValueRef r_out =
      graph.add_tensor({1, 1, prefill_seq_len, hidden_size}, vkapi::kFloat);

  VK_GET_OP_FN("et_vk.rms_norm.default")
  (graph, {r_x.value, r_weight, r_eps, r_out});

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.prepack();

  TensorFactory<ScalarType::Float> tf;

  // --- Prefill run (seq_len = 7) ---
  {
    std::vector<int64_t> shape = {1, 1, prefill_seq_len, hidden_size};
    size_t n = numel(shape);
    std::vector<float> x_data = rand_floats(n, 200);
    std::vector<float> ref_data = rms_norm_ref(x_data, w_data, shape, eps);

    graph.resize_input(0, shape);
    graph.propagate_resize();
    graph.maybe_cast_and_copy_into_staging(
        r_x.staging, x_data.data(), n, vkapi::kFloat);

    graph.execute();

    std::vector<float> vk_data(n);
    graph.maybe_cast_and_copy_from_staging(
        staging_out, vk_data.data(), n, vkapi::kFloat);

    Tensor ref_t = tf.make(to_int32(shape), ref_data);
    Tensor vk_t = tf.make(to_int32(shape), vk_data);
    EXPECT_TENSOR_CLOSE_WITH_TOL(ref_t, vk_t, 1e-3, 1e-3) << "Prefill mismatch";
  }

  // --- Decode run (seq_len = 1) ---
  {
    std::vector<int64_t> shape = {1, 1, 1, hidden_size};
    size_t n = numel(shape);
    std::vector<float> x_data = rand_floats(n, 300);
    std::vector<float> ref_data = rms_norm_ref(x_data, w_data, shape, eps);

    graph.resize_input(0, shape);
    graph.propagate_resize();
    graph.maybe_cast_and_copy_into_staging(
        r_x.staging, x_data.data(), n, vkapi::kFloat);

    graph.execute();

    std::vector<float> vk_data(n);
    graph.maybe_cast_and_copy_from_staging(
        staging_out, vk_data.data(), n, vkapi::kFloat);

    Tensor ref_t = tf.make(to_int32(shape), ref_data);
    Tensor vk_t = tf.make(to_int32(shape), vk_data);
    EXPECT_TENSOR_CLOSE_WITH_TOL(ref_t, vk_t, 1e-3, 1e-3) << "Decode mismatch";
  }
}
