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

#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>

#include "test_utils.h"

#include <cassert>
#include <iostream>

namespace torch {
namespace executor {
namespace native {

// Forward declarations of the functions we're testing
std::tuple<Tensor&, Tensor&> choose_qparams_tensor_out(
    const Tensor& input,
    int64_t quant_min,
    int64_t quant_max,
    ET_UNUSED double eps,
    ScalarType dtype,
    Tensor& scale_out,
    Tensor& zero_point_out);

std::tuple<Tensor&, Tensor&> choose_qparams_per_token_asymmetric_out(
    const Tensor& input,
    ScalarType dtype,
    Tensor& scale_out,
    Tensor& zero_point_out);

// Wrapper function for choose_qparams_tensor_out without context
Tensor& choose_qparams_tensor_out_no_context(
    const Tensor& input,
    int64_t quant_min,
    int64_t quant_max,
    ET_UNUSED double eps,
    ScalarType dtype,
    Tensor& scale_out,
    Tensor& zero_point_out) {
  torch::executor::native::choose_qparams_tensor_out(
      input, quant_min, quant_max, eps, dtype, scale_out, zero_point_out);
  return scale_out;
}

// Wrapper function for choose_qparams_per_token_asymmetric_out without context
Tensor& choose_qparams_per_token_asymmetric_out_no_context(
    const Tensor& input,
    ScalarType dtype,
    Tensor& scale_out,
    Tensor& zero_point_out) {
  torch::executor::native::choose_qparams_per_token_asymmetric_out(
      input, dtype, scale_out, zero_point_out);
  return scale_out;
}

// ATen wrapper for choose_qparams_tensor
std::tuple<at::Tensor, at::Tensor> choose_qparams_tensor_aten(
    const at::Tensor& input,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  auto scale_out = at::empty({}, at::device(at::kCPU).dtype(at::kDouble));
  auto zero_point_out = at::empty({}, at::device(at::kCPU).dtype(at::kLong));
  double eps = 1e-7;

  ScalarType et_dtype = at_scalartype_to_et_scalartype(dtype);

  // Use WRAP_TO_ATEN with the wrapper function
  WRAP_TO_ATEN(choose_qparams_tensor_out_no_context, 5)
  (input, quant_min, quant_max, eps, et_dtype, scale_out, zero_point_out);

  return {scale_out, zero_point_out};
}

// ATen wrapper for choose_qparams_per_token_asymmetric
std::tuple<at::Tensor, at::Tensor> choose_qparams_per_token_asymmetric_aten(
    const at::Tensor& input,
    at::ScalarType dtype) {
  // Calculate output sizes for scale and zero_point tensors
  std::vector<int64_t> output_sizes;
  for (int64_t i = 0; i < input.dim() - 1; i++) {
    output_sizes.push_back(input.size(i));
  }
  output_sizes.push_back(1);

  auto scale_out =
      at::empty(output_sizes, at::device(at::kCPU).dtype(at::kDouble));
  auto zero_point_out =
      at::empty(output_sizes, at::device(at::kCPU).dtype(at::kLong));

  ScalarType et_dtype = at_scalartype_to_et_scalartype(dtype);

  // Use WRAP_TO_ATEN with the wrapper function
  WRAP_TO_ATEN(choose_qparams_per_token_asymmetric_out_no_context, 2)
  (input, et_dtype, scale_out, zero_point_out);

  return {scale_out, zero_point_out};
}

} // namespace native
} // namespace executor
} // namespace torch

//
// Reference Implementation
//

/*
 * Reference implementation of choose_qparams_tensor
 */
std::tuple<at::Tensor, at::Tensor> choose_qparams_tensor_reference_impl(
    const at::Tensor& input,
    int64_t quant_min,
    int64_t quant_max) {
  // Create output tensors
  at::Tensor scale_out = at::empty({}, at::device(at::kCPU).dtype(at::kDouble));
  at::Tensor zero_point_out =
      at::empty({}, at::device(at::kCPU).dtype(at::kLong));

  // Find min and max values in the input tensor
  float min_val = input.min().item<float>();
  float max_val = input.max().item<float>();

  // Extend the [min, max] interval to ensure it contains 0
  min_val = std::min(min_val, 0.f);
  max_val = std::max(max_val, 0.f);

  // Calculate scale
  double scale =
      (static_cast<double>(max_val) - min_val) / (quant_max - quant_min);

  // Handle small scale
  constexpr float SMALL_SCALE_THRESHOLD = 6.1e-5f;
  if (float(scale) == 0.0f || std::isinf(1.0f / float(scale))) {
    scale = 0.1;
  }

  if (scale < SMALL_SCALE_THRESHOLD) {
    float org_scale = scale;
    scale = SMALL_SCALE_THRESHOLD;
    // Adjust min and max based on new scale
    if (min_val == 0.0f) {
      max_val = SMALL_SCALE_THRESHOLD * (quant_max - quant_min);
    } else if (max_val == 0.0f) {
      min_val = -SMALL_SCALE_THRESHOLD * (quant_max - quant_min);
    } else {
      float amplifier = SMALL_SCALE_THRESHOLD / org_scale;
      min_val *= amplifier;
      max_val *= amplifier;
    }
  }

  // Calculate zero point
  double zero_point_from_min = quant_min - min_val / static_cast<double>(scale);
  double zero_point_from_max = quant_max - max_val / static_cast<double>(scale);
  double zero_point_from_min_error =
      std::abs(quant_min) - std::abs(min_val / static_cast<double>(scale));
  double zero_point_from_max_error =
      std::abs(quant_max) - std::abs(max_val / static_cast<double>(scale));
  double initial_zero_point =
      zero_point_from_min_error < zero_point_from_max_error
      ? zero_point_from_min
      : zero_point_from_max;

  // Nudge zero point to be an integer
  int64_t nudged_zero_point = 0;
  if (initial_zero_point < quant_min) {
    nudged_zero_point = quant_min;
  } else if (initial_zero_point > quant_max) {
    nudged_zero_point = quant_max;
  } else {
    nudged_zero_point = std::nearbyint(static_cast<float>(initial_zero_point));
  }

  // Set output values - use item_mutable() for scalar tensors
  scale_out.fill_(scale);
  zero_point_out.fill_(nudged_zero_point);

  return std::make_tuple(scale_out, zero_point_out);
}

// Forward declaration of implementation functions
void test_vulkan_choose_qparams_tensor_impl(
    const std::vector<int>& input_sizes,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    const vkcompute::utils::StorageType in_storage,
    const vkcompute::utils::StorageType out_storage);

// Wrapper function to test both buffer and texture storage types
void test_vulkan_choose_qparams_tensor(
    const std::vector<int>& input_sizes,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  // Test with buffer storage
  test_vulkan_choose_qparams_tensor_impl(
      input_sizes,
      quant_min,
      quant_max,
      dtype,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  // Test with texture storage
  test_vulkan_choose_qparams_tensor_impl(
      input_sizes,
      quant_min,
      quant_max,
      dtype,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

void test_reference_choose_qparams_tensor(
    const std::vector<int>& input_sizes,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input =
      at::rand(input_sizes_int64, at::device(at::kCPU).dtype(at::kFloat));

  // Get reference output
  auto [reference_scale, reference_zero_point] =
      choose_qparams_tensor_reference_impl(input, quant_min, quant_max);

  // Get implementation output
  auto [impl_scale, impl_zero_point] =
      torch::executor::native::choose_qparams_tensor_aten(
          input, quant_min, quant_max, dtype);

  // Compare outputs
  const bool scale_correct = at::allclose(reference_scale, impl_scale);
  const bool zero_point_correct =
      at::equal(reference_zero_point, impl_zero_point);

  if (!scale_correct || !zero_point_correct) {
    std::cout << "\n"
              << "Failed with parameters: " << std::endl;
    std::cout << "  quant_min: " << quant_min << std::endl;
    std::cout << "  quant_max: " << quant_max << std::endl;

    std::cout << "input:" << std::endl;
    std::cout << input << std::endl;
    std::cout << "reference scale:" << std::endl;
    std::cout << reference_scale << std::endl;
    std::cout << "implementation scale:" << std::endl;
    std::cout << impl_scale << std::endl;
    std::cout << "reference zero_point:" << std::endl;
    std::cout << reference_zero_point << std::endl;
    std::cout << "implementation zero_point:" << std::endl;
    std::cout << impl_zero_point << std::endl;
  }

  ASSERT_TRUE(scale_correct && zero_point_correct);
}

void test_vulkan_choose_qparams_tensor_impl(
    const std::vector<int>& input_sizes,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    const vkcompute::utils::StorageType in_storage,
    const vkcompute::utils::StorageType out_storage) {
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input =
      at::rand(input_sizes_int64, at::device(at::kCPU).dtype(at::kFloat));

  // Get reference output
  auto [reference_scale, reference_zero_point] =
      torch::executor::native::choose_qparams_tensor_aten(
          input, quant_min, quant_max, dtype);

  // Build Vulkan choose_qparams_tensor graph
  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(in_storage);
  ComputeGraph graph(config);

  IOValueRef r_input = graph.add_input_tensor(
      input.sizes().vec(), from_at_scalartype(input.scalar_type()), in_storage);

  const ValueRef r_quant_min = graph.add_scalar<int64_t>(quant_min);
  const ValueRef r_quant_max = graph.add_scalar<int64_t>(quant_max);

  // Output tensors
  const ValueRef r_scale = graph.add_tensor({}, vkapi::kFloat, out_storage);
  const ValueRef r_zero_point = graph.add_tensor({}, vkapi::kInt, out_storage);

  VK_GET_OP_FN("choose_qparams.tensor")
  (graph,
   {
       r_input.value,
       r_quant_min,
       r_quant_max,
       r_scale,
       r_zero_point,
   });

  ValueRef staging_scale = graph.set_output_tensor(r_scale);
  ValueRef staging_zero_point = graph.set_output_tensor(r_zero_point);

  graph.prepare();
  graph.encode_prepack();
  graph.prepack();
  graph.encode_execute();

  // Run Vulkan choose_qparams_tensor
  graph.copy_into_staging(
      r_input.staging, input.const_data_ptr(), input.numel());

  graph.execute();

  // Create output tensors to hold the results - use types that match GPU output
  at::Tensor vk_scale =
      at::empty({}, at::device(at::kCPU).dtype(at::kFloat)).contiguous();
  at::Tensor vk_zero_point =
      at::empty({}, at::device(at::kCPU).dtype(at::kInt)).contiguous();

  // Copy results from GPU to CPU
  graph.copy_from_staging(
      staging_scale, vk_scale.mutable_data_ptr(), vk_scale.numel());
  graph.copy_from_staging(
      staging_zero_point,
      vk_zero_point.mutable_data_ptr(),
      vk_zero_point.numel());

  // Convert reference values to match Vulkan output types for comparison
  at::Tensor reference_scale_float = reference_scale.to(at::kFloat);
  at::Tensor reference_zero_point_int = reference_zero_point.to(at::kInt);

  // Compare outputs
  const bool scale_correct = at::allclose(reference_scale_float, vk_scale);
  const bool zero_point_correct =
      at::equal(reference_zero_point_int, vk_zero_point);

  if (!scale_correct || !zero_point_correct) {
    std::cout << "\n"
              << "Failed with parameters: " << std::endl;
    std::cout << "  quant_min: " << quant_min << std::endl;
    std::cout << "  quant_max: " << quant_max << std::endl;
    std::cout << "  storage type: "
              << (in_storage == vkcompute::utils::kBuffer ? "buffer"
                                                          : "texture")
              << std::endl;

    // make sure that there arent a ton of elements in the input tensor
    if (input.numel() < 100) {
      std::cout << "input:" << std::endl;
      std::cout << input << "\n" << std::endl;
      std::cout << "reference scale:" << std::endl;
      std::cout << reference_scale << std::endl;
      std::cout << "vulkan scale:" << std::endl;
      std::cout << vk_scale << "\n" << std::endl;
      std::cout << "reference zero_point:" << std::endl;
      std::cout << reference_zero_point << std::endl;
      std::cout << "vulkan zero_point:" << std::endl;
      std::cout << vk_zero_point << std::endl;
    }
  }

  ASSERT_TRUE(scale_correct && zero_point_correct);
}

TEST(VulkanChooseQparamsTest, test_reference_choose_qparams_tensor_int8) {
  test_reference_choose_qparams_tensor(
      {2, 3, 4}, // input sizes
      -128, // quant_min
      127, // quant_max
      at::kChar);
}
