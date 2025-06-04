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

#include <cassert>
#include <iostream>

namespace torch {
namespace executor {
namespace native {

// Forward declarations of the functions we're testing
Tensor& quantize_per_tensor_out(
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out);

Tensor& quantize_per_token_out(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out);

// Wrapper function for quantize_per_tensor_out without context
Tensor& quantize_per_tensor_out_no_context(
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  return torch::executor::native::quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, dtype, out);
}

// Wrapper function for quantize_per_token_out without context
Tensor& quantize_per_token_out_no_context(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  return torch::executor::native::quantize_per_token_out(
      input, scale, zero_point, quant_min, quant_max, dtype, out);
}

// ATen wrapper for quantize_per_tensor
at::Tensor quantize_per_tensor_aten(
    const at::Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  auto out = at::empty_like(input, dtype);
  // Convert at::ScalarType to executorch::ScalarType
  ScalarType et_dtype;
  switch (dtype) {
    case at::kByte:
      et_dtype = ScalarType::Byte;
      break;
    case at::kChar:
      et_dtype = ScalarType::Char;
      break;
    case at::kShort:
      et_dtype = ScalarType::Short;
      break;
    case at::kInt:
      et_dtype = ScalarType::Int;
      break;
    case at::kLong:
      et_dtype = ScalarType::Long;
      break;
    case at::kFloat:
      et_dtype = ScalarType::Float;
      break;
    case at::kDouble:
      et_dtype = ScalarType::Double;
      break;
    default:
      throw std::runtime_error("Unsupported dtype");
  }

  WRAP_TO_ATEN(quantize_per_tensor_out_no_context, 6)
  (input, scale, zero_point, quant_min, quant_max, et_dtype, out);
  return out;
}

// ATen wrapper for quantize_per_token
at::Tensor quantize_per_token_aten(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  auto out = at::empty_like(input, dtype);
  // Convert at::ScalarType to executorch::ScalarType
  ScalarType et_dtype;
  switch (dtype) {
    case at::kByte:
      et_dtype = ScalarType::Byte;
      break;
    case at::kChar:
      et_dtype = ScalarType::Char;
      break;
    case at::kShort:
      et_dtype = ScalarType::Short;
      break;
    case at::kInt:
      et_dtype = ScalarType::Int;
      break;
    case at::kLong:
      et_dtype = ScalarType::Long;
      break;
    case at::kFloat:
      et_dtype = ScalarType::Float;
      break;
    case at::kDouble:
      et_dtype = ScalarType::Double;
      break;
    default:
      throw std::runtime_error("Unsupported dtype");
  }

  WRAP_TO_ATEN(quantize_per_token_out_no_context, 6)
  (input, scale, zero_point, quant_min, quant_max, et_dtype, out);
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch


//
// Test functions
//

// Helper function to get the name of a ScalarType for better error messages
std::string scalar_type_name(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::kLong:
      return "c10::kLong";
    case c10::kShort:
      return "c10::kShort";
    case c10::kComplexHalf:
      return "c10::kComplexHalf";
    case c10::kComplexFloat:
      return "c10::kComplexFloat";
    case c10::kComplexDouble:
      return "c10::kComplexDouble";
    case c10::kBool:
      return "c10::kBool";
    case c10::kQInt8:
      return "c10::kQInt8";
    case c10::kQUInt8:
      return "c10::kQUInt8";
    case c10::kQInt32:
      return "c10::kQInt32";
    case c10::kBFloat16:
      return "c10::kBFloat16";
    case c10::kQUInt4x2:
      return "c10::kQUInt4x2";
    case c10::kQUInt2x4:
      return "c10::kQUInt2x4";
    default:
      return "Unknown(" + std::to_string(static_cast<int>(dtype)) + ")";
  }
}

vkcompute::vkapi::ScalarType from_at_scalartype(c10::ScalarType at_scalartype) {
  using namespace vkcompute;
  switch (at_scalartype) {
    case c10::kFloat:
      return vkapi::kFloat;
    case c10::kHalf:
      return vkapi::kHalf;
    case c10::kInt:
      return vkapi::kInt;
    case c10::kLong:
      // We don't have inherent vkapi::kLong, use kInt instead
      return vkapi::kInt;
    case c10::kChar:
      return vkapi::kChar;
    case c10::kByte:
      return vkapi::kByte;
    case c10::kDouble:
      return vkapi::kDouble;
    case c10::kShort:
      return vkapi::kShort;
    case c10::kUInt16:
      return vkapi::kUInt16;
    default:
      VK_THROW(
          "Unsupported at::ScalarType: ",
          scalar_type_name(at_scalartype),
          " (",
          static_cast<int>(at_scalartype),
          ")");
  }
}

void check_quantize_args(
    int64_t quant_min,
    int64_t quant_max,
    c10::ScalarType out_dtype) {
  using namespace vkcompute;
  int32_t quant_min_lower_bound = 0, quant_max_upper_bound = 0;
  switch (out_dtype) {
    case c10::kByte:
      quant_min_lower_bound =
          static_cast<int32_t>(std::numeric_limits<uint8_t>::min());
      quant_max_upper_bound =
          static_cast<int32_t>(std::numeric_limits<uint8_t>::max());
      break;
    case c10::kChar:
      quant_min_lower_bound =
          static_cast<int32_t>(std::numeric_limits<int8_t>::min());
      quant_max_upper_bound =
          static_cast<int32_t>(std::numeric_limits<int8_t>::max());
      break;
    case c10::kBits16:
    case c10::kUInt16:
      quant_min_lower_bound = std::numeric_limits<uint16_t>::min();
      quant_max_upper_bound = std::numeric_limits<uint16_t>::max();
      break;
    case c10::kShort:
      quant_min_lower_bound = std::numeric_limits<int16_t>::min();
      quant_max_upper_bound = std::numeric_limits<int16_t>::max();
      break;
    case c10::kInt:
      quant_min_lower_bound = std::numeric_limits<int32_t>::min();
      quant_max_upper_bound = std::numeric_limits<int32_t>::max();
      break;
    default:
      VK_CHECK_COND(false, "Unsupported dtype: ", scalar_type_name(out_dtype));
  }
  VK_CHECK_COND(
      quant_min >= quant_min_lower_bound,
      "quant_min out of bound for dtype, expected quant_min_lower_bound: ",
      quant_min_lower_bound,
      " actual quant_min: ",
      quant_min);

  VK_CHECK_COND(
      quant_max <= quant_max_upper_bound,
      "quant_max out of bound for dtype, expected quant_max_upper_bound: ",
      quant_max_upper_bound,
      " actual quant_max: ",
      quant_max);
}
/*
 * Reference implementation of quantize_per_token
 */
at::Tensor quantize_per_token_reference_impl(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  // Create output tensor with the target dtype
  at::Tensor out = at::empty_like(input, dtype);

  // Calculate number of tokens
  int num_tokens = 1;
  for (int i = 0; i < input.dim() - 1; i++) {
    num_tokens *= input.size(i);
  }

  // Verify that the number of tokens matches the size of scale and zero_point
  // tensors
  assert(num_tokens == scale.numel());
  assert(num_tokens == zero_point.numel());

  // Reshape input to [num_tokens, last_dim]
  at::Tensor reshaped_input = input.reshape({num_tokens, input.size(-1)});
  at::Tensor reshaped_out = out.reshape({num_tokens, input.size(-1)});

  // Quantize each token separately
  for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
    // Use float for scale since Vulkan doesn't support double
    float token_scale = scale[token_idx].item<float>();
    // Use int for zero_point since Vulkan doesn't support int64_t
    int token_zero_point = zero_point[token_idx].item<int>();

    float inv_scale = 1.0 / token_scale;

    // Quantize the token
    for (int i = 0; i < input.size(-1); i++) {
      float value = reshaped_input[token_idx][i].item<float>();
      int qvalue = token_zero_point + std::nearbyint(inv_scale * value);

      qvalue = std::max<int64_t>(qvalue, quant_min);
      qvalue = std::min<int64_t>(qvalue, quant_max);

      if (dtype == at::kByte) {
        reshaped_out[token_idx][i] = static_cast<uint8_t>(qvalue);
      } else if (dtype == at::kChar) {
        reshaped_out[token_idx][i] = static_cast<int8_t>(qvalue);
      } else if (dtype == at::kShort) {
        reshaped_out[token_idx][i] = static_cast<int16_t>(qvalue);
      } else if (dtype == at::kInt) {
        reshaped_out[token_idx][i] = static_cast<int32_t>(qvalue);
      } else if (dtype == at::kLong) {
        reshaped_out[token_idx][i] = static_cast<int64_t>(qvalue);
      }
    }
  }

  return out;
}

void test_vulkan_quantize_per_token_impl(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    const vkcompute::utils::StorageType in_storage,
    const vkcompute::utils::StorageType out_storage);

// Wrapper function to test both buffer and texture storage types
void test_vulkan_quantize_per_token(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  // Test with buffer storage
  test_vulkan_quantize_per_token_impl(
      input_sizes,
      scales,
      zero_points,
      quant_min,
      quant_max,
      dtype,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  // Test with texture storage
  test_vulkan_quantize_per_token_impl(
      input_sizes,
      scales,
      zero_points,
      quant_min,
      quant_max,
      dtype,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

void test_reference_quantize_per_token(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  check_quantize_args(quant_min, quant_max, dtype);
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input =
      at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kFloat));

  // Fill with a simple pattern: values from 0 to 1 in steps
  float step = 1.0 / (input.numel() - 1);
  auto flat_input = input.flatten();
  for (int i = 0; i < flat_input.numel(); i++) {
    flat_input[i] = i * step;
  }

  // Reshape back to original dimensions
  input = flat_input.reshape(input_sizes_int64);

  // Calculate number of tokens
  int num_tokens = 1;
  for (int i = 0; i < input.dim() - 1; i++) {
    num_tokens *= input.size(i);
  }

  // Verify that the number of tokens matches the size of scales and zero_points
  ASSERT_EQ(num_tokens, scales.size());
  ASSERT_EQ(num_tokens, zero_points.size());

  // Create scale and zero_point tensors
  at::Tensor scale_tensor =
      at::tensor(scales, at::device(at::kCPU).dtype(at::kDouble));
  at::Tensor zero_point_tensor =
      at::tensor(zero_points, at::device(at::kCPU).dtype(at::kLong));

  // Get reference output
  at::Tensor reference_out = quantize_per_token_reference_impl(
      input, scale_tensor, zero_point_tensor, quant_min, quant_max, dtype);

  // Get implementation output
  at::Tensor impl_out = torch::executor::native::quantize_per_token_aten(
      input, scale_tensor, zero_point_tensor, quant_min, quant_max, dtype);

  // Convert to int for consistent display regardless of underlying type
  at::Tensor reference_int = reference_out.to(at::kInt);
  at::Tensor impl_int = impl_out.to(at::kInt);

  const bool output_correct = at::equal(reference_int, impl_out);
  if (!output_correct) {
    std::cout << "\n"
              << "Failed with parameters: " << std::endl;
    std::cout << "  scale(s):";
    for (size_t i = 0; i < scales.size(); i++) {
      std::cout << " " << scales[i] << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "  zero_point(s):";
    for (size_t i = 0; i < zero_points.size(); i++) {
      std::cout << " " << zero_points[i] << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "  quant_min: " << quant_min << std::endl;
    std::cout << "  quant_max: " << quant_max << std::endl;

    std::cout << "input:" << std::endl;
    std::cout << input << std::endl;
    std::cout << "reference:" << std::endl;
    std::cout << reference_int << std::endl;
    std::cout << "my_reference:" << std::endl;
    std::cout << impl_out << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

void test_vulkan_quantize_per_token_impl(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    const vkcompute::utils::StorageType in_storage =
        vkcompute::utils::kTexture3D,
    const vkcompute::utils::StorageType out_storage =
        vkcompute::utils::kTexture3D) {
  check_quantize_args(quant_min, quant_max, dtype);
  int num_tokens = 1;
  for (int i = 0; i < input_sizes.size() - 1; i++) {
    num_tokens *= input_sizes[i];
  }

  ASSERT_EQ(num_tokens, scales.size());
  ASSERT_EQ(num_tokens, zero_points.size());

  // Create input tensor with random values
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input =
      at::rand(input_sizes_int64, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor scale_tensor =
      at::tensor(scales, at::device(at::kCPU).dtype(at::kDouble));
  at::Tensor zero_point_tensor =
      at::tensor(zero_points, at::device(at::kCPU).dtype(at::kLong));

  // Get reference output to show what we would compare against
  at::Tensor reference_out = torch::executor::native::quantize_per_token_aten(
      input, scale_tensor, zero_point_tensor, quant_min, quant_max, dtype);

  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(in_storage);
  ComputeGraph graph(config);

  IOValueRef r_input = graph.add_input_tensor(
      input.sizes().vec(), from_at_scalartype(input.scalar_type()), in_storage);
  IOValueRef r_scale = graph.add_input_tensor(
      scale_tensor.sizes().vec(), vkapi::kFloat, in_storage);
  IOValueRef r_zero_point = graph.add_input_tensor(
      zero_point_tensor.sizes().vec(), vkapi::kInt, in_storage);

  const ValueRef r_quant_min = graph.add_scalar<int64_t>(quant_min);
  const ValueRef r_quant_max = graph.add_scalar<int64_t>(quant_max);

  const ValueRef r_out = graph.add_tensor(
      input.sizes().vec(), from_at_scalartype(dtype), out_storage);

  VK_GET_OP_FN("quantize_per_token.default")
  (graph,
   {
       r_input.value,
       r_scale.value,
       r_zero_point.value,
       r_quant_min,
       r_quant_max,
       r_out,
   });

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.encode_prepack();
  graph.prepack();
  graph.encode_execute();

  // Copy input data to GPU
  graph.copy_into_staging(
      r_input.staging, input.const_data_ptr(), input.numel());

  // Convert scale tensor to float and copy to GPU
  at::Tensor scale_float = scale_tensor.to(at::kFloat);
  graph.copy_into_staging(
      r_scale.staging, scale_float.const_data_ptr(), scale_float.numel());

  // Convert zero_point tensor to int and copy to GPU
  at::Tensor zero_point_int = zero_point_tensor.to(at::kInt);
  graph.copy_into_staging(
      r_zero_point.staging,
      zero_point_int.const_data_ptr(),
      zero_point_int.numel());

  // Execute the graph
  graph.execute();

  // Copy output data back to CPU
  at::Tensor vk_out = at::empty_like(reference_out).contiguous();
  graph.copy_from_staging(
      staging_out, vk_out.mutable_data_ptr(), vk_out.numel());

  // Compare outputs
  at::Tensor reference_int = reference_out.to(at::kInt);
  at::Tensor vk_int = vk_out.to(at::kInt);

  const bool output_correct = at::equal(reference_int, vk_int);
  if (!output_correct) {
    at::Tensor diffs = at::abs(reference_int - vk_int);

    std::cout << "\n"
              << "Failed with parameters: " << std::endl;
    std::cout << "  scale(s):";
    for (size_t i = 0; i < scales.size(); i++) {
      std::cout << " " << scales[i] << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "  zero_point(s):";
    for (size_t i = 0; i < zero_points.size(); i++) {
      std::cout << " " << zero_points[i] << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "  quant_min: " << quant_min << std::endl;
    std::cout << "  quant_max: " << quant_max << std::endl;
    std::cout << "  storage type: "
              << (in_storage == vkcompute::utils::kBuffer ? "buffer"
                                                          : "texture")
              << std::endl;

    std::cout << "input:" << std::endl;
    std::cout << input << std::endl;
    std::cout << "reference:" << std::endl;
    std::cout << reference_int << std::endl;
    std::cout << "vulkan:" << std::endl;
    std::cout << vk_int << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

TEST(VulkanQuantizePerTensorTest, test_reference_quantize_per_token_int8) {
  std::vector<float> scales = {0.1, 0, 0.3, 0.1, 0.2, 0.3};
  std::vector<int> zero_points = {1, 2, 3, 0, -1, -2};

  test_reference_quantize_per_token(
      {2, 3, 4}, // input sizes (2*3=6 tokens)
      scales,
      zero_points,
      -128, // quant_min
      127, // quant_max
      at::kChar);
}
