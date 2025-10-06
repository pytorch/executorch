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
#include <limits>

float eps = 1e-7;

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

Tensor& quantize_per_channel_out(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out);

Tensor& quantize_per_tensor_tensor_args_out(
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

// Wrapper function for quantize_per_channel_out without context
Tensor& quantize_per_channel_out_no_context(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  return torch::executor::native::quantize_per_channel_out(
      input, scale, zero_point, axis, quant_min, quant_max, dtype, out);
}

// Wrapper function for quantize_per_tensor_tensor_args_out without context
Tensor& quantize_per_tensor_tensor_args_out_no_context(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  return torch::executor::native::quantize_per_tensor_tensor_args_out(
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
  ScalarType et_dtype = at_scalartype_to_et_scalartype(dtype);

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
  ScalarType et_dtype = at_scalartype_to_et_scalartype(dtype);

  WRAP_TO_ATEN(quantize_per_token_out_no_context, 6)
  (input, scale, zero_point, quant_min, quant_max, et_dtype, out);
  return out;
}

// ATen wrapper for quantize_per_channel
at::Tensor quantize_per_channel_aten(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  auto out = at::empty_like(input, dtype);
  ScalarType et_dtype = at_scalartype_to_et_scalartype(dtype);

  WRAP_TO_ATEN(quantize_per_channel_out_no_context, 7)
  (input, scale, zero_point, axis, quant_min, quant_max, et_dtype, out);
  return out;
}

// ATen wrapper for quantize_per_tensor with tensor args
at::Tensor quantize_per_tensor_tensor_args_aten(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  auto out = at::empty_like(input, dtype);
  ScalarType et_dtype = at_scalartype_to_et_scalartype(dtype);

  WRAP_TO_ATEN(quantize_per_tensor_tensor_args_out_no_context, 6)
  (input, scale, zero_point, quant_min, quant_max, et_dtype, out);
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

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

/**
 * Helper function to validate quantize_per_channel arguments
 * Similar to the validation in op_quantize.cpp
 */
void check_quantize_per_channel_args(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t axis) {
  // Normalize axis
  int64_t normalized_axis = axis;
  if (normalized_axis < 0) {
    normalized_axis += input_sizes.size();
  }

  ASSERT_GE(normalized_axis, 0)
      << "axis " << axis << " is not legal, normalized axis " << normalized_axis
      << " should be >= 0";

  ASSERT_LT(normalized_axis, static_cast<int64_t>(input_sizes.size()))
      << "axis " << axis << " is not legal, normalized axis " << normalized_axis
      << " should be < input.dim() " << input_sizes.size();

  int64_t num_channels = input_sizes[normalized_axis];

  ASSERT_EQ(num_channels, static_cast<int64_t>(scales.size()))
      << "Expected scales.size() to match input.size(axis) (" << num_channels
      << "), but got " << scales.size();

  ASSERT_EQ(num_channels, static_cast<int64_t>(zero_points.size()))
      << "Expected zero_points.size() to match input.size(axis) ("
      << num_channels << "), but got " << zero_points.size();
}

//
// Reference Implementation
//

/*
 * Reference implementation of quantize_per_tensor
 */
at::Tensor quantize_per_tensor_reference_impl(
    const at::Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  // Create output tensor with the target dtype
  at::Tensor out = at::empty_like(input, dtype);

  // Quantize the input tensor
  float inv_scale = 1.0 / scale;

  // Iterate through the tensor and quantize each element
  at::Tensor float_input = input.to(at::kFloat);
  at::Tensor float_values = float_input.flatten();

  auto out_flat = out.flatten();

  for (int i = 0; i < float_values.numel(); i++) {
    float value = float_values[i].item<float>();
    int64_t qvalue = zero_point + std::nearbyint(inv_scale * value);

    qvalue = std::max<int64_t>(qvalue, quant_min);
    qvalue = std::min<int64_t>(qvalue, quant_max);

    if (dtype == at::kByte) {
      out_flat[i] = static_cast<uint8_t>(qvalue);
    } else if (dtype == at::kChar) {
      out_flat[i] = static_cast<int8_t>(qvalue);
    } else if (dtype == at::kShort) {
      out_flat[i] = static_cast<int16_t>(qvalue);
    } else if (dtype == at::kInt) {
      out_flat[i] = static_cast<int32_t>(qvalue);
    } else if (dtype == at::kLong) {
      out_flat[i] = static_cast<int64_t>(qvalue);
    }
  }

  return out.reshape(input.sizes());
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

/*
 * Reference implementation of quantize_per_channel
 */
at::Tensor quantize_per_channel_reference_impl(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  // Normalize axis to handle negative values
  int64_t normalized_axis = axis;
  if (normalized_axis < 0) {
    normalized_axis += input.dim();
  }

  // Create output tensor with the same shape as input but with target dtype
  at::Tensor output = at::empty_like(input, dtype);

  // Get the number of channels along the quantization axis
  int64_t num_channels = input.size(normalized_axis);

  // Calculate strides for efficient indexing
  std::vector<int64_t> input_strides;
  std::vector<int64_t> input_sizes;
  for (int64_t i = 0; i < input.dim(); i++) {
    input_sizes.push_back(input.size(i));
    input_strides.push_back(input.stride(i));
  }

  // Get data pointers
  const float* input_data = input.const_data_ptr<float>();
  const double* scale_data = scale.const_data_ptr<double>();
  const int64_t* zero_point_data = zero_point.const_data_ptr<int64_t>();

  // Iterate through all elements in the tensor
  int64_t total_elements = input.numel();

  // Helper lambda to convert flat index to multi-dimensional coordinates
  auto flat_to_coords = [&](int64_t flat_idx, std::vector<int64_t>& coords) {
    int64_t remaining = flat_idx;
    for (int64_t dim = input.dim() - 1; dim >= 0; dim--) {
      coords[dim] = remaining % input_sizes[dim];
      remaining /= input_sizes[dim];
    }
  };

  // Process each element
  std::vector<int64_t> coords(input.dim());
  for (int64_t flat_idx = 0; flat_idx < total_elements; flat_idx++) {
    // Convert flat index to coordinates
    flat_to_coords(flat_idx, coords);

    // Get the channel index for this element
    int64_t channel_idx = coords[normalized_axis];

    // Get the quantization parameters for this channel
    double channel_scale = scale_data[channel_idx];
    int64_t channel_zero_point = zero_point_data[channel_idx];

    // Get the input value
    float input_value = input_data[flat_idx];

    // Apply quantization formula: round(input / scale) + zero_point
    float inv_scale = 1.0f / static_cast<float>(channel_scale);
    int64_t quantized_value = static_cast<int64_t>(
        static_cast<int32_t>(channel_zero_point) +
        std::nearbyint(static_cast<float>(inv_scale * input_value)));

    // Clamp to quantization bounds
    quantized_value = std::max<int64_t>(quantized_value, quant_min);
    quantized_value = std::min<int64_t>(quantized_value, quant_max);

    // Store the result based on output dtype
    switch (dtype) {
      case at::kByte: {
        uint8_t* output_data = output.mutable_data_ptr<uint8_t>();
        output_data[flat_idx] = static_cast<uint8_t>(quantized_value);
        break;
      }
      case at::kChar: {
        int8_t* output_data = output.mutable_data_ptr<int8_t>();
        output_data[flat_idx] = static_cast<int8_t>(quantized_value);
        break;
      }
      case at::kShort: {
        int16_t* output_data = output.mutable_data_ptr<int16_t>();
        output_data[flat_idx] = static_cast<int16_t>(quantized_value);
        break;
      }
      case at::kInt: {
        int32_t* output_data = output.mutable_data_ptr<int32_t>();
        output_data[flat_idx] = static_cast<int32_t>(quantized_value);
        break;
      }
      default:
        assert(false && "Unsupported output dtype");
    }
  }

  return output;
}

// Forward declaration of implementation functions
void test_vulkan_quantize_per_token_impl(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype,
    at::ScalarType dtype,
    const vkcompute::utils::StorageType in_storage,
    const vkcompute::utils::StorageType out_storage);

void test_vulkan_quantize_per_channel_impl(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype,
    at::ScalarType dtype,
    const vkcompute::utils::StorageType in_storage,
    const vkcompute::utils::StorageType out_storage);

void test_vulkan_quantize_per_tensor_tensor_impl(
    const std::vector<int>& input_sizes,
    float scale,
    int zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype,
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
    at::ScalarType in_dtype = at::kFloat,
    at::ScalarType dtype = at::kInt) {
  // Test with buffer storage
  test_vulkan_quantize_per_token_impl(
      input_sizes,
      scales,
      zero_points,
      quant_min,
      quant_max,
      in_dtype,
      dtype,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  // If the in_dtype is a double, convert to float for texture implementation
  // since they don't support 64bit as inputs
  if (in_dtype == at::kDouble) {
    in_dtype = at::kFloat;
  }

  // Test with texture storage
  test_vulkan_quantize_per_token_impl(
      input_sizes,
      scales,
      zero_points,
      quant_min,
      quant_max,
      in_dtype,
      dtype,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

// Wrapper function to test both buffer and texture storage types
void test_vulkan_quantize_per_channel(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype = at::kFloat,
    at::ScalarType dtype = at::kInt) {
  // Test with buffer storage
  test_vulkan_quantize_per_channel_impl(
      input_sizes,
      scales,
      zero_points,
      axis,
      quant_min,
      quant_max,
      in_dtype,
      dtype,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  // If the in_dtype is a double, convert to float for texture implementation
  // since they don't support 64bit as inputs
  if (in_dtype == at::kDouble) {
    in_dtype = at::kFloat;
  }

  test_vulkan_quantize_per_channel_impl(
      input_sizes,
      scales,
      zero_points,
      axis,
      quant_min,
      quant_max,
      in_dtype,
      dtype,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

// Wrapper function to test both buffer and texture storage types
void test_vulkan_quantize_per_tensor_tensor(
    const std::vector<int>& input_sizes,
    float scale,
    int zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype = at::kFloat,
    at::ScalarType dtype = at::kInt) {
  // Test with buffer storage
  test_vulkan_quantize_per_tensor_tensor_impl(
      input_sizes,
      scale,
      zero_point,
      quant_min,
      quant_max,
      in_dtype,
      dtype,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  // If the in_dtype is a double, convert to float for texture implementation
  // since they don't support 64bit as inputs
  if (in_dtype == at::kDouble) {
    in_dtype = at::kFloat;
  }

  // Test with texture storage
  test_vulkan_quantize_per_tensor_tensor_impl(
      input_sizes,
      scale,
      zero_point,
      quant_min,
      quant_max,
      in_dtype,
      dtype,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

void test_reference_quantize_per_tensor(
    const std::vector<int>& input_sizes,
    float scale,
    int zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype = at::kFloat,
    at::ScalarType dtype = at::kInt) {
  check_quantize_args(quant_min, quant_max, dtype);
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input =
      at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(in_dtype));

  // Fill with a simple pattern: values from 0 to 1 in steps
  float step = 1.0f / (input.numel() - 1);
  auto flat_input = input.flatten();
  for (int i = 0; i < flat_input.numel(); i++) {
    flat_input[i] = i * step;
  }

  // Reshape back to original dimensions
  input = flat_input.reshape(input_sizes_int64);

  scale = scale < eps ? eps : scale;

  // Get reference output
  at::Tensor reference_out = quantize_per_tensor_reference_impl(
      input, scale, zero_point, quant_min, quant_max, dtype);

  // Get implementation output
  at::Tensor impl_out = torch::executor::native::quantize_per_tensor_aten(
      input, scale, zero_point, quant_min, quant_max, dtype);

  // Convert to int for consistent display regardless of underlying type
  at::Tensor reference_int = reference_out.to(at::kInt);
  at::Tensor impl_int = impl_out.to(at::kInt);

  const bool output_correct = at::equal(reference_int, impl_int);
  if (!output_correct) {
    at::Tensor diffs = at::abs(reference_int - impl_int);

    std::cout << "\n"
              << "Failed with parameters: " << std::endl;
    std::cout << "  scale: " << scale << std::endl;
    std::cout << "  zero_point: " << zero_point << std::endl;
    std::cout << "  quant_min: " << quant_min << std::endl;
    std::cout << "  quant_max: " << quant_max << std::endl;

    std::cout << "input:" << std::endl;
    std::cout << input << std::endl;
    std::cout << "reference:" << std::endl;
    std::cout << reference_int << std::endl;
    std::cout << "my_reference:" << std::endl;
    std::cout << impl_int << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

TEST(
    VulkanQuantizePerTensorTest,
    test_reference_quantize_per_tensor_float_to_int8) {
  test_reference_quantize_per_tensor(
      {2, 3, 4}, // input sizes
      0.1, // scale
      0, // zero_point
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);
}

TEST(
    VulkanQuantizePerTensorTest,
    test_reference_quantize_per_tensor_float_to_int32) {
  test_reference_quantize_per_tensor(
      {2, 3, 4}, // input sizes
      0.04, // scale
      5, // zero_point
      std::numeric_limits<int32_t>::min(), // quant_min
      std::numeric_limits<int32_t>::max(), // quant_max
      at::kFloat,
      at::kInt);
}

TEST(
    VulkanQuantizePerTensorTest,
    test_reference_quantize_per_tensor_half_to_uint8) {
  test_reference_quantize_per_tensor(
      {2, 3, 4}, // input sizes
      0.2, // scale
      2, // zero_point
      0, // quant_min
      255, // quant_max
      at::kHalf,
      at::kByte);
}

TEST(
    VulkanQuantizePerTensorTest,
    test_reference_quantize_per_tensor_half_to_int32) {
  test_reference_quantize_per_tensor(
      {2, 3, 4}, // input sizes
      0.01, // scale
      1, // zero_point
      std::numeric_limits<int32_t>::min(), // quant_min
      std::numeric_limits<int32_t>::max(), // quant_max
      at::kHalf,
      at::kInt);
}

// No Vulkan tests for quantized_decomposed.quantize_per_tensor.default
// because it is not going to be implemented in Vulkan since we will
// be handling any future calls to this op via the export stage

void test_reference_quantize_per_token(
    const std::vector<int>& input_sizes,
    const std::vector<float>& pre_scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype = at::kFloat,
    at::ScalarType dtype = at::kInt) {
  check_quantize_args(quant_min, quant_max, dtype);
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input =
      at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(in_dtype));

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
  ASSERT_EQ(num_tokens, pre_scales.size());
  ASSERT_EQ(num_tokens, zero_points.size());

  std::vector<float> scales = pre_scales;
  for (auto& s : scales) {
    s = s < eps ? eps : s;
  }

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
    const std::vector<float>& pre_scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype = at::kFloat,
    at::ScalarType dtype = at::kInt,
    const vkcompute::utils::StorageType in_storage =
        vkcompute::utils::kTexture3D,
    const vkcompute::utils::StorageType out_storage =
        vkcompute::utils::kTexture3D) {
  check_quantize_args(quant_min, quant_max, dtype);
  int num_tokens = 1;
  for (int i = 0; i < input_sizes.size() - 1; i++) {
    num_tokens *= input_sizes[i];
  }

  ASSERT_EQ(num_tokens, pre_scales.size());
  ASSERT_EQ(num_tokens, zero_points.size());

  std::vector<float> scales = pre_scales;
  for (auto& s : scales) {
    s = s < eps ? eps : s;
  }

  // Create input tensor with random values
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input =
      at::rand(input_sizes_int64, at::device(at::kCPU).dtype(in_dtype));
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
      scale_tensor.sizes().vec(),
      vkapi::kFloat,
      utils::kBuffer,
      utils::kWidthPacked);
  IOValueRef r_zero_point = graph.add_input_tensor(
      zero_point_tensor.sizes().vec(),
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);

  const ValueRef r_quant_min = graph.add_scalar<int64_t>(quant_min);
  const ValueRef r_quant_max = graph.add_scalar<int64_t>(quant_max);

  const ValueRef r_out = graph.add_tensor(
      input.sizes().vec(), from_at_scalartype(dtype), out_storage);

  const ValueRef r_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(dtype));

  VK_GET_OP_FN("quantized_decomposed.quantize_per_token.default")
  (graph,
   {
       r_input.value,
       r_scale.value,
       r_zero_point.value,
       r_quant_min,
       r_quant_max,
       r_dtype,
       r_out,
   });

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();

  graph.prepack();

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

  // Tolerance is 1 to address rounding errors and fp math differences between
  // CPU/GPU
  const bool output_correct =
      at::allclose(reference_int, vk_int, /*rtol=*/1, /*atol=*/1);
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

TEST(
    VulkanQuantizePerTokenTest,
    test_reference_quantize_per_token_float_to_int8) {
  std::vector<float> scales = {0.1, 0, 0.3, 0.1, 0.2, 0.3};
  std::vector<int> zero_points = {1, 2, 3, 0, -1, -2};

  test_reference_quantize_per_token(
      {2, 3, 4}, // input sizes (2*3=6 tokens)
      scales,
      zero_points,
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);
}

TEST(
    VulkanQuantizePerTokenTest,
    test_reference_quantize_per_token_float_to_int32) {
  std::vector<float> scales = {0.1, 0, 0.3, 0.1, 0.2, 0.3};
  std::vector<int> zero_points = {1, 2, 3, 0, -1, -2};

  test_reference_quantize_per_token(
      {2, 3, 4}, // input sizes (2*3=6 tokens)
      scales,
      zero_points,
      std::numeric_limits<int32_t>::min(), // quant_min
      std::numeric_limits<int32_t>::max(), // quant_max
      at::kFloat,
      at::kInt);
}

TEST(
    VulkanQuantizePerTokenTest,
    test_reference_quantize_per_token_half_to_int32) {
  std::vector<float> scales = {0.1, 0, 0.3, 0.1, 0.2, 0.3};
  std::vector<int> zero_points = {1, 2, 3, 0, -1, -2};

  test_reference_quantize_per_token(
      {2, 3, 4}, // input sizes (2*3=6 tokens)
      scales,
      zero_points,
      std::numeric_limits<int32_t>::min(), // quant_min
      std::numeric_limits<int32_t>::max(), // quant_max
      at::kHalf,
      at::kInt);
}

TEST(
    VulkanQuantizePerTokenTest,
    test_reference_quantize_per_token_half_to_uint8) {
  std::vector<float> scales = {0.1, 0, 0.3, 0.1, 0.2, 0.3};
  std::vector<int> zero_points = {1, 2, 3, 0, -1, -2};

  test_reference_quantize_per_token(
      {2, 3, 4}, // input sizes (2*3=6 tokens)
      scales,
      zero_points,
      0, // quant_min
      255, // quant_max
      at::kHalf,
      at::kByte);
}

TEST(
    VulkanQuantizePerTokenTest,
    test_vulkan_quantize_per_token_float_to_uint8) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales = {
      -0.5, -0.3, -0.2, 0, 0.1, 0.8, 0.1, 0.2, 0.3, 0.4};
  std::vector<int> zero_points = {-8, 0, 15, 20, 19, 12, 47, 1, -50, -12};

  test_vulkan_quantize_per_token(
      {5, 2, 4}, // input sizes (5*2=10 tokens)
      scales,
      zero_points,
      0, // quant_min
      255, // quant_max
      at::kFloat,
      at::kByte);
}

TEST(VulkanQuantizePerTokenTest, test_vulkan_quantize_per_token_float_to_int8) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales = {
      -0.5, -0.3, -0.2, 0, 0.1, 0.8, 0.1, 0.2, 0.3, 0.4};
  std::vector<int> zero_points = {-8, 0, 15, 20, 19, 12, 47, 1, -50, -12};

  test_vulkan_quantize_per_token(
      {5, 2, 4}, // input sizes (5 tokens)
      scales,
      zero_points,
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);
}

TEST(
    VulkanQuantizePerTokenTest,
    test_vulkan_quantize_per_token_float_to_int32) {
  std::vector<float> scales = {
      -0.5, -0.3, -0.2, 0, 0.1, 0.8, 0.1, 0.2, 0.3, 0.4};
  std::vector<int> zero_points = {-8, 0, 15, 20, 19, 12, 47, 1, -50, -12};

  test_vulkan_quantize_per_token(
      {5, 2, 4}, // input sizes (5*2=10 tokens)
      scales,
      zero_points,
      -2147483648, // quant_min
      2147483647, // quant_max
      at::kFloat,
      at::kInt);
}

TEST(
    VulkanQuantizePerTokenTest,
    test_vulkan_quantize_per_token_float_to_int32_small_scales) {
  std::vector<float> scales = {
      0,
      2.9387358770557188e-39f,
      1.40129846e-45f,
      1.17549435e-38f,
      0.0000000000001};
  std::vector<int> zero_points = {20, -10, 15, 200, 50};

  test_vulkan_quantize_per_token(
      {5, 2}, // input sizes (3 tokens)
      scales,
      zero_points,
      -2147483648, // quant_min
      2147483647, // quant_max
      at::kFloat,
      at::kInt);
}

TEST(
    VulkanQuantizePerTokenTest,
    test_vulkan_quantize_per_token_float_to_uint8_many_tokens) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales(18, 0.1);
  std::vector<int> zero_points(18, 5);

  // Alternate scale values
  for (size_t i = 0; i < scales.size(); i++) {
    scales[i] = (i % 2 == 0) ? 0.3 : -0.5;
  }

  test_vulkan_quantize_per_token(
      {3, 3, 2, 3}, // input sizes (3*3*2=18 tokens)
      scales,
      zero_points,
      0, // quant_min
      125, // quant_max
      at::kFloat,
      at::kByte);
}

TEST(VulkanQuantizePerTokenTest, test_vulkan_quantize_per_token_half_to_int8) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_float16_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales = {0.1, 0.2};
  std::vector<int> zero_points = {0, 5};

  test_vulkan_quantize_per_token(
      {2, 2}, // input sizes (2*2=4 tokens)
      scales,
      zero_points,
      -128, // quant_min
      127, // quant_max
      at::kHalf, // input dtype
      at::kChar); // output dtype
}

TEST(
    VulkanQuantizePerTokenTest,
    test_vulkan_quantize_per_token_double_to_int8) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales = {0.1, 0.2};
  std::vector<int> zero_points = {0, 5};

  test_vulkan_quantize_per_token(
      {2, 2}, // input sizes (2*2=4 tokens)
      scales,
      zero_points,
      -128, // quant_min
      127, // quant_max
      at::kDouble, // input dtype
      at::kChar); // output dtype
}

void test_reference_quantize_per_channel(
    const std::vector<int>& input_sizes,
    const std::vector<float>& pre_scales,
    const std::vector<int>& zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype = at::kFloat,
    at::ScalarType dtype = at::kInt) {
  check_quantize_args(quant_min, quant_max, dtype);
  check_quantize_per_channel_args(input_sizes, pre_scales, zero_points, axis);

  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input =
      at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(in_dtype));

  // Fill with a simple pattern: values from 0 to 1 in steps
  float step = 1.0f / (input.numel() - 1);
  auto flat_input = input.flatten();
  for (int i = 0; i < flat_input.numel(); i++) {
    flat_input[i] = i * step;
  }

  // Reshape back to original dimensions
  input = flat_input.reshape(input_sizes_int64);

  std::vector<float> scales = pre_scales;
  for (auto& s : scales) {
    s = s < eps ? eps : s;
  }

  // Create scale and zero_point tensors
  at::Tensor scale_tensor =
      at::tensor(scales, at::device(at::kCPU).dtype(at::kDouble));
  at::Tensor zero_point_tensor =
      at::tensor(zero_points, at::device(at::kCPU).dtype(at::kLong));

  // Get reference output
  at::Tensor my_ref = quantize_per_channel_reference_impl(
      input,
      scale_tensor,
      zero_point_tensor,
      axis,
      quant_min,
      quant_max,
      dtype);

  // Get implementation output
  at::Tensor cpu_ref = torch::executor::native::quantize_per_channel_aten(
      input,
      scale_tensor,
      zero_point_tensor,
      axis,
      quant_min,
      quant_max,
      dtype);

  // Get direct ATen implementation output
  c10::ScalarType aten_dtype = dtype;
  if (dtype == at::kChar) {
    aten_dtype = c10::kQInt8;
  } else if (dtype == at::kByte) {
    aten_dtype = c10::kQUInt8;
  }

  // Normalize axis for ATen (it doesn't handle negative values)
  int64_t normalized_axis = axis;
  if (normalized_axis < 0) {
    normalized_axis += input.dim();
  }

  at::Tensor aten_ref = at::quantize_per_channel(
      input, scale_tensor, zero_point_tensor, normalized_axis, aten_dtype);

  // Convert to int for consistent display regardless of underlying type
  at::Tensor my_ref_int = my_ref.to(at::kInt);
  at::Tensor cpu_ref_int = cpu_ref.to(at::kInt);
  // For quantized tensors, we need to use int_repr() to get the underlying
  // integer values
  at::Tensor aten_ref_int = aten_ref.int_repr().to(at::kInt);

  const bool output_correct = at::equal(my_ref_int, cpu_ref_int);
  if (!output_correct) {
    std::cout << "\n"
              << "Failed with parameters: " << std::endl;
    std::cout << "  axis: " << axis << std::endl;
    std::cout << "  input sizes:";
    for (size_t i = 0; i < input_sizes.size(); i++) {
      std::cout << " " << input_sizes[i] << " ";
    }
    std::cout << "" << std::endl;
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
    std::cout << "aten_ref:" << std::endl;
    std::cout << aten_ref_int << std::endl;
    std::cout << "cpu_ref:" << std::endl;
    std::cout << cpu_ref_int << std::endl;
    std::cout << "my_ref:" << std::endl;
    std::cout << my_ref_int << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

void test_vulkan_quantize_per_channel_impl(
    const std::vector<int>& input_sizes,
    const std::vector<float>& pre_scales,
    const std::vector<int>& zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype = at::kFloat,
    at::ScalarType dtype = at::kInt,
    const vkcompute::utils::StorageType in_storage =
        vkcompute::utils::kTexture3D,
    const vkcompute::utils::StorageType out_storage =
        vkcompute::utils::kTexture3D) {
  check_quantize_args(quant_min, quant_max, dtype);
  check_quantize_per_channel_args(input_sizes, pre_scales, zero_points, axis);

  std::vector<float> scales = pre_scales;
  for (auto& s : scales) {
    s = s < eps ? eps : s;
  }

  // Create input tensor with random values
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input =
      at::rand(input_sizes_int64, at::device(at::kCPU).dtype(in_dtype));
  at::Tensor scale_tensor =
      at::tensor(scales, at::device(at::kCPU).dtype(at::kDouble));
  at::Tensor zero_point_tensor =
      at::tensor(zero_points, at::device(at::kCPU).dtype(at::kLong));

  // Get reference output
  at::Tensor reference_out = torch::executor::native::quantize_per_channel_aten(
      input,
      scale_tensor,
      zero_point_tensor,
      axis,
      quant_min,
      quant_max,
      dtype);

  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(in_storage);
  ComputeGraph graph(config);

  IOValueRef r_input = graph.add_input_tensor(
      input.sizes().vec(), from_at_scalartype(input.scalar_type()), in_storage);
  IOValueRef r_scale = graph.add_input_tensor(
      scale_tensor.sizes().vec(),
      vkapi::kFloat,
      utils::kBuffer,
      utils::kWidthPacked);
  IOValueRef r_zero_point = graph.add_input_tensor(
      zero_point_tensor.sizes().vec(),
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);

  const ValueRef r_axis = graph.add_scalar<int64_t>(axis);
  const ValueRef r_quant_min = graph.add_scalar<int64_t>(quant_min);
  const ValueRef r_quant_max = graph.add_scalar<int64_t>(quant_max);

  const ValueRef r_out = graph.add_tensor(
      input.sizes().vec(), from_at_scalartype(dtype), out_storage);

  const ValueRef r_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(dtype));

  VK_GET_OP_FN("quantized_decomposed.quantize_per_channel.default")
  (graph,
   {
       r_input.value,
       r_scale.value,
       r_zero_point.value,
       r_axis,
       r_quant_min,
       r_quant_max,
       r_dtype,
       r_out,
   });

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.prepack();

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

  // Tolerance is 1 to address rounding errors and fp math differences between
  // CPU/GPU
  const bool output_correct =
      at::allclose(reference_int, vk_int, /*rtol=*/1, /*atol=*/1);
  if (!output_correct) {
    at::Tensor diffs = at::abs(reference_int - vk_int);

    std::cout << "\n"
              << "Failed with parameters: " << std::endl;
    std::cout << "  axis: " << axis << std::endl;
    std::cout << "  input sizes:";
    for (size_t i = 0; i < input_sizes.size(); i++) {
      std::cout << " " << input_sizes[i] << " ";
    }
    std::cout << "" << std::endl;
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

TEST(
    VulkanQuantizePerChannelTest,
    test_reference_quantize_per_channel_float_to_int8_3D_axis0) {
  std::vector<float> scales = {0.1, 0.2, 0.3};
  std::vector<int> zero_points = {0, 5, -2};

  test_reference_quantize_per_channel(
      {3, 4, 2}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);
}

TEST(
    VulkanQuantizePerChannelTest,
    test_reference_quantize_per_channel_float_to_int8_3D_axis2) {
  std::vector<float> scales = {0.1, 0.2};
  std::vector<int> zero_points = {0, 5};

  test_reference_quantize_per_channel(
      {3, 4, 2}, // input sizes
      scales,
      zero_points,
      2, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);
}

TEST(
    VulkanQuantizePerChannelTest,
    test_reference_quantize_per_channel_float_to_int8_3D_axisn1) {
  std::vector<float> scales = {0.1, 0.2};
  std::vector<int> zero_points = {0, 5};

  test_reference_quantize_per_channel(
      {3, 4, 2}, // input sizes
      scales,
      zero_points,
      -1, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);
}

TEST(
    VulkanQuantizePerChannelTest,
    test_reference_quantize_per_channel_float_to_int8_4D_axis0) {
  std::vector<float> scales = {0.1, 0.2, 0.00002};
  std::vector<int> zero_points = {0, 5, -4};

  test_reference_quantize_per_channel(
      {3, 4, 2, 5}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);
}

// END OF REFERENCE TESTS

TEST(
    VulkanQuantizePerChannelTest,
    test_vulkan_quantize_per_channel_float_to_int8_axis0) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales(9, 0.1f);
  std::vector<int> zero_points(9, 2);

  // 1D Tensor
  test_vulkan_quantize_per_channel(
      {9}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);

  // 2D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);

  // 3D Tensor
  test_vulkan_quantize_per_channel(
      {9, 7, 11}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 17, 5, 5}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);

  // 4D Tensor (negative axis)
  test_vulkan_quantize_per_channel(
      {5, 17, 5, 9}, // input sizes
      scales,
      zero_points,
      -1, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);
}

TEST(
    VulkanQuantizePerChannelTest,
    test_vulkan_quantize_per_channel_float_to_int8_axis1) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales(14, 0.001f);
  std::vector<int> zero_points(14, -5);

  // 2D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14}, // input sizes
      scales,
      zero_points,
      1, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);

  // 3D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14, 11}, // input sizes
      scales,
      zero_points,
      1, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14, 5, 5}, // input sizes
      scales,
      zero_points,
      1, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);

  // 4D Tensor (negative axis)
  test_vulkan_quantize_per_channel(
      {9, 7, 14, 5}, // input sizes
      scales,
      zero_points,
      -2, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);
}

TEST(
    VulkanQuantizePerChannelTest,
    test_vulkan_quantize_per_channel_float_to_int8_axis2) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales(11, 0.5f);
  std::vector<int> zero_points(11, 12);

  // 3D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14, 11}, // input sizes
      scales,
      zero_points,
      2, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14, 11, 5}, // input sizes
      scales,
      zero_points,
      2, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);

  // 4D Tensor (negative axis)
  test_vulkan_quantize_per_channel(
      {9, 11, 14, 5}, // input sizes
      scales,
      zero_points,
      -3, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);
}

TEST(
    VulkanQuantizePerChannelTest,
    test_vulkan_quantize_per_channel_float_to_int8_axis3) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales(7, 0.5f);
  std::vector<int> zero_points(7, 12);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      3, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);

  // 4D Tensor (negative axis)
  test_vulkan_quantize_per_channel(
      {7, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      -4, // axis
      -128, // quant_min
      127, // quant_max
      at::kFloat,
      at::kChar);
}

TEST(
    VulkanQuantizePerChannelTest,
    test_vulkan_quantize_per_channel_float_to_uint8_comprehensive) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales = {0.1, 0.2, 0.0001, 0.5, 0.02};
  std::vector<int> zero_points = {0, 5, -5, 1, 12};

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {5, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      0, // axis
      0, // quant_min
      255, // quant_max
      at::kFloat,
      at::kByte);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 5, 11, 7}, // input sizes
      scales,
      zero_points,
      1, // axis
      0, // quant_min
      255, // quant_max
      at::kFloat,
      at::kByte);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14, 5, 7}, // input sizes
      scales,
      zero_points,
      2, // axis
      0, // quant_min
      255, // quant_max
      at::kFloat,
      at::kByte);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14, 11, 5}, // input sizes
      scales,
      zero_points,
      3, // axis
      0, // quant_min
      255, // quant_max
      at::kFloat,
      at::kByte);

  // 4D Tensor (negative axis)
  test_vulkan_quantize_per_channel(
      {5, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      -4, // axis
      0, // quant_min
      255, // quant_max
      at::kFloat,
      at::kByte);
}

TEST(
    VulkanQuantizePerChannelTest,
    test_vulkan_quantize_per_channel_half_to_8bit) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_float16_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales = {0.1, 0.2, 0.01, 0.5, 0.02};
  std::vector<int> zero_points = {0, 5, 5, 1, 12};

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {5, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kHalf,
      at::kChar);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 5, 11, 7}, // input sizes
      scales,
      zero_points,
      1, // axis
      -128, // quant_min
      127, // quant_max
      at::kHalf,
      at::kChar);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14, 5, 7}, // input sizes
      scales,
      zero_points,
      2, // axis
      0, // quant_min
      255, // quant_max
      at::kHalf,
      at::kByte);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14, 11, 5}, // input sizes
      scales,
      zero_points,
      3, // axis
      -128, // quant_min
      127, // quant_max
      at::kHalf,
      at::kChar);

  // 4D Tensor (negative axis)
  test_vulkan_quantize_per_channel(
      {5, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      -4, // axis
      0, // quant_min
      255, // quant_max
      at::kHalf,
      at::kByte);
}

TEST(
    VulkanQuantizePerChannelTest,
    test_vulkan_quantize_per_channel_double_to_8bit) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales = {0.1, 0.2, 0.01, 0.5, 0.02};
  std::vector<int> zero_points = {0, 5, 5, 1, 12};

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {5, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kDouble,
      at::kChar);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 5, 11, 7}, // input sizes
      scales,
      zero_points,
      1, // axis
      -128, // quant_min
      127, // quant_max
      at::kDouble,
      at::kChar);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14, 5, 7}, // input sizes
      scales,
      zero_points,
      2, // axis
      0, // quant_min
      255, // quant_max
      at::kDouble,
      at::kByte);

  // 4D Tensor
  test_vulkan_quantize_per_channel(
      {9, 14, 11, 5}, // input sizes
      scales,
      zero_points,
      3, // axis
      -128, // quant_min
      127, // quant_max
      at::kDouble,
      at::kChar);

  // 4D Tensor (negative axis)
  test_vulkan_quantize_per_channel(
      {5, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      -4, // axis
      0, // quant_min
      255, // quant_max
      at::kDouble,
      at::kByte);
}

void test_vulkan_quantize_per_tensor_tensor_impl(
    const std::vector<int>& input_sizes,
    float scale,
    int zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype = at::kFloat,
    at::ScalarType dtype = at::kInt,
    const vkcompute::utils::StorageType in_storage =
        vkcompute::utils::kTexture3D,
    const vkcompute::utils::StorageType out_storage =
        vkcompute::utils::kTexture3D) {
  check_quantize_args(quant_min, quant_max, dtype);
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input =
      at::rand(input_sizes_int64, at::device(at::kCPU).dtype(in_dtype));

  scale = scale < eps ? eps : scale;

  // Create scale and zero_point as tensors (single element tensors)
  at::Tensor scale_tensor =
      at::tensor({scale}, at::device(at::kCPU).dtype(at::kDouble));
  at::Tensor zero_point_tensor =
      at::tensor({zero_point}, at::device(at::kCPU).dtype(at::kLong));

  // Get reference output using tensor variant
  at::Tensor reference_out =
      torch::executor::native::quantize_per_tensor_tensor_args_aten(
          input, scale_tensor, zero_point_tensor, quant_min, quant_max, dtype);

  // Build Vulkan quantize_per_tensor.tensor graph
  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(in_storage);
  ComputeGraph graph(config);

  IOValueRef r_input = graph.add_input_tensor(
      input.sizes().vec(), from_at_scalartype(input.scalar_type()), in_storage);

  // Add scale and zero_point as tensor inputs (buffer storage, width packed)
  IOValueRef r_scale = graph.add_input_tensor(
      scale_tensor.sizes().vec(),
      vkapi::kFloat,
      utils::kBuffer,
      utils::kWidthPacked);
  IOValueRef r_zero_point = graph.add_input_tensor(
      zero_point_tensor.sizes().vec(),
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);

  const ValueRef r_quant_min = graph.add_scalar<int64_t>(quant_min);
  const ValueRef r_quant_max = graph.add_scalar<int64_t>(quant_max);

  const ValueRef r_out = graph.add_tensor(
      input.sizes().vec(), from_at_scalartype(dtype), out_storage);

  const ValueRef r_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(dtype));

  VK_GET_OP_FN("quantized_decomposed.quantize_per_tensor.tensor")
  (graph,
   {
       r_input.value,
       r_scale.value,
       r_zero_point.value,
       r_quant_min,
       r_quant_max,
       r_dtype,
       r_out,
   });

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.prepack();

  // Run Vulkan quantize_per_tensor.tensor
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

  graph.execute();

  at::Tensor vk_out = at::empty_like(reference_out).contiguous();
  graph.copy_from_staging(
      staging_out, vk_out.mutable_data_ptr(), vk_out.numel());

  // Compare outputs
  // For quantized types, we need to compare the actual integer values
  at::Tensor reference_int = reference_out.to(at::kInt);
  at::Tensor vk_int = vk_out.to(at::kInt);

  // Tolerance is 1 to address rounding errors and fp math differences between
  // CPU/GPU
  const bool output_correct =
      at::allclose(reference_int, vk_int, /*rtol=*/1, /*atol=*/1);
  if (!output_correct) {
    at::Tensor diffs = at::abs(reference_int - vk_int);

    std::cout << "\n"
              << "Failed with parameters: " << std::endl;
    std::cout << "  scale: " << scale << std::endl;
    std::cout << "  zero_point: " << zero_point << std::endl;
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

TEST(
    VulkanQuantizePerTensorTensorTest,
    test_vulkan_quantize_per_tensor_tensor_float_to_int8) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  test_vulkan_quantize_per_tensor_tensor(
      {2, 3, 4}, // input sizes
      0.01, // scale
      1, // zero_point
      -128, // quant_min
      127, // quant_max
      at::kFloat, // input dtype
      at::kChar); // output dtype
}

TEST(
    VulkanQuantizePerTensorTensorTest,
    test_vulkan_quantize_per_tensor_tensor_float_to_uint8) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  test_vulkan_quantize_per_tensor_tensor(
      {2, 3, 4, 12}, // input sizes
      0.1, // scale
      5, // zero_point
      0, // quant_min
      255, // quant_max
      at::kFloat, // input dtype
      at::kByte); // output dtype
}

TEST(
    VulkanQuantizePerTensorTensorTest,
    test_vulkan_quantize_per_tensor_tensor_float_to_int32) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  test_vulkan_quantize_per_tensor_tensor(
      {2, 3}, // input sizes
      0.01, // scale
      12, // zero_point
      std::numeric_limits<int32_t>::min(), // quant_min
      std::numeric_limits<int32_t>::max(), // quant_max
      at::kFloat, // input dtype
      at::kInt); // output dtype
}

TEST(
    VulkanQuantizePerTensorTensorTest,
    test_vulkan_quantize_per_tensor_tensor_half_to_uint8) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  test_vulkan_quantize_per_tensor_tensor(
      {3, 4}, // input sizes
      0.3, // scale
      2, // zero_point
      0, // quant_min
      255, // quant_max
      at::kHalf, // input dtype
      at::kByte); // output dtype
}

TEST(
    VulkanQuantizePerTensorTensorTest,
    test_vulkan_quantize_per_tensor_tensor_double_to_int8) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  test_vulkan_quantize_per_tensor_tensor(
      {2, 3, 4}, // input sizes
      0.03, // scale
      -2, // zero_point
      -128, // quant_min
      127, // quant_max
      at::kDouble, // input dtype
      at::kChar); // output dtype
}
