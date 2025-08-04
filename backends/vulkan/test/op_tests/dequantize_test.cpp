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
#include <cstdint>
#include <iostream>
#include <limits>

namespace torch {
namespace executor {
namespace native {

// Forward declarations of the functions we're testing
Tensor& dequantize_per_tensor_out(
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    executorch::aten::optional<ScalarType> out_dtype,
    Tensor& out);

Tensor& dequantize_per_token_out(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    ScalarType out_dtype,
    Tensor& out);

Tensor& dequantize_per_channel_out(
    const Tensor& input,
    const Tensor& scale,
    const std::optional<Tensor>& zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    executorch::aten::optional<ScalarType> out_dtype,
    Tensor& out);

Tensor& dequantize_per_tensor_tensor_args_out(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    executorch::aten::optional<ScalarType> out_dtype,
    Tensor& out);

// Wrapper function for dequantize_per_tensor_out without context
Tensor& dequantize_per_tensor_out_no_context(
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    executorch::aten::optional<ScalarType> out_dtype,
    Tensor& out) {
  return torch::executor::native::dequantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, dtype, out_dtype, out);
}

// Wrapper function for dequantize_per_token_out without context
Tensor& dequantize_per_token_out_no_context(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    ScalarType out_dtype,
    Tensor& out) {
  return torch::executor::native::dequantize_per_token_out(
      input, scale, zero_points, quant_min, quant_max, dtype, out_dtype, out);
}

// Wrapper function for dequantize_per_channel_out without context
Tensor& dequantize_per_channel_out_no_context(
    const Tensor& input,
    const Tensor& scale,
    const std::optional<Tensor>& zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    executorch::aten::optional<ScalarType> out_dtype,
    Tensor& out) {
  return torch::executor::native::dequantize_per_channel_out(
      input,
      scale,
      zero_points,
      axis,
      quant_min,
      quant_max,
      dtype,
      out_dtype,
      out);
}

// Wrapper function for dequantize_per_tensor_tensor_args_out without context
Tensor& dequantize_per_tensor_tensor_args_out_no_context(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    executorch::aten::optional<ScalarType> out_dtype,
    Tensor& out) {
  return torch::executor::native::dequantize_per_tensor_tensor_args_out(
      input, scale, zero_point, quant_min, quant_max, dtype, out_dtype, out);
}

// ATen wrapper for dequantize_per_tensor
at::Tensor dequantize_per_tensor_aten(
    const at::Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  auto out = at::empty_like(input, out_dtype);
  // Convert at::ScalarType to executorch::ScalarType
  ScalarType et_dtype = at_scalartype_to_et_scalartype(dtype);
  ScalarType et_out_dtype = at_scalartype_to_et_scalartype(out_dtype);

  executorch::aten::optional<ScalarType> opt_et_out_dtype(et_out_dtype);

  WRAP_TO_ATEN(dequantize_per_tensor_out_no_context, 7)
  (input,
   scale,
   zero_point,
   quant_min,
   quant_max,
   et_dtype,
   opt_et_out_dtype,
   out);
  return out;
}

// ATen wrapper for dequantize_per_token
at::Tensor dequantize_per_token_aten(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  auto out = at::empty_like(input, out_dtype);
  // Convert at::ScalarType to executorch::ScalarType
  ScalarType et_dtype = at_scalartype_to_et_scalartype(dtype);
  ScalarType et_out_dtype = at_scalartype_to_et_scalartype(out_dtype);

  WRAP_TO_ATEN(dequantize_per_token_out_no_context, 7)
  (input,
   scale,
   zero_points,
   quant_min,
   quant_max,
   et_dtype,
   et_out_dtype,
   out);
  return out;
}

// ATen wrapper for dequantize_per_channel
at::Tensor dequantize_per_channel_aten(
    const at::Tensor& input,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  auto out = at::empty_like(input, out_dtype);
  // Convert at::ScalarType to executorch::ScalarType
  ScalarType et_dtype = at_scalartype_to_et_scalartype(dtype);
  ScalarType et_out_dtype = at_scalartype_to_et_scalartype(out_dtype);

  executorch::aten::optional<ScalarType> opt_et_out_dtype(et_out_dtype);

  WRAP_TO_ATEN(dequantize_per_channel_out_no_context, 8)
  (input,
   scale,
   zero_points,
   axis,
   quant_min,
   quant_max,
   et_dtype,
   opt_et_out_dtype,
   out);
  return out;
}

// ATen wrapper for dequantize_per_tensor with tensor args
at::Tensor dequantize_per_tensor_tensor_args_aten(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  auto out = at::empty_like(input, out_dtype);
  // Convert at::ScalarType to executorch::ScalarType
  ScalarType et_dtype = at_scalartype_to_et_scalartype(dtype);
  ScalarType et_out_dtype = at_scalartype_to_et_scalartype(out_dtype);

  executorch::aten::optional<ScalarType> opt_et_out_dtype(et_out_dtype);

  WRAP_TO_ATEN(dequantize_per_tensor_tensor_args_out_no_context, 7)
  (input,
   scale,
   zero_point,
   quant_min,
   quant_max,
   et_dtype,
   opt_et_out_dtype,
   out);
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

void check_dequantize_args(
    int64_t quant_min,
    int64_t quant_max,
    c10::ScalarType in_dtype,
    c10::ScalarType out_dtype) {
  using namespace vkcompute;

  // Check that quant_min <= quant_max
  VK_CHECK_COND(
      quant_min <= quant_max,
      "quant_min must be <= quant_max, got quant_min: ",
      quant_min,
      " quant_max: ",
      quant_max);

  // Check that input dtype is a quantized type
  switch (in_dtype) {
    case c10::kByte:
    case c10::kChar:
    case c10::kShort:
    case c10::kInt:
    case c10::kLong:
      break;
    default:
      VK_THROW(
          "Unsupported input dtype: ",
          scalar_type_name(in_dtype),
          " (",
          static_cast<int>(in_dtype),
          ")");
  }

  // Check that output dtype is a floating point type
  switch (out_dtype) {
    case c10::kHalf:
    case c10::kFloat:
    case c10::kDouble:
      break;
    default:
      VK_THROW(
          "Unsupported output dtype: ",
          scalar_type_name(out_dtype),
          " (",
          static_cast<int>(out_dtype),
          ")");
  }
}

/**
 * Helper function to validate dequantize_per_channel arguments
 * Similar to the validation in quantize_test.cpp
 */
void check_dequantize_per_channel_args(
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
 * Reference implementation of dequantize_per_tensor
 */
at::Tensor dequantize_per_tensor_reference_impl(
    const at::Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  // Create output tensor with the target dtype
  at::Tensor out = at::empty_like(input, out_dtype);

  // Dequantize the input tensor
  at::Tensor flat_input = input.flatten();
  at::Tensor flat_out = out.flatten();

  // Store casted values to avoid repeated casting
  const int32_t zero_point_int32 = static_cast<int32_t>(zero_point);
  const float scale_float = static_cast<float>(scale);

  for (int i = 0; i < flat_input.numel(); i++) {
    double dequantized_value = 0.0;

    // Extract quantized value and dequantize based on input dtype
    // Following the CPU implementation pattern: (input - zero_point) * scale
    if (dtype == at::kByte) {
      uint8_t qvalue = flat_input[i].item<uint8_t>();
      dequantized_value = (qvalue - zero_point_int32) * scale_float;
    } else if (dtype == at::kChar) {
      int8_t qvalue = flat_input[i].item<int8_t>();
      dequantized_value = (qvalue - zero_point_int32) * scale_float;
    } else if (dtype == at::kShort) {
      int16_t qvalue = flat_input[i].item<int16_t>();
      dequantized_value = (qvalue - zero_point_int32) * scale_float;
    } else if (dtype == at::kInt) {
      int32_t qvalue = flat_input[i].item<int32_t>();
      dequantized_value = (qvalue - zero_point_int32) * scale_float;
    } else if (dtype == at::kLong) {
      int64_t qvalue = flat_input[i].item<int64_t>();
      dequantized_value = (qvalue - zero_point_int32) * scale_float;
    }

    // Store result based on output dtype
    if (out_dtype == at::kFloat) {
      flat_out[i] = static_cast<float>(dequantized_value);
    } else if (out_dtype == at::kDouble) {
      flat_out[i] = dequantized_value;
    } else if (out_dtype == at::kHalf) {
      flat_out[i] = static_cast<c10::Half>(dequantized_value);
    }
  }

  return out.reshape(input.sizes());
}

/*
 * Reference implementation of dequantize_per_token
 */
at::Tensor dequantize_per_token_reference_impl(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  // Create output tensor with the target dtype
  at::Tensor out = at::empty_like(input, out_dtype);

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

  // Dequantize each token separately
  for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
    // Get scale and zero_point for this token
    float token_scale = scale[token_idx].item<float>();
    int64_t token_zero_point = zero_point[token_idx].item<int64_t>();

    // Store casted values to avoid repeated casting
    const int32_t token_zero_point_int32 =
        static_cast<int32_t>(token_zero_point);

    // Dequantize the token
    for (int i = 0; i < input.size(-1); i++) {
      double dequantized_value = 0.0;

      // Extract quantized value and dequantize based on input dtype
      // Following the CPU implementation pattern: (input - zero_point) * scale
      if (dtype == at::kByte) {
        uint8_t qvalue = reshaped_input[token_idx][i].item<uint8_t>();
        dequantized_value = (qvalue - token_zero_point_int32) * token_scale;
      } else if (dtype == at::kChar) {
        int8_t qvalue = reshaped_input[token_idx][i].item<int8_t>();
        dequantized_value = (qvalue - token_zero_point_int32) * token_scale;
      } else if (dtype == at::kShort) {
        int16_t qvalue = reshaped_input[token_idx][i].item<int16_t>();
        dequantized_value = (qvalue - token_zero_point_int32) * token_scale;
      } else if (dtype == at::kInt) {
        int32_t qvalue = reshaped_input[token_idx][i].item<int32_t>();
        dequantized_value = (qvalue - token_zero_point_int32) * token_scale;
      } else if (dtype == at::kLong) {
        int64_t qvalue = reshaped_input[token_idx][i].item<int64_t>();
        dequantized_value = (qvalue - token_zero_point_int32) * token_scale;
      } else {
        throw std::runtime_error("Unsupported input dtype");
      }

      // Store result based on output dtype
      if (out_dtype == at::kFloat) {
        reshaped_out[token_idx][i] = static_cast<float>(dequantized_value);
      } else if (out_dtype == at::kDouble) {
        reshaped_out[token_idx][i] = dequantized_value;
      } else if (out_dtype == at::kHalf) {
        reshaped_out[token_idx][i] = static_cast<c10::Half>(dequantized_value);
      }
    }
  }

  return out;
}

/*
 * Reference implementation of dequantize_per_channel
 */
at::Tensor dequantize_per_channel_reference_impl(
    const at::Tensor& input,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  // Normalize axis to handle negative values
  int64_t normalized_axis = axis;
  if (normalized_axis < 0) {
    normalized_axis += input.dim();
  }

  // Create output tensor with the same shape as input but with target dtype
  at::Tensor output = at::empty_like(input, out_dtype);

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
  const double* scale_data = scale.const_data_ptr<double>();
  const int64_t* zero_point_data = nullptr;
  if (zero_point.has_value()) {
    zero_point_data = zero_point.value().const_data_ptr<int64_t>();
  }

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
    int64_t channel_zero_point = 0;
    if (zero_point_data != nullptr) {
      channel_zero_point = zero_point_data[channel_idx];
    }

    // Store casted values to avoid repeated casting
    const int32_t channel_zero_point_int32 =
        static_cast<int32_t>(channel_zero_point);
    const float channel_scale_float = static_cast<float>(channel_scale);

    // Get the input value and dequantize
    double dequantized_value = 0.0;

    // Extract quantized value and dequantize based on input dtype
    // Following the CPU implementation pattern: (input - zero_point) * scale
    if (dtype == at::kByte) {
      uint8_t qvalue = input.flatten()[flat_idx].item<uint8_t>();
      dequantized_value =
          (qvalue - channel_zero_point_int32) * channel_scale_float;
    } else if (dtype == at::kChar) {
      int8_t qvalue = input.flatten()[flat_idx].item<int8_t>();
      dequantized_value =
          (qvalue - channel_zero_point_int32) * channel_scale_float;
    } else if (dtype == at::kShort) {
      int16_t qvalue = input.flatten()[flat_idx].item<int16_t>();
      dequantized_value =
          (qvalue - channel_zero_point_int32) * channel_scale_float;
    } else if (dtype == at::kInt) {
      int32_t qvalue = input.flatten()[flat_idx].item<int32_t>();
      dequantized_value =
          (qvalue - channel_zero_point_int32) * channel_scale_float;
    } else if (dtype == at::kLong) {
      int64_t qvalue = input.flatten()[flat_idx].item<int64_t>();
      dequantized_value =
          (qvalue - channel_zero_point_int32) * channel_scale_float;
    } else {
      throw std::runtime_error("Unsupported input dtype");
    }

    // Store the result based on output dtype
    if (out_dtype == at::kFloat) {
      output.flatten()[flat_idx] = static_cast<float>(dequantized_value);
    } else if (out_dtype == at::kDouble) {
      output.flatten()[flat_idx] = dequantized_value;
    } else if (out_dtype == at::kHalf) {
      output.flatten()[flat_idx] = static_cast<c10::Half>(dequantized_value);
    }
  }

  return output;
}

// Forward declaration of implementation functions
void test_vulkan_dequantize_per_token_impl(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype,
    const vkcompute::utils::StorageType in_storage,
    const vkcompute::utils::StorageType out_storage);

void test_vulkan_dequantize_per_channel_impl(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype,
    const vkcompute::utils::StorageType in_storage,
    const vkcompute::utils::StorageType out_storage);

void test_vulkan_dequantize_per_tensor_tensor_impl(
    const std::vector<int>& input_sizes,
    float scale,
    int zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype,
    const vkcompute::utils::StorageType in_storage,
    const vkcompute::utils::StorageType out_storage);

// Wrapper function to test both buffer and texture storage types
void test_vulkan_dequantize_per_token(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  // Test with buffer storage
  test_vulkan_dequantize_per_token_impl(
      input_sizes,
      scales,
      zero_points,
      quant_min,
      quant_max,
      dtype,
      out_dtype,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  // Telling the system to expect a float instead of a double
  // since the shader can only return 32bit anyways
  if (out_dtype == at::kDouble) {
    out_dtype = at::kFloat;
  }

  // Test with texture storage
  test_vulkan_dequantize_per_token_impl(
      input_sizes,
      scales,
      zero_points,
      quant_min,
      quant_max,
      dtype,
      out_dtype,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

// Wrapper function to test both buffer and texture storage types
void test_vulkan_dequantize_per_channel(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  // Test with buffer storage
  test_vulkan_dequantize_per_channel_impl(
      input_sizes,
      scales,
      zero_points,
      axis,
      quant_min,
      quant_max,
      dtype,
      out_dtype,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  // Telling the system to expect a float instead of a double
  // since the shader can only return 32bit anyways
  if (out_dtype == at::kDouble) {
    out_dtype = at::kFloat;
  }

  // Test with texture storage
  test_vulkan_dequantize_per_channel_impl(
      input_sizes,
      scales,
      zero_points,
      axis,
      quant_min,
      quant_max,
      dtype,
      out_dtype,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

// Wrapper function to test both buffer and texture storage types
void test_vulkan_dequantize_per_tensor_tensor(
    const std::vector<int>& input_sizes,
    float scale,
    int zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  // Test with buffer storage
  test_vulkan_dequantize_per_tensor_tensor_impl(
      input_sizes,
      scale,
      zero_point,
      quant_min,
      quant_max,
      dtype,
      out_dtype,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  // Telling the system to expect a float instead of a double
  // since the shader can only return 32bit anyways
  if (out_dtype == at::kDouble) {
    out_dtype = at::kFloat;
  }

  // Test with texture storage
  test_vulkan_dequantize_per_tensor_tensor_impl(
      input_sizes,
      scale,
      zero_point,
      quant_min,
      quant_max,
      dtype,
      out_dtype,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

void test_reference_dequantize_per_tensor(
    const std::vector<int>& input_sizes,
    float scale,
    int zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  check_dequantize_args(quant_min, quant_max, dtype, out_dtype);
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());

  // Create a quantized input tensor with values from quant_min to quant_max
  at::Tensor input;
  if (dtype == at::kByte) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kByte));
  } else if (dtype == at::kChar) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kChar));
  } else if (dtype == at::kShort) {
    input =
        at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kShort));
  } else if (dtype == at::kInt) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kInt));
  } else {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kLong));
  }

  // Fill with a simple pattern: values from quant_min to quant_max in steps
  float step = 1.0f;
  if (input.numel() > 1) {
    step = static_cast<float>(quant_max - quant_min) / (input.numel() - 1);
  }

  auto flat_input = input.flatten();
  for (int i = 0; i < flat_input.numel(); i++) {
    int64_t qvalue = quant_min + i * step;
    if (dtype == at::kByte) {
      flat_input[i] = static_cast<uint8_t>(qvalue);
    } else if (dtype == at::kChar) {
      flat_input[i] = static_cast<int8_t>(qvalue);
    } else if (dtype == at::kShort) {
      flat_input[i] = static_cast<int16_t>(qvalue);
    } else if (dtype == at::kInt) {
      flat_input[i] = static_cast<int32_t>(qvalue);
    } else if (dtype == at::kLong) {
      flat_input[i] = static_cast<int64_t>(qvalue);
    }
  }

  // Reshape back to original dimensions
  input = flat_input.reshape(input_sizes_int64);

  // Get reference output
  at::Tensor reference_out = dequantize_per_tensor_reference_impl(
      input, scale, zero_point, quant_min, quant_max, dtype, out_dtype);

  // Get implementation output
  at::Tensor impl_out = torch::executor::native::dequantize_per_tensor_aten(
      input, scale, zero_point, quant_min, quant_max, dtype, out_dtype);

  // Compare outputs
  const bool output_correct = at::allclose(reference_out, impl_out);
  if (!output_correct) {
    std::cout << "\n"
              << "Failed with parameters: " << std::endl;
    std::cout << "  scale: " << scale << std::endl;
    std::cout << "  zero_point: " << zero_point << std::endl;
    std::cout << "  quant_min: " << quant_min << std::endl;
    std::cout << "  quant_max: " << quant_max << std::endl;
    std::cout << "  input dtype: " << dtype << std::endl;
    std::cout << "  output dtype: " << out_dtype << std::endl;

    std::cout << "input:" << std::endl;
    std::cout << input << std::endl;
    std::cout << "reference:" << std::endl;
    std::cout << reference_out << std::endl;
    std::cout << "implementation:" << std::endl;
    std::cout << impl_out << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

TEST(
    VulkanDequantizePerTensorTest,
    test_reference_dequantize_per_tensor_uint8_to_float) {
  test_reference_dequantize_per_tensor(
      {2, 3, 4}, // input sizes
      0.1, // scale
      5, // zero_point
      0, // quant_min
      255, // quant_max
      at::kByte, // input dtype
      at::kFloat); // output dtype
}

TEST(
    VulkanDequantizePerTensorTest,
    test_reference_dequantize_per_tensor_int8_to_float) {
  test_reference_dequantize_per_tensor(
      {3, 4, 5}, // input sizes
      0.05, // scale
      0, // zero_point
      -128, // quant_min
      127, // quant_max
      at::kChar, // input dtype
      at::kFloat); // output dtype
}

TEST(
    VulkanDequantizePerTensorTest,
    test_reference_dequantize_per_tensor_int32_to_float) {
  test_reference_dequantize_per_tensor(
      {4, 6, 2}, // input sizes
      0.2, // scale
      2, // zero_point
      std::numeric_limits<int32_t>::min(), // quant_min
      std::numeric_limits<int32_t>::max(), // quant_max
      at::kInt, // input dtype
      at::kFloat); // output dtype
}

TEST(
    VulkanDequantizePerTensorTest,
    test_reference_dequantize_per_tensor_uint8_to_half) {
  test_reference_dequantize_per_tensor(
      {7, 4}, // input sizes
      0.1, // scale
      10, // zero_point
      0, // quant_min
      255, // quant_max
      at::kByte, // input dtype (uint8)
      at::kHalf); // output dtype
}

TEST(
    VulkanDequantizePerTensorTest,
    test_reference_dequantize_per_tensor_int32_to_half) {
  test_reference_dequantize_per_tensor(
      {2, 6, 5}, // input sizes
      0.3, // scale
      -10, // zero_point
      std::numeric_limits<int32_t>::min(), // quant_min
      std::numeric_limits<int32_t>::max(), // quant_max
      at::kInt, // input dtype
      at::kHalf); // output dtype
}

// No Vulkan tests for quantized_decomposed.dequantize_per_tensor.default
// because it is not going to be implemented in Vulkan since we will
// be handling any future calls to this op via the export stage

void test_reference_dequantize_per_token(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  check_dequantize_args(quant_min, quant_max, dtype, out_dtype);
  int num_tokens = 1;
  for (int i = 0; i < input_sizes.size() - 1; i++) {
    num_tokens *= input_sizes[i];
  }

  ASSERT_EQ(num_tokens, scales.size());
  ASSERT_EQ(num_tokens, zero_points.size());

  // Create input tensor with quantized values
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input;
  if (dtype == at::kByte) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kByte));
  } else if (dtype == at::kChar) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kChar));
  } else if (dtype == at::kShort) {
    input =
        at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kShort));
  } else if (dtype == at::kInt) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kInt));
  } else {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kLong));
  }

  // Fill with a simple pattern: values from quant_min to quant_max in steps
  at::Tensor reshaped_input = input.reshape({num_tokens, input.size(-1)});
  for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
    float step = 1.0f;
    if (input.size(-1) > 1) {
      step = static_cast<float>(quant_max - quant_min) / (input.size(-1) - 1);
    }

    for (int i = 0; i < input.size(-1); i++) {
      int64_t qvalue = quant_min + i * step;
      if (dtype == at::kByte) {
        reshaped_input[token_idx][i] = static_cast<uint8_t>(qvalue);
      } else if (dtype == at::kChar) {
        reshaped_input[token_idx][i] = static_cast<int8_t>(qvalue);
      } else if (dtype == at::kShort) {
        reshaped_input[token_idx][i] = static_cast<int16_t>(qvalue);
      } else if (dtype == at::kInt) {
        reshaped_input[token_idx][i] = static_cast<int32_t>(qvalue);
      } else if (dtype == at::kLong) {
        reshaped_input[token_idx][i] = static_cast<int64_t>(qvalue);
      }
    }
  }

  // Reshape back to original dimensions
  input = reshaped_input.reshape(input_sizes_int64);

  // Create scale and zero_point tensors
  at::Tensor scale_tensor =
      at::tensor(scales, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor zero_point_tensor =
      at::tensor(zero_points, at::device(at::kCPU).dtype(at::kLong));

  // Get reference output
  at::Tensor reference_out = dequantize_per_token_reference_impl(
      input,
      scale_tensor,
      zero_point_tensor,
      quant_min,
      quant_max,
      dtype,
      out_dtype);

  // Get implementation output
  at::Tensor impl_out = torch::executor::native::dequantize_per_token_aten(
      input,
      scale_tensor,
      zero_point_tensor,
      quant_min,
      quant_max,
      dtype,
      out_dtype);

  // Compare outputs
  const bool output_correct = at::allclose(reference_out, impl_out);
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
    std::cout << "  input dtype: " << dtype << std::endl;
    std::cout << "  output dtype: " << out_dtype << std::endl;

    std::cout << "input:" << std::endl;
    std::cout << input << std::endl;
    std::cout << "reference:" << std::endl;
    std::cout << reference_out << std::endl;
    std::cout << "implementation:" << std::endl;
    std::cout << impl_out << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

void test_vulkan_dequantize_per_token_impl(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype,
    const vkcompute::utils::StorageType in_storage,
    const vkcompute::utils::StorageType out_storage) {
  check_dequantize_args(quant_min, quant_max, dtype, out_dtype);
  int num_tokens = 1;
  for (int i = 0; i < input_sizes.size() - 1; i++) {
    num_tokens *= input_sizes[i];
  }

  ASSERT_EQ(num_tokens, scales.size());
  ASSERT_EQ(num_tokens, zero_points.size());

  // Create input tensor with quantized values
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input;
  if (dtype == at::kByte) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kByte));
  } else if (dtype == at::kChar) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kChar));
  } else if (dtype == at::kShort) {
    input =
        at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kShort));
  } else if (dtype == at::kInt) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kInt));
  } else {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kLong));
  }

  // Fill with a simple pattern: values from quant_min to quant_max in steps
  at::Tensor reshaped_input = input.reshape({num_tokens, input.size(-1)});
  for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
    float step = 1.0f;
    if (input.size(-1) > 1) {
      step = static_cast<float>(quant_max - quant_min) / (input.size(-1) - 1);
    }

    for (int i = 0; i < input.size(-1); i++) {
      int64_t qvalue = quant_min + i * step;
      if (dtype == at::kByte) {
        reshaped_input[token_idx][i] = static_cast<uint8_t>(qvalue);
      } else if (dtype == at::kChar) {
        reshaped_input[token_idx][i] = static_cast<int8_t>(qvalue);
      } else if (dtype == at::kShort) {
        reshaped_input[token_idx][i] = static_cast<int16_t>(qvalue);
      } else if (dtype == at::kInt) {
        reshaped_input[token_idx][i] = static_cast<int32_t>(qvalue);
      } else if (dtype == at::kLong) {
        reshaped_input[token_idx][i] = static_cast<int64_t>(qvalue);
      }
    }
  }

  // Reshape back to original dimensions
  input = reshaped_input.reshape(input_sizes_int64);

  // Create scale and zero_point tensors
  at::Tensor scale_tensor =
      at::tensor(scales, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor zero_point_tensor =
      at::tensor(zero_points, at::device(at::kCPU).dtype(at::kLong));

  // Get reference output
  at::Tensor reference_out = torch::executor::native::dequantize_per_token_aten(
      input,
      scale_tensor,
      zero_point_tensor,
      quant_min,
      quant_max,
      dtype,
      out_dtype);

  // Build Vulkan dequantize_per_token graph
  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(in_storage);
  ComputeGraph graph(config);

  IOValueRef r_input = graph.add_input_tensor(
      input.sizes().vec(), from_at_scalartype(dtype), in_storage);
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
      input.sizes().vec(), from_at_scalartype(out_dtype), out_storage);

  const ValueRef r_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(out_dtype));

  VK_GET_OP_FN("quantized_decomposed.dequantize_per_token.default")
  (graph,
   {
       r_input.value,
       r_scale.value,
       r_zero_point.value,
       r_quant_min,
       r_quant_max,
       r_dtype,
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

  // Compare outputs with appropriate tolerance for half precision
  bool output_correct;
  if (out_dtype == at::kHalf) {
    // Use higher tolerance for half precision due to limited precision
    output_correct =
        at::allclose(reference_out, vk_out, /*rtol=*/1e-2, /*atol=*/1e-2);
  } else {
    output_correct =
        at::allclose(reference_out, vk_out, /*rtol=*/1e-5, /*atol=*/1e-5);
  }
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
    std::cout << "  storage type: "
              << (in_storage == vkcompute::utils::kBuffer ? "buffer"
                                                          : "texture")
              << std::endl;
    std::cout << "  input dtype: " << dtype << std::endl;
    std::cout << "  output dtype: " << out_dtype << std::endl;

    std::cout << "input:" << std::endl;
    std::cout << input << std::endl;
    std::cout << "reference:" << std::endl;
    std::cout << reference_out << std::endl;
    std::cout << "vulkan:" << std::endl;
    std::cout << vk_out << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

TEST(
    VulkanDequantizePerTokenTest,
    test_reference_dequantize_per_token_uint8_to_float) {
  std::vector<float> scales = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<int> zero_points = {5, 10, 15, 20, 25, 30};

  test_reference_dequantize_per_token(
      {2, 3, 4}, // input sizes (2*3=6 tokens)
      scales,
      zero_points,
      0, // quant_min
      255, // quant_max
      at::kByte, // input dtype
      at::kFloat); // output dtype
}

TEST(
    VulkanDequantizePerTokenTest,
    test_reference_dequantize_per_token_int8_to_float) {
  std::vector<float> scales = {0.05, 0.1, 0.15, 0.2};
  std::vector<int> zero_points = {0, -5, 5, 10};

  test_reference_dequantize_per_token(
      {2, 2, 5}, // input sizes (2*2=4 tokens)
      scales,
      zero_points,
      -128, // quant_min
      127, // quant_max
      at::kChar, // input dtype
      at::kFloat); // output dtype
}

TEST(
    VulkanDequantizePerTokenTest,
    test_reference_dequantize_per_token_int32_to_float) {
  std::vector<float> scales = {0.05, 0.1, 0.15, 0.2};
  std::vector<int> zero_points = {0, -5, 5, 10};

  test_reference_dequantize_per_token(
      {2, 2, 10}, // input sizes (2*2=4 tokens)
      scales,
      zero_points,
      std::numeric_limits<int32_t>::min(), // quant_min
      std::numeric_limits<int32_t>::max(), // quant_max
      at::kInt, // input dtype
      at::kFloat); // output dtype
}

TEST(
    VulkanDequantizePerTokenTest,
    test_reference_dequantize_per_token_int8_to_half) {
  std::vector<float> scales = {0.05, 0.1, 0.15, 0.2};
  std::vector<int> zero_points = {0, -5, 5, 10};

  test_reference_dequantize_per_token(
      {4, 1, 5}, // input sizes (4*1=4 tokens)
      scales,
      zero_points,
      -128, // quant_min
      127, // quant_max
      at::kChar, // input dtype (int8)
      at::kHalf); // output dtype
}

TEST(
    VulkanDequantizePerTokenTest,
    test_reference_dequantize_per_token_int32_to_half) {
  std::vector<float> scales = {0.05, 0.1};
  std::vector<int> zero_points = {0, -5};

  test_reference_dequantize_per_token(
      {2, 2}, // input sizes (2 tokens)
      scales,
      zero_points,
      std::numeric_limits<int32_t>::min(), // quant_min
      std::numeric_limits<int32_t>::max(), // quant_max
      at::kInt, // input dtype
      at::kHalf); // output dtype
}

TEST(
    VulkanDequantizePerTokenTest,
    test_vulkan_dequantize_per_token_uint8_to_float) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<int> zero_points = {5, 10, 15, 20, 25, 30};

  test_vulkan_dequantize_per_token(
      {2, 3, 6}, // input sizes (2*3=6 tokens)
      scales,
      zero_points,
      0, // quant_min
      255, // quant_max
      at::kByte, // input dtype
      at::kFloat); // output dtype
}

TEST(
    VulkanDequantizePerTokenTest,
    test_vulkan_dequantize_per_token_int8_to_float) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales = {0.05, 0.0};
  std::vector<int> zero_points = {10, -5};

  test_vulkan_dequantize_per_token(
      {2, 2}, // input sizes (2*2=4 tokens)
      scales,
      zero_points,
      -128, // quant_min
      127, // quant_max
      at::kChar, // input dtype
      at::kFloat); // output dtype
}

TEST(
    VulkanDequantizePerTokenTest,
    test_vulkan_dequantize_per_token_int32_to_float) {
  std::vector<float> scales = {
      0.0001, 0.0002, 0.0003, 0.0, 0.0011, 0.0102, 0.1003, 0.0};
  std::vector<int> zero_points = {100, -100, 50, -50, 12, -6, 4, -24};

  test_vulkan_dequantize_per_token(
      {2, 2, 2, 12}, // input sizes (2*2=4 tokens)
      scales,
      zero_points,
      -2147483648, // quant_min
      2147483647, // quant_max
      at::kInt, // input dtype
      at::kFloat); // output dtype
}

TEST(
    VulkanDequantizePerTokenTest,
    test_vulkan_dequantize_per_token_int8_to_half) {
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
  std::vector<float> scales = {0.05, 0.2};
  std::vector<int> zero_points = {2, -5};

  test_vulkan_dequantize_per_token(
      {2, 2}, // input sizes (2=4 tokens)
      scales,
      zero_points,
      -128, // quant_min
      127, // quant_max
      at::kChar, // input dtype
      at::kHalf); // output dtype
}

TEST(
    VulkanDequantizePerTokenTest,
    test_vulkan_dequantize_per_token_int32_to_half) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_float16_buffers_support()) {
    GTEST_SKIP();
  }
  // Use much smaller scales to avoid overflow to infinity in half precision
  // Half precision max value is ~65504, so with int32 values around 2e9,
  // we need scales smaller than 65504/2e9 ≈ 3e-5 to avoid overflow
  std::vector<float> scales = {1e-5, 2e-5, 1.5e-5};
  std::vector<int> zero_points = {20, -15, 1};

  test_vulkan_dequantize_per_token(
      {3, 6}, // input sizes (3 tokens)
      scales,
      zero_points,
      std::numeric_limits<int32_t>::min(), // quant_min
      std::numeric_limits<int32_t>::max(), // quant_max
      at::kInt, // input dtype
      at::kHalf); // output dtype
}

TEST(
    VulkanDequantizePerTokenTest,
    test_vulkan_dequantize_per_token_int8_to_double) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales = {0.05, 0.001};
  std::vector<int> zero_points = {10, -5};

  test_vulkan_dequantize_per_token(
      {2, 2}, // input sizes (2 tokens)
      scales,
      zero_points,
      -128, // quant_min
      127, // quant_max
      at::kChar, // input dtype
      at::kDouble); // output dtype
}

void test_reference_dequantize_per_channel(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  check_dequantize_args(quant_min, quant_max, dtype, out_dtype);
  check_dequantize_per_channel_args(input_sizes, scales, zero_points, axis);

  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());

  // Create input tensor with quantized values
  at::Tensor input;
  if (dtype == at::kByte) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kByte));
  } else if (dtype == at::kChar) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kChar));
  } else if (dtype == at::kShort) {
    input =
        at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kShort));
  } else if (dtype == at::kInt) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kInt));
  } else {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kLong));
  }

  // Fill with a simple pattern: values from quant_min to quant_max in steps
  float step = 1.0f;
  if (input.numel() > 1) {
    step = static_cast<float>(quant_max - quant_min) / (input.numel() - 1);
  }

  auto flat_input = input.flatten();
  for (int i = 0; i < flat_input.numel(); i++) {
    int64_t qvalue = quant_min + i * step;
    if (dtype == at::kByte) {
      flat_input[i] = static_cast<uint8_t>(qvalue);
    } else if (dtype == at::kChar) {
      flat_input[i] = static_cast<int8_t>(qvalue);
    } else if (dtype == at::kShort) {
      flat_input[i] = static_cast<int16_t>(qvalue);
    } else if (dtype == at::kInt) {
      flat_input[i] = static_cast<int32_t>(qvalue);
    } else if (dtype == at::kLong) {
      flat_input[i] = static_cast<int64_t>(qvalue);
    }
  }

  // Reshape back to original dimensions
  input = flat_input.reshape(input_sizes_int64);

  // Create scale and zero_point tensors
  at::Tensor scale_tensor =
      at::tensor(scales, at::device(at::kCPU).dtype(at::kDouble));
  at::Tensor zero_point_tensor =
      at::tensor(zero_points, at::device(at::kCPU).dtype(at::kLong));

  // Get reference output
  at::Tensor my_ref = dequantize_per_channel_reference_impl(
      input,
      scale_tensor,
      zero_point_tensor,
      axis,
      quant_min,
      quant_max,
      dtype,
      out_dtype);

  // Get implementation output
  at::Tensor cpu_ref = torch::executor::native::dequantize_per_channel_aten(
      input,
      scale_tensor,
      zero_point_tensor,
      axis,
      quant_min,
      quant_max,
      dtype,
      out_dtype);

  // Compare outputs
  const bool output_correct = at::allclose(my_ref, cpu_ref);
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
    std::cout << "  input dtype: " << dtype << std::endl;
    std::cout << "  output dtype: " << out_dtype << std::endl;

    std::cout << "input:" << std::endl;
    std::cout << input << std::endl;
    std::cout << "cpu_ref:" << std::endl;
    std::cout << cpu_ref << std::endl;
    std::cout << "my_ref:" << std::endl;
    std::cout << my_ref << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

void test_vulkan_dequantize_per_channel_impl(
    const std::vector<int>& input_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype,
    const vkcompute::utils::StorageType in_storage,
    const vkcompute::utils::StorageType out_storage) {
  check_dequantize_args(quant_min, quant_max, dtype, out_dtype);
  check_dequantize_per_channel_args(input_sizes, scales, zero_points, axis);

  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());

  // Create random float tensor
  at::Tensor float_x =
      at::rand(input_sizes_int64, at::device(at::kCPU).dtype(at::kFloat));

  // Create scale and zero_point tensors
  at::Tensor scale_tensor =
      at::tensor(scales, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor zero_point_tensor =
      at::tensor(zero_points, at::device(at::kCPU).dtype(at::kInt));

  // Map the dtype to the corresponding quantized type and quantize the float
  // tensor
  c10::ScalarType qtype;
  at::Tensor adjusted_zero_points = zero_point_tensor;

  if (dtype == at::kByte) {
    qtype = c10::kQUInt8;
    // ATEN ONLY: Adjust zero points for unsigned types (must be non-negative)
    adjusted_zero_points = at::clamp_min(zero_point_tensor, 0);
  } else if (dtype == at::kChar) {
    qtype = c10::kQInt8;
  } else if (dtype == at::kInt) {
    qtype = c10::kQInt32;
  } else {
    std::cout << "invalid dtype for ATEN: " << dtype << std::endl;
    std::cout << " --> Delegating to c10::kQInt32" << std::endl;
    qtype = c10::kQInt32;
  }

  // Normalize axis for ATen (ATen doesn't handle negative axes in
  // quantize_per_channel)
  int64_t normalized_axis = axis;
  if (normalized_axis < 0) {
    normalized_axis += input_sizes_int64.size();
  }

  // Quantize using ATen
  at::Tensor quantized_aten = at::quantize_per_channel(
      float_x, scale_tensor, adjusted_zero_points, normalized_axis, qtype);

  // Get ATen dequantized output
  at::Tensor aten_out = at::dequantize(quantized_aten).to(out_dtype);

  // Extract the quantized values (int_repr) to use with our implementations
  at::Tensor quantized_input = quantized_aten.int_repr().to(dtype);

  // Get reference output using
  // torch::executor::native::dequantize_per_channel_aten
  at::Tensor reference_out =
      torch::executor::native::dequantize_per_channel_aten(
          quantized_input,
          scale_tensor.to(at::kDouble),
          zero_point_tensor.to(at::kLong),
          axis,
          quant_min,
          quant_max,
          dtype,
          out_dtype);

  // Build Vulkan dequantize_per_channel graph
  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(in_storage);
  ComputeGraph graph(config);

  // Add tensors to graph
  IOValueRef r_input = graph.add_input_tensor(
      quantized_input.sizes().vec(),
      from_at_scalartype(quantized_input.scalar_type()),
      in_storage);

  IOValueRef r_scale = graph.add_input_tensor(
      scale_tensor.sizes().vec(),
      vkapi::kFloat,
      utils::kBuffer,
      utils::kWidthPacked);

  IOValueRef r_zero_point = graph.add_input_tensor(
      adjusted_zero_points.sizes().vec(),
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);

  ValueRef r_out = graph.add_tensor(
      quantized_input.sizes().vec(),
      from_at_scalartype(out_dtype),
      out_storage);

  const ValueRef r_axis = graph.add_scalar<int64_t>(axis);
  const ValueRef r_quant_min = graph.add_scalar<int64_t>(quant_min);
  const ValueRef r_quant_max = graph.add_scalar<int64_t>(quant_max);
  const ValueRef r_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(dtype));
  const ValueRef r_output_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(out_dtype));

  VK_GET_OP_FN("quantized_decomposed.dequantize_per_channel.default")
  (graph,
   {
       r_input.value,
       r_scale.value,
       r_zero_point.value,
       r_axis,
       r_quant_min,
       r_quant_max,
       r_dtype,
       r_output_dtype,
       r_out,
   });

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.prepack();

  // Copy input data to GPU
  graph.copy_into_staging(
      r_input.staging,
      quantized_input.const_data_ptr(),
      quantized_input.numel());

  // copy scale tensor to GPU
  graph.copy_into_staging(
      r_scale.staging, scale_tensor.const_data_ptr(), scale_tensor.numel());

  // copy zero_point tensor to GPU
  graph.copy_into_staging(
      r_zero_point.staging,
      zero_point_tensor.const_data_ptr(),
      zero_point_tensor.numel());

  // Execute the graph
  graph.execute();

  // Copy output data back to CPU
  at::Tensor vk_out = at::empty_like(reference_out).contiguous();
  graph.copy_from_staging(
      staging_out, vk_out.mutable_data_ptr(), vk_out.numel());

  // Compare outputs with appropriate tolerance for half precision
  bool output_correct;
  if (out_dtype == at::kHalf) {
    // Use higher tolerance for half precision due to limited precision
    output_correct =
        at::allclose(reference_out, vk_out, /*rtol=*/1e-2, /*atol=*/1e-2);
  } else {
    output_correct =
        at::allclose(reference_out, vk_out, /*rtol=*/1e-5, /*atol=*/1e-5);
  }
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
    std::cout << "  input dtype: " << dtype << std::endl;
    std::cout << "  output dtype: " << out_dtype << std::endl;
    std::cout << "  storage: " << in_storage << std::endl;
    std::cout << std::endl;

    std::cout << "\033[91m quantized_input: \033[0m" << std::endl;
    std::cout << quantized_input << std::endl;
    std::cout << "\033[91m aten: \033[0m" << std::endl;
    std::cout << aten_out << std::endl;
    std::cout << "\033[91m reference: \033[0m" << std::endl;
    std::cout << reference_out << std::endl;
    std::cout << "\033[91m vulkan: \033[0m" << std::endl;
    std::cout << vk_out << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

TEST(
    VulkanDequantizePerChannelTest,
    test_reference_dequantize_per_channel_uint8_to_float_3D_axis0) {
  std::vector<float> scales = {0.1, 0.2, 0.3};
  std::vector<int> zero_points = {0, 5, -2};

  test_reference_dequantize_per_channel(
      {3, 4, 2}, // input sizes
      scales,
      zero_points,
      0, // axis
      0, // quant_min
      255, // quant_max
      at::kByte,
      at::kFloat);
}

TEST(
    VulkanDequantizePerChannelTest,
    test_reference_dequantize_per_channel_int8_to_float_3D_axis2) {
  std::vector<float> scales = {0.1, 0.2};
  std::vector<int> zero_points = {0, 5};

  test_reference_dequantize_per_channel(
      {3, 4, 2}, // input sizes
      scales,
      zero_points,
      2, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);
}

TEST(
    VulkanDequantizePerChannelTest,
    test_reference_dequantize_per_channel_int8_to_float_3D_axisn1) {
  std::vector<float> scales = {0.1, 0.2};
  std::vector<int> zero_points = {0, 5};

  test_reference_dequantize_per_channel(
      {3, 4, 2}, // input sizes
      scales,
      zero_points,
      -1, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);
}

TEST(
    VulkanDequantizePerChannelTest,
    test_reference_dequantize_per_channel_int32_to_float_4D_axis0) {
  std::vector<float> scales = {0.1, 0.2, 0.00002};
  std::vector<int> zero_points = {0, 5, -4};

  test_reference_dequantize_per_channel(
      {3, 4, 2, 5}, // input sizes
      scales,
      zero_points,
      0, // axis
      std::numeric_limits<int32_t>::min(), // quant_min
      std::numeric_limits<int32_t>::max(), // quant_max
      at::kInt,
      at::kFloat);
}

// END OF REFERENCE TESTS

TEST(
    VulkanDequantizePerChannelTest,
    test_vulkan_dequantize_per_channel_int8_to_float_axis0) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales(9, 0.1f);
  std::vector<int> zero_points(9, 2);

  // 1D Tensor
  test_vulkan_dequantize_per_channel(
      {9}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);

  // 2D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);

  // 3D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 7, 11}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 17, 5, 5}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);

  // 4D Tensor (negative axis)
  test_vulkan_dequantize_per_channel(
      {5, 17, 5, 9}, // input sizes
      scales,
      zero_points,
      -1, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);
}

TEST(
    VulkanDequantizePerChannelTest,
    test_vulkan_dequantize_per_channel_int8_to_float_axis1) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales(14, 0.001f);
  std::vector<int> zero_points(14, -5);

  // 2D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14}, // input sizes
      scales,
      zero_points,
      1, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);

  // 3D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14, 11}, // input sizes
      scales,
      zero_points,
      1, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14, 5, 5}, // input sizes
      scales,
      zero_points,
      1, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);

  // 4D Tensor (negative axis)
  test_vulkan_dequantize_per_channel(
      {9, 7, 14, 5}, // input sizes
      scales,
      zero_points,
      -2, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);
}

TEST(
    VulkanDequantizePerChannelTest,
    test_vulkan_dequantize_per_channel_int8_to_float_axis2) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales(11, 0.5f);
  std::vector<int> zero_points(11, 12);

  // 3D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14, 11}, // input sizes
      scales,
      zero_points,
      2, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14, 11, 5}, // input sizes
      scales,
      zero_points,
      2, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);

  // 4D Tensor (negative axis)
  test_vulkan_dequantize_per_channel(
      {9, 11, 14, 5}, // input sizes
      scales,
      zero_points,
      -3, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);
}

TEST(
    VulkanDequantizePerChannelTest,
    test_vulkan_dequantize_per_channel_int8_to_float_axis3) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales(7, 0.5f);
  std::vector<int> zero_points(7, 12);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      3, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);

  // 4D Tensor (negative axis)
  test_vulkan_dequantize_per_channel(
      {7, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      -4, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kFloat);
}

TEST(
    VulkanDequantizePerChannelTest,
    test_vulkan_dequantize_per_channel_uint8_to_float_comprehensive) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales = {0.1, 0.2, 0.0001, 0.5, 0.02};
  std::vector<int> zero_points = {0, 5, -5, 1, 12};

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {5, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      0, // axis
      0, // quant_min
      255, // quant_max
      at::kByte,
      at::kFloat);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 5, 11, 7}, // input sizes
      scales,
      zero_points,
      1, // axis
      0, // quant_min
      255, // quant_max
      at::kByte,
      at::kFloat);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14, 5, 7}, // input sizes
      scales,
      zero_points,
      2, // axis
      0, // quant_min
      255, // quant_max
      at::kByte,
      at::kFloat);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14, 11, 5}, // input sizes
      scales,
      zero_points,
      3, // axis
      0, // quant_min
      255, // quant_max
      at::kByte,
      at::kFloat);

  // 4D Tensor (negative axis)
  test_vulkan_dequantize_per_channel(
      {5, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      -4, // axis
      0, // quant_min
      255, // quant_max
      at::kByte,
      at::kFloat);
}

TEST(
    VulkanDequantizePerChannelTest,
    test_vulkan_dequantize_per_channel_8bit_to_half) {
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
  test_vulkan_dequantize_per_channel(
      {5, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kHalf);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 5, 11, 7}, // input sizes
      scales,
      zero_points,
      1, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kHalf);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14, 5, 7}, // input sizes
      scales,
      zero_points,
      2, // axis
      0, // quant_min
      255, // quant_max
      at::kByte,
      at::kHalf);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14, 11, 5}, // input sizes
      scales,
      zero_points,
      3, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kHalf);

  // 4D Tensor (negative axis)
  test_vulkan_dequantize_per_channel(
      {5, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      -4, // axis
      0, // quant_min
      255, // quant_max
      at::kByte,
      at::kHalf);
}

TEST(
    VulkanDequantizePerChannelTest,
    test_vulkan_dequantize_per_channel_8bit_to_double) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  std::vector<float> scales = {0.1, 0.2, 0.01, 0.5, 0.02};
  std::vector<int> zero_points = {0, 5, 5, 1, 12};

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {5, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      0, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kDouble);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 5, 11, 7}, // input sizes
      scales,
      zero_points,
      1, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kDouble);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14, 5, 7}, // input sizes
      scales,
      zero_points,
      2, // axis
      0, // quant_min
      255, // quant_max
      at::kByte,
      at::kDouble);

  // 4D Tensor
  test_vulkan_dequantize_per_channel(
      {9, 14, 11, 5}, // input sizes
      scales,
      zero_points,
      3, // axis
      -128, // quant_min
      127, // quant_max
      at::kChar,
      at::kDouble);

  // 4D Tensor (negative axis)
  test_vulkan_dequantize_per_channel(
      {5, 14, 11, 7}, // input sizes
      scales,
      zero_points,
      -4, // axis
      0, // quant_min
      255, // quant_max
      at::kByte,
      at::kDouble);
}

void test_vulkan_dequantize_per_tensor_tensor_impl(
    const std::vector<int>& input_sizes,
    float scale,
    int zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype,
    const vkcompute::utils::StorageType in_storage,
    const vkcompute::utils::StorageType out_storage) {
  check_dequantize_args(quant_min, quant_max, dtype, out_dtype);
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());

  // Create a quantized input tensor with values from quant_min to quant_max
  at::Tensor input;
  if (dtype == at::kByte) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kByte));
  } else if (dtype == at::kChar) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kChar));
  } else if (dtype == at::kShort) {
    input =
        at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kShort));
  } else if (dtype == at::kInt) {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kInt));
  } else {
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kLong));
  }

  // Fill with a simple pattern: values from quant_min to quant_max in steps
  float step = 1.0f;
  if (input.numel() > 1) {
    step = static_cast<float>(quant_max - quant_min) / (input.numel() - 1);
  }

  auto flat_input = input.flatten();
  for (int i = 0; i < flat_input.numel(); i++) {
    int64_t qvalue = quant_min + i * step;
    if (dtype == at::kByte) {
      flat_input[i] = static_cast<uint8_t>(qvalue);
    } else if (dtype == at::kChar) {
      flat_input[i] = static_cast<int8_t>(qvalue);
    } else if (dtype == at::kShort) {
      flat_input[i] = static_cast<int16_t>(qvalue);
    } else if (dtype == at::kInt) {
      flat_input[i] = static_cast<int32_t>(qvalue);
    } else if (dtype == at::kLong) {
      flat_input[i] = static_cast<int64_t>(qvalue);
    }
  }

  // Reshape back to original dimensions
  input = flat_input.reshape(input_sizes_int64);

  // Create scale and zero_point as tensors (single element tensors)
  at::Tensor scale_tensor =
      at::tensor({scale}, at::device(at::kCPU).dtype(at::kDouble));
  at::Tensor zero_point_tensor =
      at::tensor({zero_point}, at::device(at::kCPU).dtype(at::kLong));

  // Get reference output using tensor variant
  at::Tensor reference_out =
      torch::executor::native::dequantize_per_tensor_tensor_args_aten(
          input,
          scale_tensor,
          zero_point_tensor,
          quant_min,
          quant_max,
          dtype,
          out_dtype);

  // Build Vulkan dequantize_per_tensor.tensor graph
  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(in_storage);
  ComputeGraph graph(config);

  IOValueRef r_input = graph.add_input_tensor(
      input.sizes().vec(), from_at_scalartype(dtype), in_storage);

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
      input.sizes().vec(), from_at_scalartype(out_dtype), out_storage);

  const ValueRef r_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(dtype));
  const ValueRef r_out_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(out_dtype));

  VK_GET_OP_FN("quantized_decomposed.dequantize_per_tensor.tensor")
  (graph,
   {
       r_input.value,
       r_scale.value,
       r_zero_point.value,
       r_quant_min,
       r_quant_max,
       r_dtype,
       r_out_dtype,
       r_out,
   });

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.prepack();

  // Run Vulkan dequantize_per_tensor.tensor
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

  // Compare outputs with appropriate tolerance for half precision
  bool output_correct;
  if (out_dtype == at::kHalf) {
    // Use higher tolerance for half precision due to limited precision
    output_correct =
        at::allclose(reference_out, vk_out, /*rtol=*/1e-2, /*atol=*/1e-2);
  } else {
    output_correct =
        at::allclose(reference_out, vk_out, /*rtol=*/1e-5, /*atol=*/1e-5);
  }
  if (!output_correct) {
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
    std::cout << "  input dtype: " << dtype << std::endl;
    std::cout << "  output dtype: " << out_dtype << std::endl;

    std::cout << "input:" << std::endl;
    std::cout << input << std::endl;
    std::cout << "reference:" << std::endl;
    std::cout << reference_out << std::endl;
    std::cout << "vulkan:" << std::endl;
    std::cout << vk_out << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

TEST(
    VulkanDequantizePerTensorTensorTest,
    test_vulkan_dequantize_per_tensor_tensor_int8_to_float) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  test_vulkan_dequantize_per_tensor_tensor(
      {2, 3, 4}, // input sizes
      0.01, // scale
      1, // zero_point
      -128, // quant_min
      127, // quant_max
      at::kChar, // input dtype
      at::kFloat); // output dtype
}

TEST(
    VulkanDequantizePerTensorTensorTest,
    test_vulkan_dequantize_per_tensor_tensor_uint8_to_float) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  test_vulkan_dequantize_per_tensor_tensor(
      {2, 3, 4, 12}, // input sizes
      0.1, // scale
      5, // zero_point
      0, // quant_min
      255, // quant_max
      at::kByte, // input dtype
      at::kFloat); // output dtype
}

TEST(
    VulkanDequantizePerTensorTensorTest,
    test_vulkan_dequantize_per_tensor_tensor_int32_to_float) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  test_vulkan_dequantize_per_tensor_tensor(
      {2, 3}, // input sizes
      0.01, // scale
      12, // zero_point
      std::numeric_limits<int32_t>::min(), // quant_min
      std::numeric_limits<int32_t>::max(), // quant_max
      at::kInt, // input dtype
      at::kFloat); // output dtype
}

TEST(
    VulkanDequantizePerTensorTensorTest,
    test_vulkan_dequantize_per_tensor_tensor_uint8_to_half) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  test_vulkan_dequantize_per_tensor_tensor(
      {3, 4}, // input sizes
      0.3, // scale
      2, // zero_point
      0, // quant_min
      255, // quant_max
      at::kByte, // input dtype
      at::kHalf); // output dtype
}

TEST(
    VulkanDequantizePerTensorTensorTest,
    test_vulkan_dequantize_per_tensor_tensor_int8_to_double) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  test_vulkan_dequantize_per_tensor_tensor(
      {2, 3, 4}, // input sizes
      0.03, // scale
      -2, // zero_point
      -128, // quant_min
      127, // quant_max
      at::kChar, // input dtype
      at::kDouble); // output dtype
}
