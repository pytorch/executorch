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
  ScalarType et_dtype;
  ScalarType et_out_dtype;

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
    default:
      throw std::runtime_error("Unsupported dtype");
  }

  switch (out_dtype) {
    case at::kFloat:
      et_out_dtype = ScalarType::Float;
      break;
    case at::kDouble:
      et_out_dtype = ScalarType::Double;
      break;
    default:
      throw std::runtime_error("Unsupported out_dtype");
  }

  executorch::aten::optional<ScalarType> opt_et_out_dtype(et_out_dtype);

  WRAP_TO_ATEN(dequantize_per_tensor_out_no_context, 7)
  (input, scale, zero_point, quant_min, quant_max, et_dtype, opt_et_out_dtype, out);
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
  ScalarType et_dtype;
  ScalarType et_out_dtype;

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
    default:
      throw std::runtime_error("Unsupported dtype");
  }

  switch (out_dtype) {
    case at::kFloat:
      et_out_dtype = ScalarType::Float;
      break;
    case at::kDouble:
      et_out_dtype = ScalarType::Double;
      break;
    default:
      throw std::runtime_error("Unsupported out_dtype");
  }

  WRAP_TO_ATEN(dequantize_per_token_out_no_context, 7)
  (input, scale, zero_points, quant_min, quant_max, et_dtype, et_out_dtype, out);
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
  at::Tensor int_input = input.to(at::kInt);
  at::Tensor flat_input = int_input.flatten();
  at::Tensor flat_out = out.flatten();

  for (int i = 0; i < flat_input.numel(); i++) {
    int64_t qvalue = flat_input[i].item<int64_t>();
    float value = static_cast<float>((qvalue - zero_point) * scale);

    if (out_dtype == at::kFloat) {
      flat_out[i] = value;
    } else if (out_dtype == at::kDouble) {
      flat_out[i] = static_cast<double>(value);
    }
  }

  return out.reshape(input.sizes());
}

// Forward declaration of implementation functions
void test_vulkan_dequantize_per_tensor_impl(
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
void test_vulkan_dequantize_per_tensor(
    const std::vector<int>& input_sizes,
    float scale,
    int zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype,
    at::ScalarType out_dtype) {
  // Test with buffer storage
  test_vulkan_dequantize_per_tensor_impl(
      input_sizes,
      scale,
      zero_point,
      quant_min,
      quant_max,
      dtype,
      out_dtype,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  // Test with texture storage
  test_vulkan_dequantize_per_tensor_impl(
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
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kShort));
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
  const bool output_correct = at::allclose(reference_out, impl_out, 1e-5, 1e-5);
  if (!output_correct) {
    std::cout << "\n"
              << "Failed with parameters: " << std::endl;
    std::cout << "  scale: " << scale << std::endl;
    std::cout << "  zero_point: " << zero_point << std::endl;
    std::cout << "  quant_min: " << quant_min << std::endl;
    std::cout << "  quant_max: " << quant_max << std::endl;

    std::cout << "input:" << std::endl;
    std::cout << input << std::endl;
    std::cout << "reference:" << std::endl;
    std::cout << reference_out << std::endl;
    std::cout << "implementation:" << std::endl;
    std::cout << impl_out << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

void test_vulkan_dequantize_per_tensor_impl(
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
    input = at::zeros(input_sizes_int64, at::device(at::kCPU).dtype(at::kShort));
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
  at::Tensor reference_out = torch::executor::native::dequantize_per_tensor_aten(
      input, scale, zero_point, quant_min, quant_max, dtype, out_dtype);

  // Build Vulkan dequantize_per_tensor graph
  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(in_storage);
  ComputeGraph graph(config);

  IOValueRef r_input = graph.add_input_tensor(
      input.sizes().vec(), from_at_scalartype(dtype), in_storage);

  const ValueRef r_scale = graph.add_scalar<double>(scale);
  const ValueRef r_zero_point = graph.add_scalar<int64_t>(zero_point);
  const ValueRef r_quant_min = graph.add_scalar<int64_t>(quant_min);
  const ValueRef r_quant_max = graph.add_scalar<int64_t>(quant_max);

  const ValueRef r_out = graph.add_tensor(
      input.sizes().vec(), from_at_scalartype(out_dtype), out_storage);

  VK_GET_OP_FN("dequantize_per_tensor.default")
  (graph,
   {
       r_input.value,
       r_scale,
       r_zero_point,
       r_quant_min,
       r_quant_max,
       r_out,
   });

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.encode_prepack();
  graph.prepack();
  graph.encode_execute();

  // Run Vulkan dequantize_per_tensor
  graph.copy_into_staging(
      r_input.staging, input.const_data_ptr(), input.numel());

  graph.execute();

  at::Tensor vk_out = at::empty_like(reference_out).contiguous();
  graph.copy_from_staging(
      staging_out, vk_out.mutable_data_ptr(), vk_out.numel());

  // Compare outputs
  const bool output_correct = at::allclose(reference_out, vk_out, 1e-5, 1e-5);
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

    std::cout << "input:" << std::endl;
    std::cout << input << std::endl;
    std::cout << "reference:" << std::endl;
    std::cout << reference_out << std::endl;
    std::cout << "vulkan:" << std::endl;
    std::cout << vk_out << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

// Test cases for dequantize_per_tensor
TEST(VulkanDequantizePerTensorTest, test_reference_dequantize_per_tensor_uint8_to_float) {
  test_reference_dequantize_per_tensor(
      {2, 3, 4}, // input sizes
      0.1, // scale
      5, // zero_point
      0, // quant_min
      255, // quant_max
      at::kByte, // input dtype
      at::kFloat); // output dtype
}

TEST(VulkanDequantizePerTensorTest, test_reference_dequantize_per_tensor_int8_to_float) {
  test_reference_dequantize_per_tensor(
      {3, 4, 5}, // input sizes
      0.05, // scale
      0, // zero_point
      -128, // quant_min
      127, // quant_max
      at::kChar, // input dtype
      at::kFloat); // output dtype
}

TEST(VulkanDequantizePerTensorTest, test_reference_dequantize_per_tensor_int16_to_float) {
  test_reference_dequantize_per_tensor(
      {2, 2, 3}, // input sizes
      0.001, // scale
      -10, // zero_point
      -32768, // quant_min
      32767, // quant_max
      at::kShort, // input dtype
      at::kFloat); // output dtype
}
