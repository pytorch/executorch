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
