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

  // Use WRAP_TO_ATEN with the wrapper function
  WRAP_TO_ATEN(choose_qparams_per_token_asymmetric_out_no_context, 2)
  (input, et_dtype, scale_out, zero_point_out);

  return {scale_out, zero_point_out};
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
