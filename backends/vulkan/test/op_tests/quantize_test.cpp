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
