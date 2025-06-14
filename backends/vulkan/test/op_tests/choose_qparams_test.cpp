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
