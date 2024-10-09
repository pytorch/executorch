/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <stdexcept>

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace my_custom_kernels {
namespace native {

using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::resize_tensor;
using executorch::runtime::tensors_have_same_shape_and_dtype;

namespace {

/**
 * Element-wise relu activation of `input` overwriting `out`.
 *
 *
 * Assumes that the tensors are contiguous, are the same shape, and have the
 * same dtype. CTYPE should be the C type (like `float` or `double`) that
 * matches the dtype of the tensors.
 */
template <typename CTYPE>
void relu(const Tensor& input, Tensor& output) {
  const CTYPE* in_data = input.data_ptr<CTYPE>();
  CTYPE* out_data = output.data_ptr<CTYPE>();
  size_t lim = input.numel();
  Tensor::SizesType expected_output_size[16];
  for (size_t i = 0; i < output.dim(); ++i) {
    expected_output_size[i] = input.size(i);
  }
  auto error = resize_tensor(
      output, {expected_output_size, static_cast<size_t>(output.dim())});
  // TODO: Construct error message with requested output sizes.
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  // relu(x) = max(x, 0)
  for (size_t i = 0; i < lim; ++i) {
    const CTYPE x = in_data[i];
    out_data[i] = (std::isnan(x) || x >= CTYPE(0.0)) ? x : CTYPE(0.0);
  }
}

} // namespace

/**
 * Element-wise Relu of `input`, overwriting `out`.
 *
 * Asserts that all tensors have the same dtype and shape.
 *
 * relu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
 */
Tensor&
my_relu_out(KernelRuntimeContext& context, const Tensor& input, Tensor& out) {
  (void)context;
  resize(out, input.sizes());
  ET_KERNEL_CHECK(
      context,
      tensors_have_same_shape_and_dtype(input, out),
      InvalidArgument,
      out);

// helper for generating the cases for different data types
#define RELU(ctype, dtype)   \
  case ScalarType::dtype:    \
    relu<ctype>(input, out); \
    break;

  switch (input.scalar_type()) {
    ET_FORALL_REAL_TYPES(RELU)
    default:
      ET_KERNEL_CHECK_MSG(
          context,
          false,
          InvalidArgument,
          out,
          "Unhandled dtype %hhd",
          input.scalar_type());
  }
#undef RELU

  return out;
}

} // namespace native
} // namespace my_custom_kernels
