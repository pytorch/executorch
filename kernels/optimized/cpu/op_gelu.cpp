/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef __aarch64__
#include <arm_neon.h>
#include <sleef.h>
#endif

#include <cmath>

#include <ATen/native/cpu/Gelu.h>
#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using string_view = executorch::aten::string_view;

namespace {

/**
 * Element-wise gelu activation of `input` overwriting `out`.
 *
 * 'approximate' specifies the method used to approximation the Gelu function
 *  either 'none' to not approximate or 'tanh'
 *
 * Assumes that the tensors are contiguous, are the same shape, and have the
 * same dtype. CTYPE should be the C type (like `float` or `double`) that
 * matches the dtype of the tensors.
 */
template <typename CTYPE>
void gelu(
    executorch::runtime::KernelRuntimeContext& context,
    const Tensor& input,
    string_view approximate,
    Tensor& output) {
  const CTYPE* in_data = input.const_data_ptr<CTYPE>();
  CTYPE* out_data = output.mutable_data_ptr<CTYPE>();
  size_t lim = input.numel();

  if (approximate == "tanh") {
    using Vec = at::vec::Vectorized<CTYPE>;
    int i = 0;
    for (; i < lim - (lim % Vec::size()); i += Vec::size()) {
      Vec x = Vec::loadu(in_data + i);
      at::native::vectorized_gelu_approximated_with_tanh(x).store(out_data + i);
    }
    for (; i < lim; ++i) {
      out_data[i] = at::native::scalar_gelu_approximated_with_tanh(in_data[i]);
    }
  } else if (approximate == "none") {
    using Vec = at::vec::Vectorized<CTYPE>;
    int i = 0;
    for (; i < lim - (lim % Vec::size()); i += Vec::size()) {
      Vec x = Vec::loadu(in_data + i);
      at::native::vectorized_gelu(x).store(out_data + i);
    }
    for (; i < lim; ++i) {
      out_data[i] = at::native::scalar_gelu(in_data[i]);
    }
  } else {
    ET_KERNEL_CHECK_MSG(
        context,
        false,
        InvalidArgument,
        ,
        "Invalid approximation format: %.*s for gelu",
        static_cast<int>(approximate.length()),
        approximate.data());
  }
}

} // namespace

/**
 * Element-wise Gelu of `input`, overwriting `out`.
 *
 * Asserts that all tensors have the same dtype and shape.
 *
 * gelu.out(Tensor self, str approximate, *, Tensor(a!) out) -> Tensor(a!)
 */
Tensor& opt_gelu_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    string_view approximate,
    Tensor& out) {
  (void)context;
  ET_KERNEL_CHECK(
      context, check_gelu_args(input, approximate, out), InvalidArgument, out);

  ET_SWITCH_FLOATHBF16_TYPES(
      input.scalar_type(), context, "gelu.out", CTYPE, [&]() {
        gelu<CTYPE>(context, input, approximate, out);
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
