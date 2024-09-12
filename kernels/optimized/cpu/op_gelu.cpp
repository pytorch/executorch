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

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using string_view = exec_aten::string_view;

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

  // TODO: Add fast path for tanh using sleef's tanh
  if (approximate == "tanh") {
    // 0.5 * x * (1 + Tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))
    for (size_t i = 0; i < lim; ++i) {
      const CTYPE x = in_data[i];
      const CTYPE kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
      const CTYPE kKappa = 0.044715;
      auto x_cube = x * x * x;
      auto inner = kBeta * (x + kKappa * x_cube);
      out_data[i] = CTYPE(0.5) * x * (CTYPE(1) + std::tanh(inner));
    }
  } else if (approximate == "none") { // dont appx
    // GELU(x) = x * Φ(x) where Φ(x) is the is the Cumulative Distribution
    // Function for Gaussian Distribution.

#ifndef __aarch64__
    for (size_t i = 0; i < lim; ++i) {
      const CTYPE x = in_data[i];
      out_data[i] = CTYPE(0.5) * x * (CTYPE(1) + std::erf(x * M_SQRT1_2));
    }
#else
    size_t i = 0;
    if (std::is_same<CTYPE, float>::value) {
      for (; i + 4 < lim; i += 4) {
        const float32x4_t in =
            vld1q_f32(static_cast<const float*>(&in_data[i]));
        const float32x4_t m_sqrt1_2x4 = {
            M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, M_SQRT1_2};
        const float32x4_t ones = vmovq_n_f32(1.0);
        const float32x4_t halves = vmovq_n_f32(0.5);
        float32x4_t out = Sleef_erff4_u10(vmulq_f32(in, m_sqrt1_2x4));
        vst1q_f32(
            static_cast<float*>(&out_data[i]),
            vmulq_f32(vmulq_f32(vaddq_f32(out, ones), in), halves));
      }
    }
    for (; i < lim; ++i) {
      const CTYPE x = in_data[i];
      out_data[i] = CTYPE(0.5) * x * (CTYPE(1) + std::erf(x * M_SQRT1_2));
    }
#endif // __aarch64__

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
      context,
      tensors_have_same_shape_and_dtype(input, out),
      InvalidArgument,
      out);

// helper for generating the cases for different data types
#define GELU(ctype, dtype)                         \
  case ScalarType::dtype:                          \
    gelu<ctype>(context, input, approximate, out); \
    break;

  switch (input.scalar_type()) {
    // TODO support Double as well
    GELU(float, Float)
    default:
      ET_KERNEL_CHECK_MSG(
          context,
          false,
          InvalidArgument,
          out,
          "Unhandled dtype %" PRId8,
          static_cast<int8_t>(input.scalar_type()));
  }
#undef GELU

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
