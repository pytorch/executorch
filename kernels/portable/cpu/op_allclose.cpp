/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/compiler.h>
#include <math.h>
#include <string.h>

namespace torch {
namespace executor {
namespace native {
using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using Scalar = exec_aten::Scalar;
namespace {

/**
 * Returns true if the two arrays are close according to the description on
 * `tensors_are_close()`.
 *
 * T must be a floating point type. Non-floating point data should be compared
 * directly.
 */
template <typename T>
bool data_is_close(
    const T* a,
    const T* b,
    size_t numel,
    double rtol,
    double atol) {
  for (size_t i = 0; i < numel; i++) {
    if (rtol == 0 && atol == 0) {
      // Exact comparison; avoid unnecessary math.
      if (a[i] != b[i]) {
        return false;
      }
    } else {
      auto allowed_error = atol + fabs(rtol * b[i]);
      auto actual_error = fabs(a[i] - b[i]);
      if (!isfinite(actual_error) || actual_error > allowed_error) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Returns true if the tensors are of the same shape and dtype, and if all
 * elements are close to each other.
 *
 * A number A is close to B when either:
 *
 * (1) A is equal to B.
 * (2) The error abs(A - B) is finite and less than the max error
 *     (atol + abs(rtol * B)).
 *
 * NOTE: rtol/atol are ignored for non-floating-point dtypes.
 */
bool tensors_are_close(
    const Tensor& a,
    const Tensor& b,
    double rtol,
    double atol) {
  // TODO(dbort): Listen to strides instead of assuming that the data is
  // contiguous.

  if (a.scalar_type() == ScalarType::Float) {
    return data_is_close<float>(
        a.const_data_ptr<float>(),
        b.const_data_ptr<float>(),
        a.numel(),
        rtol,
        atol);
  } else if (a.scalar_type() == ScalarType::Double) {
    return data_is_close<double>(
        a.const_data_ptr<double>(),
        b.const_data_ptr<double>(),
        a.numel(),
        rtol,
        atol);
  } else {
    // Non-floating-point types can be compared bitwise.
    return memcmp(a.mutable_data_ptr(), b.mutable_data_ptr(), a.nbytes()) == 0;
  }
}
} // namespace

Tensor& allclose_out(
    const Tensor& self,
    const Tensor& other,
    double rtol,
    double atol,
    ET_UNUSED bool equal_nan,
    ET_UNUSED bool dummy_param,
    Tensor& out) {
  ET_CHECK_SAME_SHAPE_AND_DTYPE2(self, other);
  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Bool,
      "Out tensor must be type Bool; saw type %" PRId8,
      static_cast<int8_t>(out.scalar_type()));
  ET_CHECK_MSG(
      tensors_have_same_dim_order(self, other, out),
      "self, other and out tensors should have same dim order");
  ET_CHECK_MSG(
      out.numel() == 1,
      "Out tensor must be a single element; saw %zu elements",
      (size_t)out.numel());
  auto out_data = out.mutable_data_ptr<bool>();
  out_data[0] = tensors_are_close(self, other, rtol, atol);
  return out;
}

/**
 * Note: This custom operator contains two variants: allclose.Tensor (a
 * functional variant, no inplace mutating on the arguments) and allclose.out
 * (an out variant, mutating out). We need to register both into the PyTorch
 * runtime so that they can be visible from ExecuTorch compiler side. Eventually
 * only allclose.out will be seen from ExecuTorch runtime. With this setup, the
 * portable kernel for allclose.Tensor can be implemented as a wrapper of
 * allclose.out. We can easily instantiate an at::Tensor for the out argument,
 * then pass it into allclose.out. This logic will only need to work out in
 * "ATen mode" for ExecuTorch compiler, since we won't expose allclose.Tensor in
 * ExecuTorch runtime.
 */
Tensor allclose_tensor(
    ET_UNUSED const Tensor& self,
    ET_UNUSED const Tensor& other,
    ET_UNUSED double rtol,
    ET_UNUSED double atol,
    ET_UNUSED bool equal_nan,
    ET_UNUSED bool dummy_param) {
#ifdef USE_ATEN_LIB
  Tensor out =
      torch::tensor({false}, c10::TensorOptions(c10::ScalarType::Bool));
  allclose_out(self, other, rtol, atol, equal_nan, dummy_param, out);
  return out;
#else
  ET_ASSERT_UNREACHABLE();
#endif
}

Tensor& allclose_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    const Tensor& other,
    double rtol,
    double atol,
    ET_UNUSED bool equal_nan,
    ET_UNUSED bool dummy_param,
    Tensor& out) {
  (void)ctx;
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  return allclose_out(self, other, rtol, atol, equal_nan, dummy_param, out);
}

Tensor allclose_tensor(
    ET_UNUSED KernelRuntimeContext& ctx,
    ET_UNUSED const Tensor& self,
    ET_UNUSED const Tensor& other,
    ET_UNUSED double rtol,
    ET_UNUSED double atol,
    ET_UNUSED bool equal_nan,
    ET_UNUSED bool dummy_param) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  ET_ASSERT_UNREACHABLE();
}
} // namespace native
} // namespace executor
} // namespace torch
