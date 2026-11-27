/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& complex_out(
    KernelRuntimeContext& ctx,
    const Tensor& real,
    const Tensor& imag,
    Tensor& out) {
  // Check Input Dtypes
  const ScalarType dtype = real.scalar_type();
  ET_KERNEL_CHECK(
      ctx,
      dtype == ScalarType::Half || dtype == ScalarType::Float ||
          dtype == ScalarType::Double,
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(ctx, imag.scalar_type() == dtype, InvalidArgument, out);

  // Check Output Dtype
  ET_KERNEL_CHECK(
      ctx,
      out.scalar_type() == executorch::runtime::toComplexType(dtype),
      InvalidArgument,
      out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx,
      tensor_is_default_dim_order(real) && tensor_is_default_dim_order(imag) &&
          tensor_is_default_dim_order(out),
      InvalidArgument,
      out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(real, imag, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_SWITCH_FLOATH_TYPES(dtype, ctx, "complex.out", CTYPE, [&] {
    using Complex = executorch::aten::complex<CTYPE>;
    apply_binary_elementwise_fn<CTYPE, CTYPE, Complex>(
        [](CTYPE real_value, CTYPE imag_value) {
          return Complex(real_value, imag_value);
        },
        real,
        imag,
        out);
  });
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
