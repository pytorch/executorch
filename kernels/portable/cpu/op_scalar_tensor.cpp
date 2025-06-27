/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor&
scalar_tensor_out(KernelRuntimeContext& ctx, const Scalar& s, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, {}) == Error::Ok, InvalidArgument, out);

  ScalarType out_type = out.scalar_type();

  constexpr auto name = "scalar_tensor.out";

  if (s.isFloatingPoint() &&
      executorch::runtime::isIntegralType(out_type, false)) {
    ET_SWITCH_INT_TYPES(out_type, ctx, name, CTYPE, [&]() {
      out.mutable_data_ptr<CTYPE>()[0] =
          static_cast<CTYPE>(utils::scalar_to<int64_t>(s));
    });
  } else {
    ET_SWITCH_REALHBBF16_TYPES(out_type, ctx, name, CTYPE, [&]() {
      out.mutable_data_ptr<CTYPE>()[0] = utils::scalar_to<CTYPE>(s);
    });
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
