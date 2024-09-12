/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace internal {

Tensor& unary_ufunc_realhb_to_bool(
    bool (*fn)(double),
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK_MSG(
      ctx,
      out.scalar_type() == exec_aten::ScalarType::Bool,
      InvalidArgument,
      out,
      "Expected out tensor to have dtype Bool, but got %" PRId8 " instead.",
      static_cast<int8_t>(out.scalar_type()));

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  const auto in_type = in.scalar_type();

  ET_SWITCH_REALHB_TYPES(in_type, ctx, __func__, CTYPE_IN, [&] {
    apply_unary_map_fn(
        [fn](const CTYPE_IN val_in) { return fn(val_in); },
        in.const_data_ptr<CTYPE_IN>(),
        out.mutable_data_ptr<bool>(),
        in.numel());
  });

  return out;
}

} // namespace internal
} // namespace native
} // namespace executor
} // namespace torch
