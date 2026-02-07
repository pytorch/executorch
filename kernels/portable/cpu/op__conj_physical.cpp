/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::Tensor;

Tensor& _conj_physical_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out) {
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dtype(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "_conj_physical.out";

  ET_SWITCH_COMPLEXH_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&] {
    apply_unary_map_fn<CTYPE, CTYPE>(
        [](const CTYPE val_in) -> CTYPE {
          return CTYPE(val_in.real_, -val_in.imag_);
        },
        in.const_data_ptr<CTYPE>(),
        out.mutable_data_ptr<CTYPE>(),
        in.numel());
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
