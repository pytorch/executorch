/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

// copy.out(const Tensor& in, const Tensor& src, bool non_blocking, Tensor(a!)
// out) -> Tensor(a!), see caffe2/aten/src/ATen/native/Copy.cpp
// TODO: We actually shouldn't see this op with the proper functionalization,
// and this op needs to be deleted
Tensor& copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& src,
    bool non_blocking,
    Tensor& out) {
  (void)ctx;
  // Right now we only support blocking data transfer
  ET_KERNEL_CHECK(ctx, non_blocking == false, InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensors_have_same_dtype(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensor_is_broadcastable_to(src, in), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "copy.out";

  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, "copy.out", CTYPE, [&]() {
    utils::apply_bitensor_elementwise_fn<CTYPE, op_name>(
        [](ET_UNUSED const CTYPE _, const CTYPE val_src) { return val_src; },
        ctx,
        in,
        utils::SupportedTensorDtypes::REALHBBF16,
        src,
        utils::SupportedTensorDtypes::REALHBBF16,
        out,
        utils::SupportedTensorDtypes::REALHBBF16);
  });

  return out;
}

Tensor& copy_(
    KernelRuntimeContext& ctx,
    Tensor& in,
    const Tensor& src,
    bool non_blocking) {
  (void)ctx;
  // Right now we only support blocking data transfer
  ET_KERNEL_CHECK(ctx, non_blocking == false, InvalidArgument, in);

  ET_KERNEL_CHECK(
      ctx, tensor_is_broadcastable_to(src, in), InvalidArgument, in);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, src), InvalidArgument, in);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "copy_";

  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, "copy_", CTYPE, [&]() {
    utils::apply_bitensor_elementwise_fn<CTYPE, op_name>(
        [](ET_UNUSED const CTYPE _, const CTYPE val_src) { return val_src; },
        ctx,
        in,
        utils::SupportedTensorDtypes::REALHBBF16,
        src,
        utils::SupportedTensorDtypes::REALHBBF16,
        in,
        utils::SupportedTensorDtypes::REALHBBF16);
  });

  return in;
}

} // namespace native
} // namespace executor
} // namespace torch
