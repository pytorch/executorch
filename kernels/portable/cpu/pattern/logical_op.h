/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace internal {

/**
 * Implements an op pattern for ops that take two broadcastable input tensors
 * and performs an element-wise binary logical operation `fn`.
 */
template <const char* op_name>
Tensor& logical_tensor_out(
    bool (*fn)(bool, bool),
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  utils::apply_bitensor_elementwise_fn<
      bool,
      op_name,
      utils::SupportedTensorDtypes::REALHBBF16>(
      // TODO: rewrite this to be vectorization-capable.
      fn,
      ctx,
      a,
      utils::SupportedTensorDtypes::REALHBBF16,
      b,
      utils::SupportedTensorDtypes::REALHBBF16,
      out);

  return out;
}

} // namespace internal
} // namespace native
} // namespace executor
} // namespace torch
