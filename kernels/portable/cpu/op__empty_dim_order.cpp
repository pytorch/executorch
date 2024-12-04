/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <cstdint>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Tensor;
using OptionalIntArrayRef = exec_aten::OptionalArrayRef<int64_t>;
using DimOrderArrayRef = exec_aten::ArrayRef<executorch::aten::DimOrderType>;

namespace {

bool _check__empty_out_dim_order(OptionalIntArrayRef dim_order, Tensor& out) {
  DimOrderArrayRef out_dim_order = out.dim_order();

  if (dim_order.has_value()) {
    // out tensor's dim order shall equal to input dim order
    IntArrayRef dim_order_ref = dim_order.value();

    ET_LOG_AND_RETURN_IF_FALSE(
        is_channels_last_dim_order(
            dim_order.value().data(), dim_order.value().size()) ||
        is_contiguous_dim_order(
            dim_order.value().data(), dim_order.value().size()));

    // Out tensor shall have same dim order as dim_order
    ET_LOG_AND_RETURN_IF_FALSE(out_dim_order.size() == dim_order_ref.size());
    for (size_t i = 0; i < dim_order_ref.size(); i++) {
      ET_LOG_AND_RETURN_IF_FALSE(out_dim_order[i] == dim_order_ref[i]);
    }
  } else { // dim_order is not set, out tensor should be contiguous memory
           // format
    ET_LOG_AND_RETURN_IF_FALSE(
        is_contiguous_dim_order(out_dim_order.data(), out_dim_order.size()));
  }
  return true;
}

} // namespace

/*
 * Empty out tensor with specified dim order
 *
 * _empty_dim_order.out(SymInt[] size, *, int[]? dim_order=None, Tensor(a!) out)
 * -> Tensor(a!)
 */
Tensor& _empty_dim_order_out(
    KernelRuntimeContext& context,
    IntArrayRef size,
    OptionalIntArrayRef dim_order,
    Tensor& out) {
  (void)context;

  // Check if dim_order is valid
  _check__empty_out_dim_order(dim_order, out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      context,
      resize_tensor(out, size) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
