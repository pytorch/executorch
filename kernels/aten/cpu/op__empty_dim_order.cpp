/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright 2026 Arm Limited and/or its affiliates.
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

using executorch::aten::IntArrayRef;
using executorch::aten::Tensor;
using OptionalIntArrayRef = executorch::aten::OptionalArrayRef<int64_t>;
using DimOrderArrayRef =
    executorch::aten::ArrayRef<executorch::aten::DimOrderType>;
// Out Aten tensor shall have same memory format stride as dim_order
const size_t kMaxNumOfDimensions = 16;

namespace {

inline bool check_empty_out_dim_order_ref(
    executorch::aten::ArrayRef<int64_t> dim_order_ref,
    Tensor& out) {
  // dim order size shall equal to input dim
  ET_LOG_AND_RETURN_IF_FALSE(dim_order_ref.size() == out.dim());

  ET_LOG_AND_RETURN_IF_FALSE(
      is_channels_last_dim_order(dim_order_ref.data(), dim_order_ref.size()) ||
      is_contiguous_dim_order(dim_order_ref.data(), dim_order_ref.size()));

  ET_LOG_AND_RETURN_IF_FALSE(kMaxNumOfDimensions >= out.dim());
  executorch::aten::StridesType target_strides[kMaxNumOfDimensions];
  dim_order_to_stride_nocheck(
      out.sizes().data(),
      dim_order_ref.data(),
      dim_order_ref.size(),
      target_strides);

  for (size_t i = 0; i < dim_order_ref.size(); i++) {
    ET_LOG_AND_RETURN_IF_FALSE(target_strides[i] == out.strides()[i]);
  }

  return true;
}

inline bool _check__empty_out_dim_order(
    OptionalIntArrayRef dim_order,
    Tensor& out) {
  if (dim_order.has_value()) {
    // out tensor's dim order shall equal to input dim order
    return check_empty_out_dim_order_ref(
        executorch::aten::ArrayRef<int64_t>(
            dim_order.value().data(), dim_order.value().size()),
        out);
  } else { // dim_order is not set, out tensor should be contiguous dim order
    const auto ndim = out.dim();
    ET_LOG_AND_RETURN_IF_FALSE(
        ndim <= static_cast<ssize_t>(kMaxNumOfDimensions));
    int64_t dim_order_arr[kMaxNumOfDimensions];
    for (ssize_t i = 0; i < ndim; i++) {
      dim_order_arr[i] = i;
    }
    return check_empty_out_dim_order_ref(
        executorch::aten::ArrayRef<int64_t>(
            dim_order_arr, static_cast<size_t>(ndim)),
        out);
  }
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
  ET_KERNEL_CHECK(
      context,
      _check__empty_out_dim_order(dim_order, out),
      InvalidArgument,
      out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      context,
      resize_tensor(out, size) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  return out;
}

Tensor& _empty_dim_order_out(
    IntArrayRef size,
    OptionalIntArrayRef dim_order,
    Tensor& out) {
  KernelRuntimeContext ctx{};
  return _empty_dim_order_out(ctx, size, dim_order, out);
}

} // namespace native
} // namespace executor
} // namespace torch
