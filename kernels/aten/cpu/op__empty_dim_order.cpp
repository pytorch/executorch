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

using exec_aten::IntArrayRef;
using exec_aten::Tensor;
using OptionalIntArrayRef = exec_aten::OptionalArrayRef<int64_t>;
using DimOrderArrayRef = exec_aten::ArrayRef<executorch::aten::DimOrderType>;
// Out Aten tensor shall have same memory format stride as dim_order
const size_t kMaxNumOfDimensions = 16;

namespace {

inline bool _check__empty_out_dim_order(
    OptionalIntArrayRef dim_order,
    Tensor& out) {
  exec_aten::ArrayRef<int64_t> dim_order_ref;
  std::vector<int64_t> dim_order_vec;

  if (dim_order.has_value()) {
    // out tensor's dim order shall equal to input dim order
    dim_order_ref = exec_aten::ArrayRef<int64_t>(
        dim_order.value().data(), dim_order.value().size());
  } else { // dim_order is not set, out tensor should be contiguous dim order
    for (int i = 0; i < out.dim(); i++) {
      dim_order_vec.push_back(i);
    }
    dim_order_ref = exec_aten::ArrayRef<int64_t>(dim_order_vec);
  }

  // dim order size shall equal to input dim
  ET_LOG_AND_RETURN_IF_FALSE(dim_order_ref.size() == out.dim());

  ET_LOG_AND_RETURN_IF_FALSE(
      is_channels_last_dim_order(dim_order_ref.data(), dim_order_ref.size()) ||
      is_contiguous_dim_order(dim_order_ref.data(), dim_order_ref.size()));

  ET_LOG_AND_RETURN_IF_FALSE(kMaxNumOfDimensions >= out.dim());
  exec_aten::StridesType target_strides[kMaxNumOfDimensions];
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
  executorch::runtime::KernelRuntimeContext ctx{};
  return _empty_dim_order_out(ctx, size, dim_order, out);
}

} // namespace native
} // namespace executor
} // namespace torch
