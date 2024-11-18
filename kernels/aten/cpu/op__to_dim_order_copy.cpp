/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using SizesArrayRef = exec_aten::ArrayRef<exec_aten::SizesType>;
using DimOrderArrayRef = exec_aten::ArrayRef<exec_aten::DimOrderType>;
using MemoryFormat = exec_aten::MemoryFormat;

template <typename T>
using OptionalArrayRef = exec_aten::OptionalArrayRef<T>;

template <typename T>
using Optional = exec_aten::optional<T>;

namespace {
Optional<MemoryFormat> get_memory_format(OptionalArrayRef<int64_t> dim_order) {
  if (!dim_order.has_value()) {
    return exec_aten::nullopt;
  }
  if (is_contiguous_dim_order(
          dim_order.value().data(), dim_order.value().size())) {
    return MemoryFormat::Contiguous;
  } else if (is_channels_last_dim_order(
                 dim_order.value().data(), dim_order.value().size())) {
    return MemoryFormat::ChannelsLast;
  } else {
    ET_ASSERT_UNREACHABLE();
  }
}

bool check__to_dim_order_copy_args(
    const Tensor& input,
    bool non_blocking,
    exec_aten::OptionalArrayRef<int64_t> dim_order,
    Tensor& out) {
  // Right now we only support blocking data transfer
  ET_LOG_AND_RETURN_IF_FALSE(non_blocking == false);

  // dim_order is set, the target dim_order will be either contiguous or
  // channels_last memory format
  if (dim_order.has_value()) {
    exec_aten::ArrayRef<int64_t> dim_order_ref = dim_order.value();

    // dim order size shall equal to input dim
    ET_LOG_AND_RETURN_IF_FALSE(dim_order_ref.size() == input.dim());

    ET_LOG_AND_RETURN_IF_FALSE(
        is_channels_last_dim_order(
            dim_order.value().data(), dim_order.value().size()) ||
        is_contiguous_dim_order(
            dim_order.value().data(), dim_order.value().size()));

    // Out Aten tensor shall have same memory format stride as dim_order
    const size_t kMaxNumOfDimensions = 16;
    ET_LOG_AND_RETURN_IF_FALSE(kMaxNumOfDimensions >= out.dim());
    exec_aten::StridesType target_strides[kMaxNumOfDimensions];
    dim_order_to_stride_nocheck(
        out.sizes().data(),
        dim_order_ref.data(),
        dim_order_ref.size(),
        target_strides);
    ET_LOG_AND_RETURN_IF_FALSE(out.dim() == dim_order_ref.size());
    for (size_t i = 0; i < dim_order_ref.size(); i++) {
      ET_LOG_AND_RETURN_IF_FALSE(target_strides[i] == out.strides()[i]);
    }

  } else { // dim_order is not set, preserve the dim order of input

    auto out_strides = out.strides();
    auto input_strides = input.strides();
    ET_LOG_AND_RETURN_IF_FALSE(input_strides.size() == out_strides.size());
    for (size_t i = 0; i < input_strides.size(); i++) {
      ET_LOG_AND_RETURN_IF_FALSE(input_strides[i] == out_strides[i]);
    }
  }
  return true;
}
} // namespace

// _to_dim_order_copy.out(Tensor self, *, bool non_blocking=False, int[]?
// dim_order=None, Tensor(a!) out) -> Tensor(a!)
Tensor& _to_dim_order_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    bool non_blocking,
    OptionalArrayRef<int64_t> dim_order,
    Tensor& out) {
  // TODO(T181345875): enable sanity check in aten mode
  ET_KERNEL_CHECK(
      ctx,
      check__to_dim_order_copy_args(self, non_blocking, dim_order, out),
      InvalidArgument,
      out);

  Optional<MemoryFormat> memory_format = get_memory_format(dim_order);
  at::_to_copy_outf(self, non_blocking, memory_format, out);

  return out;
}

Tensor& _to_dim_order_copy_out(
    const Tensor& self,
    bool non_blocking,
    OptionalArrayRef<int64_t> dim_order,
    Tensor& out) {
  executorch::runtime::KernelRuntimeContext ctx{};
  return _to_dim_order_copy_out(ctx, self, non_blocking, dim_order, out);
}

} // namespace native
} // namespace executor
} // namespace torch
