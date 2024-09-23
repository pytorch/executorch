/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
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

// TODO(T179241236): Update core/exec_aten/util/tensor_util.h to support dim
// order other than contiguous.
int64_t coordinateToIndexWithDimOrder(
    const Tensor& self,
    const size_t* cur_indices) {
  int64_t index = 0;
  exec_aten::StridesType strides[kTensorDimensionLimit];
  SizesArrayRef sizes = self.sizes();
  DimOrderArrayRef dim_order = self.dim_order();

  dim_order_to_stride_nocheck(
      sizes.data(), dim_order.data(), sizes.size(), strides);
  for (size_t i = 0; i < self.dim(); ++i) {
    index += cur_indices[i] * strides[i];
  }
  return index;
}

template <typename SELF_CTYPE, typename OUT_CTYPE>
void _to_dim_order_copy_impl(const Tensor& self, Tensor& out) {
  auto self_data = self.mutable_data_ptr<SELF_CTYPE>();
  auto out_data = out.mutable_data_ptr<OUT_CTYPE>();

  size_t coordinate[kTensorDimensionLimit] = {0};

  // Copy data from self to out index by index. Same index in self and out
  // should have same value, no matter the order of dimensions.
  for (ssize_t i = 0; i < self.numel(); i++) {
    // Update the current indices.
    for (ssize_t j = self.dim() - 1; j >= 0; j--) {
      if (coordinate[j] + 1 < self.size(j)) {
        coordinate[j]++;
        break;
      } else {
        coordinate[j] = 0;
      }
    }
    // Get the corresponding index of self_data and out_data by stride.
    int64_t self_data_index = coordinateToIndexWithDimOrder(self, coordinate);
    int64_t out_data_index = coordinateToIndexWithDimOrder(out, coordinate);

    out_data[out_data_index] =
        static_cast<OUT_CTYPE>(self_data[self_data_index]);
  }
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
  (void)ctx;
  ET_KERNEL_CHECK(
      ctx,
      check__to_dim_order_copy_args(self, non_blocking, dim_order, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, self.sizes()) == torch::executor::Error::Ok,
      InvalidArgument,
      out);

  ET_SWITCH_REALHB_TYPES(
      self.scalar_type(),
      ctx,
      "dim_order_ops::_to_dim_order_copy.out",
      CTYPE_IN,
      [&] {
        ET_SWITCH_REALHB_TYPES(
            out.scalar_type(),
            ctx,
            "dim_order_ops::_to_dim_order_copy.out",
            CTYPE_OUT,
            [&] { _to_dim_order_copy_impl<CTYPE_IN, CTYPE_OUT>(self, out); });
      });

  return out;
}

Tensor& _to_dim_order_copy_out(
    const Tensor& self,
    bool non_blocking,
    OptionalArrayRef<int64_t> dim_order,
    Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return _to_dim_order_copy_out(context, self, non_blocking, dim_order, out);
}

} // namespace native
} // namespace executor
} // namespace torch
