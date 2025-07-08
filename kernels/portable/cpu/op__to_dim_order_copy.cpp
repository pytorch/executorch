/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using SizesArrayRef = executorch::aten::ArrayRef<executorch::aten::SizesType>;
using DimOrderArrayRef =
    executorch::aten::ArrayRef<executorch::aten::DimOrderType>;
using MemoryFormat = executorch::aten::MemoryFormat;

template <typename T>
using OptionalArrayRef = executorch::aten::OptionalArrayRef<T>;

template <typename T>
using Optional = std::optional<T>;

namespace {

template <typename SELF_CTYPE, typename OUT_CTYPE>
void _to_dim_order_copy_impl(const Tensor& self, Tensor& out) {
  auto self_data = self.mutable_data_ptr<SELF_CTYPE>();
  auto out_data = out.mutable_data_ptr<OUT_CTYPE>();

  // Here we make a slightly off-label use of
  // BroadcastIndexesRange. It always assumes it doesn't have to care
  // about different dim_order between input and output, but we can
  // just force it to respect strides (and thus dim_order) for its
  // inputs using support_noncontiguous_input_tensors=true, and then pretend
  // the output is just another input.
  for (const auto [unused_index, self_data_index, out_data_index] :
       BroadcastIndexesRange<2, /*support_noncontiguous_input_tensors=*/true>(
           /*dummy output*/ self, self, out)) {
    (void)unused_index;
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

  if (self.numel() == 0) {
    return out;
  }

  ET_SWITCH_REALHBBF16_TYPES(
      self.scalar_type(),
      ctx,
      "dim_order_ops::_to_dim_order_copy.out",
      CTYPE_IN,
      [&] {
        ET_SWITCH_REALHBBF16_TYPES(
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
  executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext context{};
  return _to_dim_order_copy_out(context, self, non_blocking, dim_order, out);
}

} // namespace native
} // namespace executor
} // namespace torch
