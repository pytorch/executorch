/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
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

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] =
      "dim_order_ops::_to_dim_order_copy.out";

  const bool in_is_complex =
      executorch::runtime::isComplexType(self.scalar_type());
  const bool out_is_complex =
      executorch::runtime::isComplexType(out.scalar_type());

  if (in_is_complex && out_is_complex) {
    // Complex to complex: same type copy
    ET_SWITCH_COMPLEXH_TYPES(
        self.scalar_type(), ctx, op_name, CTYPE, [&] {
          _to_dim_order_copy_impl<CTYPE, CTYPE>(self, out);
        });
  } else if (!in_is_complex && out_is_complex) {
    // Real to complex: convert real value to complex with zero imaginary part
    ET_SWITCH_FLOATH_TYPES(
        self.scalar_type(), ctx, op_name, CTYPE_IN, [&] {
          ET_SWITCH_COMPLEXH_TYPES(
              out.scalar_type(), ctx, op_name, CTYPE_OUT, [&] {
                auto self_data = self.mutable_data_ptr<CTYPE_IN>();
                auto out_data = out.mutable_data_ptr<CTYPE_OUT>();
                for (const auto [unused_index, self_data_index, out_data_index] :
                     BroadcastIndexesRange<
                         2,
                         /*support_noncontiguous_input_tensors=*/true>(
                         /*dummy output*/ self, self, out)) {
                  (void)unused_index;
                  out_data[out_data_index].real_ =
                      self_data[self_data_index];
                  out_data[out_data_index].imag_ = 0;
                }
              });
        });
  } else if (in_is_complex && !out_is_complex) {
    // Complex to real: take real part
    ET_SWITCH_COMPLEXH_TYPES(
        self.scalar_type(), ctx, op_name, CTYPE_IN, [&] {
          ET_SWITCH_FLOATH_TYPES(
              out.scalar_type(), ctx, op_name, CTYPE_OUT, [&] {
                auto self_data = self.mutable_data_ptr<CTYPE_IN>();
                auto out_data = out.mutable_data_ptr<CTYPE_OUT>();
                for (const auto [unused_index, self_data_index, out_data_index] :
                     BroadcastIndexesRange<
                         2,
                         /*support_noncontiguous_input_tensors=*/true>(
                         /*dummy output*/ self, self, out)) {
                  (void)unused_index;
                  out_data[out_data_index] =
                      static_cast<CTYPE_OUT>(self_data[self_data_index].real_);
                }
              });
        });
  } else {
    // Real to real
    ET_SWITCH_REALHBBF16_TYPES(
        self.scalar_type(), ctx, op_name, CTYPE_IN, [&] {
          ET_SWITCH_REALHBBF16_TYPES(
              out.scalar_type(), ctx, op_name, CTYPE_OUT, [&] {
                _to_dim_order_copy_impl<CTYPE_IN, CTYPE_OUT>(self, out);
              });
        });
  }

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
