/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

namespace {

template <typename SELF_CTYPE, typename OUT_CTYPE>
inline void _to_impl(const Tensor& self, Tensor& out) {
  auto self_data = self.mutable_data_ptr<SELF_CTYPE>();
  auto out_data = out.mutable_data_ptr<OUT_CTYPE>();

  for (size_t i = 0, e = self.numel(); i < e; i++) {
    auto val_in = self_data[i];
    out_data[2 * i] = static_cast<OUT_CTYPE>(val_in.real_);
    out_data[2 * i + 1] = static_cast<OUT_CTYPE>(val_in.imag_);
  }
}

} // namespace

// view_as_real_copy(Tensor self) -> Tensor
Tensor& view_as_real_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    Tensor& out) {
  (void)ctx;

  // Get the output shape
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  get_view_as_real_copy_out_target_size(self, expected_output_size);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(
          out, {expected_output_size, static_cast<size_t>(out.dim())}) ==
          Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  // The input tensor must be complex type
  ET_KERNEL_CHECK_MSG(
      ctx,
      executorch::runtime::isComplexType(self.scalar_type()),
      InvalidArgument,
      out,
      "Input tensor must be complex type");

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(self, out), InvalidArgument, out);

  static constexpr auto op_name = "view_as_real_copy.out";

  ET_SWITCH_COMPLEXH_TYPES(self.scalar_type(), ctx, op_name, CTYPE_IN, [&] {
    ET_SWITCH_FLOATH_TYPES(out.scalar_type(), ctx, op_name, CTYPE_OUT, [&] {
      _to_impl<CTYPE_IN, CTYPE_OUT>(self, out);
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
