/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;

namespace {

template <typename IN_CTYPE, typename OUT_CTYPE>
inline void _to_impl(const Tensor& self, Tensor& out) {
  auto self_data = self.const_data_ptr<IN_CTYPE>();
  auto out_data = out.mutable_data_ptr<OUT_CTYPE>();

  for (size_t i = 0, e = out.numel(); i < e; i++) {
    // Read real and imaginary parts from consecutive elements
    // Use the same pattern as view_as_real_copy but in reverse
    out_data[i].real_ =
        static_cast<decltype(out_data[i].real_)>(self_data[2 * i]);
    out_data[i].imag_ =
        static_cast<decltype(out_data[i].imag_)>(self_data[2 * i + 1]);
  }
}

} // namespace

// view_as_complex_copy(Tensor self) -> Tensor
Tensor& view_as_complex_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    Tensor& out) {
  (void)ctx;

  // The input tensor must have at least one dimension
  ET_KERNEL_CHECK_MSG(
      ctx,
      self.dim() > 0,
      InvalidArgument,
      out,
      "Input tensor must have at least one dimension");

  // The last dimension must be 2 (real and imaginary parts)
  ET_KERNEL_CHECK_MSG(
      ctx,
      self.size(self.dim() - 1) == 2,
      InvalidArgument,
      out,
      "Input tensor must have a last dimension of size 2");

  // The input tensor must be float type
  ET_KERNEL_CHECK_MSG(
      ctx,
      executorch::runtime::isFloatingType(self.scalar_type()),
      InvalidArgument,
      out,
      "Input tensor must be float type");

  // The output tensor must be complex type
  ET_KERNEL_CHECK_MSG(
      ctx,
      executorch::runtime::isComplexType(out.scalar_type()),
      InvalidArgument,
      out,
      "Output tensor must be complex type");

  // Get the expected output shape (input shape without the last dimension)
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  get_view_as_complex_copy_out_target_size(self, expected_output_size);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(
          out, {expected_output_size, static_cast<size_t>(out.dim())}) ==
          Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(self, out), InvalidArgument, out);

  static constexpr auto op_name = "view_as_complex_copy.out";

  ET_SWITCH_FLOATH_TYPES(self.scalar_type(), ctx, op_name, CTYPE_IN, [&] {
    ET_SWITCH_COMPLEXH_TYPES(out.scalar_type(), ctx, op_name, CTYPE_OUT, [&] {
      _to_impl<CTYPE_IN, CTYPE_OUT>(self, out);
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
