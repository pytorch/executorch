/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/optimized/cpu/fft_utils.h>
#include <executorch/runtime/core/span.h>

namespace torch::executor::native {
Tensor& opt_fft_c2r_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size,
    Tensor& out) {
  auto in_sizes = in.sizes();
  ET_KERNEL_CHECK(ctx, in.dim() <= kTensorDimensionLimit, InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, !dim.empty(), InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, last_dim_size >= 1, InvalidArgument, out);

  // Determine the output size
  std::array<Tensor::SizesType, kTensorDimensionLimit> out_sizes_storage{};
  executorch::runtime::Span<Tensor::SizesType> out_sizes(
      out_sizes_storage.data(), in_sizes.size());
  std::copy(in_sizes.begin(), in_sizes.end(), out_sizes.begin());
  out_sizes[dim.back()] = last_dim_size;

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      in.scalar_type() == executorch::runtime::toComplexType(out.scalar_type()),
      InvalidArgument,
      out,
      "the input type for _fft_c2r must be the Complex type corresponding to the output type");

  for (auto d : dim) {
    ET_KERNEL_CHECK_MSG(
        ctx,
        d >= 0 && d < in.dim(),
        InvalidArgument,
        out,
        "dims must be in bounds (got %" PRId64 ")",
        d);
  }

  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(
          out,
          executorch::runtime::ArrayRef<Tensor::SizesType>(
              out_sizes.data(), out_sizes.size())) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor (last dim %d).",
      out_sizes[dim.back()]);

  pocketfft::shape_t axes(dim.begin(), dim.end());
  auto out_shape = shape_from_tensor(out);
  // TODO: if arbitrary strides are a possibility, we need to validate
  // these, because pocketfft README says "Strides that lead to
  // multiple accesses of the same memory address are not allowed."
  auto in_stride = stride_from_tensor(in);
  auto out_stride = stride_from_tensor(out);
  // NOTE: as of this writing, upstream PyTorch only supports
  // float/double, so we follow suit.
  ET_SWITCH_FLOAT_TYPES(out.scalar_type(), ctx, "_fft_c2r.out", CTYPE_OUT, [&] {
    auto fct = compute_fct<CTYPE_OUT>(ctx, out, dim, normalization);
    if (!fct) {
      // Check failed, just bail out of the lambda.
      return;
    }
    pocketfft::c2r<CTYPE_OUT>(
        out_shape,
        in_stride,
        out_stride,
        axes,
        false /* forward */,
        tensor_cdata<CTYPE_OUT>(in),
        out.mutable_data_ptr<CTYPE_OUT>(),
        *fct);
  });
  return out;
}
} // namespace torch::executor::native
