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
Tensor& opt_fft_r2c_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    Tensor& out) {
  auto in_sizes = in.sizes();
  ET_KERNEL_CHECK(ctx, in.dim() <= kTensorDimensionLimit, InvalidArgument, out);

  std::array<Tensor::SizesType, kTensorDimensionLimit> out_sizes_storage;
  executorch::runtime::Span<Tensor::SizesType> out_sizes(
      out_sizes_storage.data(), in_sizes.size());
  std::copy(in_sizes.begin(), in_sizes.end(), out_sizes.begin());
  ET_KERNEL_CHECK(ctx, !dim.empty(), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      onesided,
      InvalidArgument,
      out,
      "onesided=False is not supported yet in _fft_r2c");

  ET_KERNEL_CHECK_MSG(
      ctx,
      out.scalar_type() == executorch::runtime::toComplexType(in.scalar_type()),
      InvalidArgument,
      out,
      "the output type for _fft_r2c must be the Complex type corresponding to the input type");

  for (auto d : dim) {
    ET_KERNEL_CHECK_MSG(
        ctx,
        d >= 0 && d < in.dim(),
        InvalidArgument,
        out,
        "dims must be in bounds (got %" PRId64 ")",
        d);
  }

  if (onesided) {
    out_sizes[dim.back()] = out_sizes[dim.back()] / 2 + 1;
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
  auto in_shape = shape_from_tensor(in);
  // TODO: if arbitrary strides are a possibility, we need to validate
  // these, because pocketfft README says "Strides that lead to
  // multiple accesses of the same memory address are not allowed."
  auto in_stride = stride_from_tensor(in);
  auto out_stride = stride_from_tensor(out);
  // NOTE: as of this writing, upstream PyTorch only supports
  // float/double, so we follow suit.
  ET_SWITCH_FLOAT_TYPES(in.scalar_type(), ctx, "_fft_r2c.out", CTYPE_IN, [&] {
    auto fct = compute_fct<CTYPE_IN>(ctx, in, dim, normalization);
    if (!fct) {
      // Check failed, just bail out of the lambda.
      return;
    }
    pocketfft::r2c<CTYPE_IN>(
        in_shape,
        in_stride,
        out_stride,
        axes,
        true,
        in.const_data_ptr<CTYPE_IN>(),
        tensor_cdata<CTYPE_IN>(out),
        *fct);

    // TODO: fill with conjugate symmetry if not onesided; see
    // ATen/native/mkl/SpectralOps.cpp
  });
  return out;
}
} // namespace torch::executor::native
