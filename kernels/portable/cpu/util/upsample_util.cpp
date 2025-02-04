/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/upsample_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace torch {
namespace executor {

bool check_upsample_2d_common_args(
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t>& output_size,
    const executorch::aten::OptionalArrayRef<double>& scale_factors,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(in.dim() == 4);
  ET_LOG_AND_RETURN_IF_FALSE(out.dim() == 4);
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_dim_order(in));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_dim_order(out));
  ET_LOG_AND_RETURN_IF_FALSE(
      output_size.has_value() ^ scale_factors.has_value());
  if (scale_factors.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(scale_factors.value().size() == 2);
    ET_LOG_AND_RETURN_IF_FALSE(scale_factors.value()[0] > 0);
    ET_LOG_AND_RETURN_IF_FALSE(scale_factors.value()[1] > 0);
  } else if (output_size.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(output_size.value().size() == 2);
    ET_LOG_AND_RETURN_IF_FALSE(output_size.value()[0] > 0);
    ET_LOG_AND_RETURN_IF_FALSE(output_size.value()[1] > 0);
  }

  return true;
}

bool check_upsample_bilinear2d_args(
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t>& output_size,
    ET_UNUSED const bool align_corners,
    const executorch::aten::OptionalArrayRef<double>& scale_factors,
    Tensor& out) {
  return check_upsample_2d_common_args(in, output_size, scale_factors, out);
}

bool check_upsample_nearest2d_args(
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t>& output_size,
    const executorch::aten::OptionalArrayRef<double>& scale_factors,
    Tensor& out) {
  return check_upsample_2d_common_args(in, output_size, scale_factors, out);
}

Error resize_upsample_2d(
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t>& output_size,
    const executorch::aten::OptionalArrayRef<double>& scale_factors,
    double& scale_h_out,
    double& scale_w_out,
    Tensor& out) {
  // Either output_size or scale_factors are provided, not both. This
  // is checked in check_..._args.
  // Scales are transformed according to align_corners.
  std::array<Tensor::SizesType, kTensorDimensionLimit> target_size;

  const auto dim = in.dim();
  std::copy(in.sizes().cbegin(), in.sizes().cend(), target_size.begin());

  if (scale_factors.has_value()) {
    scale_h_out = scale_factors.value()[0];
    scale_w_out = scale_factors.value()[1];

    target_size[dim - 2] =
        static_cast<Tensor::SizesType>(in.sizes()[dim - 2] * scale_h_out);
    target_size[dim - 1] =
        static_cast<Tensor::SizesType>(in.sizes()[dim - 1] * scale_w_out);
  } else if (output_size.has_value()) {
    scale_h_out =
        static_cast<double>(output_size.value()[0]) / in.sizes()[dim - 2];
    scale_w_out =
        static_cast<double>(output_size.value()[1]) / in.sizes()[dim - 1];

    target_size[dim - 2] = output_size.value()[0];
    target_size[dim - 1] = output_size.value()[1];
  } else {
    ET_LOG(Error, "Invalid output_size or scale_factors");
    return Error::InvalidArgument;
  }

  ET_CHECK_OR_RETURN_ERROR(
      target_size[dim - 2] > 0 && target_size[dim - 1] > 0,
      InvalidArgument,
      "Upsampled output size must be non-empty, but was %ld x %ld.",
      static_cast<long>(target_size[dim - 2]),
      static_cast<long>(target_size[dim - 1]));

  return resize_tensor(out, {target_size.data(), static_cast<size_t>(dim)});
}

} // namespace executor
} // namespace torch
