/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/upsample_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>

#include <algorithm>

#if defined(__aarch64__)
#include <executorch/kernels/optimized/cpu/op_upsample_nearest2d_neon.h>
#endif

namespace torch {
namespace executor {
namespace native {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;

Tensor& upsample_nearest2d_vec_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    executorch::aten::OptionalArrayRef<int64_t> output_size,
    executorch::aten::OptionalArrayRef<double> scale_factors,
    Tensor& out);

Tensor& opt_upsample_nearest2d_vec_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t> output_size,
    const executorch::aten::OptionalArrayRef<double> scale_factors,
    Tensor& out) {
#if !defined(__aarch64__)
  return upsample_nearest2d_vec_out(ctx, in, output_size, scale_factors, out);
#else
  if (in.scalar_type() != ScalarType::Half ||
      !tensor_is_default_dim_order(in) || !tensor_is_default_dim_order(out) ||
      !tensor_is_contiguous(in) || !tensor_is_contiguous(out) ||
      !check_upsample_nearest2d_args(in, output_size, scale_factors, out)) {
    return upsample_nearest2d_vec_out(ctx, in, output_size, scale_factors, out);
  }

  double scale_h;
  double scale_w;
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_upsample_2d(
          in, output_size, scale_factors, scale_h, scale_w, out) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor");

  const auto kernel_scale_h =
      area_pixel_compute_scale<double>(in.size(2), out.size(2), false, scale_h);
  const auto kernel_scale_w =
      area_pixel_compute_scale<double>(in.size(3), out.size(3), false, scale_w);
  if (!tensor_is_contiguous(out) || kernel_scale_h != 0.5 ||
      kernel_scale_w != 0.5 || out.size(2) != in.size(2) * 2 ||
      out.size(3) != in.size(3) * 2) {
    return upsample_nearest2d_vec_out(ctx, in, output_size, scale_factors, out);
  }

  const int64_t output_plane_size = out.size(2) * out.size(3);
  const int64_t grain_size = std::max<int64_t>(
      1, ::executorch::extension::internal::GRAIN_SIZE / output_plane_size);
  const int64_t planes = in.size(0) * in.size(1);
  const auto* input = reinterpret_cast<const uint16_t*>(in.const_data_ptr());
  auto* output = reinterpret_cast<uint16_t*>(out.mutable_data_ptr());

  const bool success = ::executorch::extension::parallel_for(
      0, planes, grain_size, [&](const auto begin, const auto end) {
        opt_upsample_nearest2d_internal::upsample_nearest2d_2x_nchw_u16(
            input, output, begin, end, in.size(2), in.size(3));
      });
  if (!success) {
    return upsample_nearest2d_vec_out(ctx, in, output_size, scale_factors, out);
  }
  return out;
#endif
}

} // namespace native
} // namespace executor
} // namespace torch
