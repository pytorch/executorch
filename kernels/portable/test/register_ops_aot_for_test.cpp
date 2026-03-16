/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/runtime.h>

#include <torch/library.h>

namespace torch {
namespace executor {

namespace native {

// Signatures are auto-generated, so disable pass-by-value lint.
// NOLINTBEGIN(facebook-hte-ConstantArgumentPassByValue,
// facebook-hte-ParameterMightThrowOnCopy)
Tensor& upsample_bilinear2d_vec_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t> output_size,
    bool align_corners,
    const executorch::aten::OptionalArrayRef<double> scale_factors,
    Tensor& out);

Tensor& upsample_bilinear2d_vec_out_no_context(
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t> output_size,
    bool align_corners,
    const executorch::aten::OptionalArrayRef<double> scale_factors,
    Tensor& out) {
  KernelRuntimeContext ctx;
  auto& ret = upsample_bilinear2d_vec_out(
      ctx, in, output_size, align_corners, scale_factors, out);

  if (ctx.failure_state() != Error::Ok) {
    throw std::runtime_error(
        std::string("Kernel failed with error: ") +
        std::to_string((int)ctx.failure_state()));
  }

  return ret;
}

Tensor& upsample_nearest2d_vec_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t> output_size,
    const executorch::aten::OptionalArrayRef<double> scale_factors,
    Tensor& out);

Tensor& upsample_nearest2d_vec_out_no_context(
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t> output_size,
    const executorch::aten::OptionalArrayRef<double> scale_factors,
    Tensor& out) {
  KernelRuntimeContext ctx;
  auto& ret =
      upsample_nearest2d_vec_out(ctx, in, output_size, scale_factors, out);

  if (ctx.failure_state() != Error::Ok) {
    throw std::runtime_error(
        std::string("Kernel failed with error: ") +
        std::to_string((int)ctx.failure_state()));
  }

  return ret;
}

Tensor& _upsample_bilinear2d_aa_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const executorch::aten::ArrayRef<int64_t> output_size,
    bool align_corners,
    const std::optional<double> scale_h,
    const std::optional<double> scale_w,
    Tensor& out);

Tensor& _upsample_bilinear2d_aa_out_no_context(
    const Tensor& in,
    const executorch::aten::ArrayRef<int64_t> output_size,
    bool align_corners,
    const std::optional<double> scale_h,
    const std::optional<double> scale_w,
    Tensor& out) {
  KernelRuntimeContext ctx;
  auto& ret = _upsample_bilinear2d_aa_out(
      ctx, in, output_size, align_corners, scale_h, scale_w, out);

  if (ctx.failure_state() != Error::Ok) {
    throw std::runtime_error(
        std::string("Kernel failed with error: ") +
        std::to_string((int)ctx.failure_state()));
  }

  return ret;
}

Tensor& grid_sampler_2d_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    Tensor& out);

Tensor& grid_sampler_2d_out_no_context(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    Tensor& out) {
  KernelRuntimeContext ctx;
  auto& ret = grid_sampler_2d_out(
      ctx, input, grid, interpolation_mode, padding_mode, align_corners, out);

  if (ctx.failure_state() != Error::Ok) {
    throw std::runtime_error(
        std::string("Kernel failed with error: ") +
        std::to_string((int)ctx.failure_state()));
  }

  return ret;
}
// NOLINTEND(facebook-hte-ConstantArgumentPassByValue,
// facebook-hte-ParameterMightThrowOnCopy)

TORCH_LIBRARY(et_test, m) {
  m.def(
      "upsample_bilinear2d.vec_out(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors, *, Tensor(a!) out) -> Tensor(a!)",
      WRAP_TO_ATEN(upsample_bilinear2d_vec_out_no_context, 4));
  m.def(
      "upsample_nearest2d.vec_out(Tensor input, SymInt[]? output_size, float[]? scale_factors, *, Tensor(a!) out) -> Tensor(a!)",
      WRAP_TO_ATEN(upsample_nearest2d_vec_out_no_context, 3));
  m.def(
      "_upsample_bilinear2d_aa.out(Tensor input, SymInt[] output_size, bool align_corners, float? scale_h, float? scale_w, *, Tensor(a!) out) -> Tensor(a!)",
      WRAP_TO_ATEN(_upsample_bilinear2d_aa_out_no_context, 5));
  m.def(
      "grid_sampler_2d.out(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)",
      WRAP_TO_ATEN(grid_sampler_2d_out_no_context, 5));
}

} // namespace native
} // namespace executor
} // namespace torch
