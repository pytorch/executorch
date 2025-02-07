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
// NOLINTEND(facebook-hte-ConstantArgumentPassByValue,
// facebook-hte-ParameterMightThrowOnCopy)

TORCH_LIBRARY(et_test, m) {
  m.def(
      "upsample_bilinear2d.vec_out(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors, *, Tensor(a!) out) -> Tensor(a!)",
      WRAP_TO_ATEN(upsample_bilinear2d_vec_out_no_context, 4));
  m.def(
      "upsample_nearest2d.vec_out(Tensor input, SymInt[]? output_size, float[]? scale_factors, *, Tensor(a!) out) -> Tensor(a!)",
      WRAP_TO_ATEN(upsample_nearest2d_vec_out_no_context, 3));
}

} // namespace native
} // namespace executor
} // namespace torch
