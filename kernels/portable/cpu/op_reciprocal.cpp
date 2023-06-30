// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/pattern/pattern.h>

namespace torch {
namespace executor {
namespace native {
namespace {

double reciprocal(double x) {
  return 1.0 / x;
}

} // namespace

Tensor& reciprocal_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  return internal::unary_ufunc_realb_to_float(reciprocal, ctx, in, out);
}

} // namespace native
} // namespace executor
} // namespace torch
