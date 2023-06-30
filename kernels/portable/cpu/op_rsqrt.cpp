// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/pattern/pattern.h>

namespace torch {
namespace executor {
namespace native {
namespace {

double rsqrt(double x) {
  return 1.0 / std::sqrt(x);
}

} // namespace

Tensor& rsqrt_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  return internal::unary_ufunc_realb_to_float(rsqrt, ctx, in, out);
}

} // namespace native
} // namespace executor
} // namespace torch
