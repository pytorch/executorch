// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>

namespace torch {
namespace executor {
namespace native {

Tensor& asin_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  return internal::unary_ufunc_realb_to_float(std::asin, ctx, in, out);
}

} // namespace native
} // namespace executor
} // namespace torch
