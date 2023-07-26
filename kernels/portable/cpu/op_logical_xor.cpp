// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>

namespace torch {
namespace executor {
namespace native {
namespace {

bool logical_xor(bool a, bool b) {
  return a != b;
}

} // namespace

Tensor& logical_xor_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  return internal::binary_ufunc_realb_realb_to_realb_logical(
      logical_xor, ctx, a, b, out);
}

} // namespace native
} // namespace executor
} // namespace torch
