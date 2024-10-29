/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>

namespace custom {
namespace native {

using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

namespace {
void check_preconditions(const Tensor& in, Tensor& out) {
  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Float,
      "Expected out tensor to have dtype Float, but got %d instead",
      static_cast<int>(out.scalar_type()));
  ET_CHECK_MSG(
      in.scalar_type() == ScalarType::Float,
      "Expected in tensor to have dtype Float, but got %d instead",
      static_cast<int>(in.scalar_type()));
  ET_CHECK_MSG(
      out.dim() == in.dim(),
      "Number of dims of out tensor is not compatible with inputs");
  ET_CHECK_MSG(
      out.numel() == in.numel(),
      "Number of elements of out tensor %zd is not compatible with inputs %zd",
      ssize_t(out.numel()),
      ssize_t(in.numel()));
}
} // namespace

// mul4.out(Tensor input, *, Tensor(a!) output) -> Tensor(a!)
// ATen-compatible function signature, without a KernelRuntimeContext.
Tensor& mul4_out_impl(const Tensor& in, Tensor& out) {
  check_preconditions(in, out);
  float* out_data = out.mutable_data_ptr<float>();
  const float* in_data = in.const_data_ptr<float>();
  for (size_t out_idx = 0; out_idx < out.numel(); ++out_idx) {
    out_data[out_idx] = in_data[out_idx] * 4;
  }
  return out;
}

// mul4.out(Tensor input, *, Tensor(a!) output) -> Tensor(a!)
// ExecuTorch-compatible function signature, with a KernelRuntimeContext.
Tensor& mul4_out_impl(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out) {
  mul4_out_impl(in, out);
  return out;
}

} // namespace native
} // namespace custom
