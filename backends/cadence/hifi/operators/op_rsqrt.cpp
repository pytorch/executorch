/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>

using executorch::aten::RuntimeContext;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;

namespace impl {
namespace HiFi {
namespace native {
namespace {

template <typename T>
T rsqrt(T x) {
  return 1.0 / std::sqrt(x);
}

} // namespace

Tensor& rsqrt_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  bool optimized = true;

  if (out.scalar_type() != ScalarType::Float)
    optimized = false;

  if (optimized) {
    WORD32 num_elm = out.numel();

    FLOAT32* __restrict__ p_out =
        (FLOAT32* __restrict__)out.mutable_data_ptr<float>();
    const FLOAT32* __restrict__ p_inp =
        (const FLOAT32* __restrict__)in.const_data_ptr<float>();

    xa_nn_elm_rsqrt_f32_f32(p_out, p_inp, num_elm);
    return out;
  }

  return torch::executor::native::internal::
      unary_ufunc_realhbbf16_to_floathbf16(rsqrt, rsqrt, ctx, in, out);
}

} // namespace native
} // namespace HiFi
} // namespace impl
