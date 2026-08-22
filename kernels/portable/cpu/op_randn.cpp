/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <random>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::IntArrayRef;
using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

Tensor&
randn_out(KernelRuntimeContext& ctx, const IntArrayRef sizes, Tensor& out) {
  (void)ctx;

  std::mt19937 gen((std::random_device())());
  std::normal_distribution<double> dist(0.0, 1.0);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, sizes) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_SWITCH_FLOATHBF16_TYPES(out.scalar_type(), ctx, "randn.out", CTYPE, [&] {
    auto data_out = out.mutable_data_ptr<CTYPE>();
    for (const auto i : c10::irange(out.numel())) {
      data_out[i] = static_cast<CTYPE>(dist(gen));
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
