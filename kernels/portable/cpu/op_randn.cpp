/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <random>
#include <type_traits>

namespace torch::executor::native {

using executorch::aten::Tensor;

Tensor& randn_out(
    KernelRuntimeContext& context,
    IntArrayRef size,
    Tensor& out) {
  (void)context;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      context,
      resize_tensor(out, size) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  std::default_random_engine gen;
  ET_SWITCH_FLOATHBF16_TYPES(out.scalar_type(), ctx, "randn.out", CTYPE, [&]() {
    using dist_type = std::conditional_t<c10::is_reduced_floating_point_v<CTYPE>, float, CTYPE>;
    std::normal_distribution<dist_type> dist;
    std::generate_n(out.mutable_data_ptr<CTYPE>(), out.numel(), [&]() {
      return static_cast<CTYPE>(dist(gen));
    });
  });
  return out;
}

} // namespace torch::executor::native
