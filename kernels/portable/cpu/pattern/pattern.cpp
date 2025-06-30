/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/pattern.h>

namespace torch::executor::native::internal {

bool check_and_resize_inputs(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, false);
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      false,
      "Failed to resize output tensor.");
  return true;
}

} // namespace torch::executor::native::internal
