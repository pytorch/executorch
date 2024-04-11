/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include "kernels.h"

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using RuntimeContext = torch::executor::RuntimeContext;

Tensor& view_copy_out(
    RuntimeContext& ctx,
    const Tensor& input,
    const IntArrayRef size,
    Tensor& out) {
  impl::HiFi::kernels::memcpy(
      out.mutable_data_ptr(), input.const_data_ptr(), input.nbytes());
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
