/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/select_copy_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::Tensor;
using torch::executor::Error;
using torch::executor::KernelRuntimeContext;
using torch::executor::select_copy_util;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

Tensor& select_copy_int_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    int64_t index,
    Tensor& out) {
  Error err = select_copy_util(in, dim, index, out);
  if (err != Error::Ok) {
    ctx.fail(err);
  }
  return out;
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
