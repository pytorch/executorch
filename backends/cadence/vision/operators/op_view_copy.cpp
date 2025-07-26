/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>

namespace cadence {
namespace impl {
namespace MV130 {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using ::executorch::aten::IntArrayRef;

Tensor& view_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const IntArrayRef size,
    Tensor& out) {
  memcpy(out.mutable_data_ptr(), input.const_data_ptr(), input.nbytes());
  return out;
}

} // namespace native
} // namespace MV130
} // namespace impl
} // namespace cadence
