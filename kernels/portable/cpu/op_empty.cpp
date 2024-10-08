/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <cstdint>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Tensor;

/*
 * Empty out tensor
 *
 * empty.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
 */
Tensor& empty_out(
    KernelRuntimeContext& context,
    IntArrayRef size,
    exec_aten::optional<exec_aten::MemoryFormat> memory_format,
    Tensor& out) {
  (void)context;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      context,
      resize_tensor(out, size) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
