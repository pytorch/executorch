/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "executorch/backends/cadence/generic/operators/op_idma_wait.h"

#include <cstdint>

#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/core/exec_aten/util/tensor_util.h"
#include "executorch/runtime/kernel/kernel_runtime_context.h"

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

// CPU implementation of idma_wait_out
// Since there's no actual DMA operation in the CPU implementation,
// this is essentially a no-op function that just ensures the output tensor
// has the same content as the input tensor
Tensor& idma_wait_out(
    KernelRuntimeContext& ctx,
    const Tensor& src,
    const int64_t
        task_num, // Unused in CPU implementation but kept for API compatibility
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, src.numel() == out.numel(), InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, src.dtype() == out.dtype(), InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx,
      src.const_data_ptr<uint8_t>() == out.const_data_ptr<uint8_t>(),
      InvalidArgument,
      out);

  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
