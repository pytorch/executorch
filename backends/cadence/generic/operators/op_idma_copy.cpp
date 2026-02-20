/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_idma_copy.h>

#include <cstdint>
#include <cstring> // For std::memcpy

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

// CPU implementation of idma_copy_out using std::memcpy
// This function performs a direct memory copy between tensors
Tensor& idma_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& src,
    const int64_t
        task_num, // Unused in CPU implementation but kept for API compatibility
    const int64_t
        channel, // Unused in CPU implementation but kept for API compatibility
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      src.dtype() == out.dtype() && src.numel() == out.numel(),
      InvalidArgument,
      out);

  // Use std::memcpy for direct memory copy
  std::memcpy(
      out.mutable_data_ptr<uint8_t>(),
      src.const_data_ptr<uint8_t>(),
      out.nbytes());

  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
