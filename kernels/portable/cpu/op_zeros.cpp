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

namespace {

bool check_sizes(
    exec_aten::ArrayRef<int64_t> size_int64_t,
    exec_aten::ArrayRef<int32_t> size_int32_t) {
  ET_LOG_AND_RETURN_IF_FALSE(size_int64_t.size() == size_int32_t.size());
  for (int i = 0; i < size_int64_t.size(); i++) {
    ET_LOG_AND_RETURN_IF_FALSE(((int64_t)size_int32_t[i] == size_int64_t[i]));
  }

  return true;
}

} // namespace

/*
 * Zero the out tensor
 *
 * zeros.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
 */
Tensor& zeros_out(KernelRuntimeContext& ctx, IntArrayRef size, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, size) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(ctx, check_sizes(size, out.sizes()), InvalidArgument, out);

  void* out_data = out.mutable_data_ptr();
  if (out_data != nullptr) {
    /*
     * Assuming storage is contiguous and zero tensor is indeed a string of
     * zeros
     */
    memset(out_data, 0, out.nbytes());
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
