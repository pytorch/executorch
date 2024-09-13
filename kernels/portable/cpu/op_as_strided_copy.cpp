/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& as_strided_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    optional<int64_t> storage_offset,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_as_strided_copy_args(in, size, stride, storage_offset, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, size) == torch::executor::Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  if (in.numel() == 0) {
    return out;
  }

  size_t offset = storage_offset.has_value() ? storage_offset.value() : 0;

  ET_SWITCH_ALL_TYPES(in.scalar_type(), ctx, __func__, CTYPE, [&] {
    as_strided_copy<CTYPE>(in, size, stride, offset, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
