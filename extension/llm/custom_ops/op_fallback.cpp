/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/extension/llm/custom_ops/op_fallback.h>
#include <cstring>

namespace torch {
namespace executor {

namespace native {

// Copy from op_clone.cpp
Tensor& fallback_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, in.sizes()) == torch::executor::Error::Ok,
      InvalidArgument,
      out);

  // The input and out shall share same dtype and size
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_shape_and_dtype(in, out), InvalidArgument, out);

  if (in.nbytes() > 0) {
    // Note that this check is important. It's valid for a tensor with numel 0
    // to have a null data pointer, but in some environments it's invalid to
    // pass a null pointer to memcpy() even when the size is zero.
    memcpy(out.mutable_data_ptr(), in.const_data_ptr(), in.nbytes());
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

EXECUTORCH_LIBRARY(
    llama,
    "fallback.out",
    torch::executor::native::fallback_out);
