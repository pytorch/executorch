/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/prim_ops/et_select.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

using executorch::aten::SizesType;
using executorch::aten::Tensor;
using torch::executor::Error;
using torch::executor::resize_tensor;

namespace torch {
namespace executor {
namespace function {

constexpr size_t kTensorDimensionLimit = 16;

void et_select(KernelRuntimeContext& context, Span<EValue*> stack) {
  // executorch_prim::et_select.default(Tensor self, int dim, int index) -> out
  ET_KERNEL_CHECK_MSG(
      context,
      stack.size() == 4,
      InvalidProgram,
      /* void */,
      "Expected %zu args, got %zu",
      (size_t)4,
      stack.size());

  auto self = (*stack[0]).toTensor();
  auto dim = (*stack[1]).toInt();
  auto index = (*stack[2]).toInt();
  auto out = (*stack[3]).toTensor();

  ET_KERNEL_CHECK(
      context, tensors_have_same_dtype(self, out), InvalidArgument, );

  ET_KERNEL_CHECK_MSG(
      context,
      dim >= 0 && dim < self.dim(),
      InvalidArgument,
      ,
      "dim %" PRId64 " out of range for tensor with %" PRId64 " dimensions",
      dim,
      static_cast<int64_t>(self.dim()));

  // Normalize negative index (aten.select semantics)
  int64_t dim_size = self.size(dim);
  if (index < 0) {
    index += dim_size;
  }

  ET_KERNEL_CHECK_MSG(
      context,
      index >= 0 && index < dim_size,
      InvalidArgument,
      ,
      "index %" PRId64 " out of range for dim %" PRId64 " with size %" PRId64,
      index,
      dim,
      static_cast<int64_t>(dim_size));

  // Compute output sizes: self.sizes() with the selected dim removed.
  SizesType expected_output_size[kTensorDimensionLimit];
  int out_dims = 0;
  for (int i = 0; i < self.dim(); i++) {
    if (i != dim) {
      expected_output_size[out_dims++] = self.size(i);
    }
  }

  ET_KERNEL_CHECK_MSG(
      context,
      out_dims == out.dim(),
      InvalidArgument,
      ,
      "Expected output to have %d dims, got %" PRId64,
      out_dims,
      static_cast<int64_t>(out.dim()));

  ET_KERNEL_CHECK_MSG(
      context,
      resize_tensor(
          out, {expected_output_size, static_cast<size_t>(out_dims)}) ==
          Error::Ok,
      Internal,
      ,
      "Failed to resize output tensor.");

  // Compute byte offset: index * stride_at_dim * element_size
  auto stride_at_dim = self.strides()[dim];
  ssize_t byte_offset =
      static_cast<ssize_t>(index) * stride_at_dim * self.element_size();

  ET_KERNEL_CHECK_MSG(
      context,
      internal::set_tensor_data(
          out,
          static_cast<uint8_t*>(self.mutable_data_ptr()) + byte_offset,
          out.nbytes()) == Error::Ok,
      Internal,
      ,
      "Failed to set data_ptr for out to self + offset.");
}

} // namespace function
} // namespace executor
} // namespace torch
