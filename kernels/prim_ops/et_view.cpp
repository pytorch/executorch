/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/prim_ops/et_view.h>

#include <cstring>

#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/assert.h>

using exec_aten::SizesType;
using exec_aten::Tensor;
using torch::executor::Error;
using torch::executor::resize_tensor;

namespace torch {
namespace executor {
namespace function {

constexpr size_t kTensorDimensionLimit = 16;

namespace {
bool get_view_target_size(
    const exec_aten::Tensor self,
    exec_aten::ArrayRef<int64_t> size,
    int64_t dim,
    exec_aten::SizesType* out_size) {
  ET_LOG_AND_RETURN_IF_FALSE(size.size() == dim);
  int minus1_dim = -1;
  int n_zero = 0;
  int64_t numel_without_minus_1 = 1;
  for (int i = 0; i < dim; i++) {
    if (size[i] == -1) {
      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          minus1_dim == -1, "At most one view dim can be -1.");
      minus1_dim = i;
    } else {
      // The size[i] must be non-negative now, but we check size[i] >= -1
      // in case code is reordered in the future.
      ET_LOG_MSG_AND_RETURN_IF_FALSE(
          size[i] >= -1, "Negative sizes are not allowed.");

      numel_without_minus_1 *= size[i];
      out_size[i] = static_cast<exec_aten::SizesType>(size[i]);

      if (size[i] == 0) {
        n_zero++;
      }
    }
  }
  if (minus1_dim >= 0) {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        n_zero == 0, "Cannot infer dimension size if there is a zero dim.");
    out_size[minus1_dim] = self.numel() / numel_without_minus_1;
  }
  return true;
}
} // namespace

void et_view(KernelRuntimeContext& context, EValue** stack) {
  (void)context;

  auto self = (*stack[0]).toTensor();
  auto size = (*stack[1]).toIntList();
  auto out = (*stack[2]).toTensor();

  ET_CHECK(tensors_have_same_dtype(self, out));

  // Compute output size
  SizesType expected_output_size[kTensorDimensionLimit];
  ET_CHECK(get_view_target_size(self, size, out.dim(), expected_output_size));

  // Resize for dynamic shape
  ET_CHECK_MSG(
      resize_tensor(
          out, {expected_output_size, static_cast<size_t>(out.dim())}) ==
          Error::Ok,
      "Failed to resize output tensor.");

  // Do some checks
  ET_CHECK(self.numel() == out.numel());

  // Update data ptr
  ET_CHECK_MSG(
      internal::set_tensor_data(
          out,
          /*buffer=*/self.mutable_data_ptr(),
          /*buffer_size=*/out.nbytes()) == Error::Ok,
      "Failed to set data_ptr for out to self.");
}

} // namespace function
} // namespace executor
} // namespace torch
