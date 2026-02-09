/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

namespace {

void alias_call(
    KernelRuntimeContext& ctx,
    executorch::runtime::Span<executorch::runtime::EValue*> stack) {
  (void)ctx;
  // Stack layout: x, y, out_x, out_y
  Tensor x = stack[0]->toTensor();
  Tensor y = stack[1]->toTensor();
  Tensor& out_x = stack[2]->toTensor();
  Tensor& out_y = stack[3]->toTensor();

  ET_CHECK_MSG(
      tensors_have_same_shape_and_dtype(x, out_x),
      "x and out_x must have same shape and dtype.");
  ET_CHECK_MSG(
      tensors_have_same_shape_and_dtype(y, out_y),
      "y and out_y must have same shape and dtype.");
  ET_CHECK_MSG(
      tensors_have_same_dim_order(x, out_x),
      "x and out_x must have same dim order.");
  ET_CHECK_MSG(
      tensors_have_same_dim_order(y, out_y),
      "y and out_y must have same dim order.");

  if (x.nbytes() > 0) {
    std::memcpy(out_x.mutable_data_ptr(), x.const_data_ptr(), x.nbytes());
  }
  if (y.nbytes() > 0) {
    std::memcpy(out_y.mutable_data_ptr(), y.const_data_ptr(), y.nbytes());
  }
}

} // anonymous namespace

} // namespace native
} // namespace executor
} // namespace torch

// Manual registration for multi-output op
static auto alias_registration = ::executorch::runtime::register_kernel(
    ::executorch::runtime::Kernel(
        "executorch::alias.out",
        torch::executor::native::alias_call));

// Export a function to explicitly register the kernels
extern "C" void register_custom_alias_ops() {
  // Force registration by calling this function
  (void)alias_registration;
}
