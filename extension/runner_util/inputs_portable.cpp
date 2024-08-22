/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/runner_util/inputs.h>

#include <algorithm>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/log.h>

using exec_aten::Tensor;
using exec_aten::TensorImpl;
using executorch::runtime::Error;
using executorch::runtime::Method;
using executorch::runtime::TensorInfo;

namespace executorch {
namespace extension {
namespace internal {

namespace {
/**
 * Sets all elements of a tensor to 1.
 */
Error fill_ones(torch::executor::Tensor tensor) {
#define FILL_CASE(T, n)                                \
  case (torch::executor::ScalarType::n):               \
    std::fill(                                         \
        tensor.mutable_data_ptr<T>(),                  \
        tensor.mutable_data_ptr<T>() + tensor.numel(), \
        1);                                            \
    break;

  switch (tensor.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND(Bool, FILL_CASE)
    default:
      ET_LOG(Error, "Unsupported scalar type %d", (int)tensor.scalar_type());
      return Error::InvalidArgument;
  }

#undef FILL_CASE

  return Error::Ok;
}
} // namespace

Error fill_and_set_input(
    Method& method,
    TensorInfo& tensor_meta,
    size_t input_index,
    void* data_ptr) {
  TensorImpl impl = TensorImpl(
      tensor_meta.scalar_type(),
      /*dim=*/tensor_meta.sizes().size(),
      // These const pointers will not be modified because we never resize this
      // short-lived TensorImpl. It only exists so that set_input() can verify
      // that the shape is correct; the Method manages its own sizes and
      // dim_order arrays for the input.
      const_cast<TensorImpl::SizesType*>(tensor_meta.sizes().data()),
      data_ptr,
      const_cast<TensorImpl::DimOrderType*>(tensor_meta.dim_order().data()));
  Tensor t(&impl);
  ET_CHECK_OK_OR_RETURN_ERROR(fill_ones(t));
  return method.set_input(t, input_index);
}

} // namespace internal
} // namespace extension
} // namespace executorch
