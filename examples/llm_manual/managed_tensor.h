/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/assert.h>

#include <executorch/runtime/core/portable_type/tensor.h>

#pragma once

namespace torch {
namespace executor {

/**
 * A tensor wrapper takes ownership of all the memory of the necessary metadata
 * for torch::executor::Tensor. Note that it doesn't own the data memory.
 */
class ManagedTensor {
 public:
  /// The type used for elements of `sizes()`.
  using SizesType = exec_aten::SizesType;
  /// The type used for elements of `dim_order()`.
  using DimOrderType = exec_aten::DimOrderType;
  /// The type used for elements of `strides()`.
  using StridesType = exec_aten::StridesType;
  ManagedTensor() = delete;

  explicit ManagedTensor(
      void* data,
      const std::vector<SizesType>& sizes,
      ScalarType dtype)
      : dtype_(dtype), sizes_(sizes), data_ptr_(data) {
    ssize_t dim = sizes.size();
    dim_order_.resize(dim);
    strides_.resize(dim);
    for (size_t i = 0; i < dim; ++i) {
      dim_order_[i] = i;
    }
    dim_order_to_stride_nocheck(
        sizes.data(), dim_order_.data(), dim, strides_.data());
    tensor_impl_ = std::make_unique<TensorImpl>(
        dtype_,
        dim,
        sizes_.data(),
        data_ptr_,
        dim_order_.data(),
        strides_.data(),
        TensorShapeDynamism::DYNAMIC_BOUND);
  }

  /**
   * Get the Tensor object managed by this class.
   */
  Tensor get_tensor() {
    return Tensor(tensor_impl_.get());
  }

 private:
  void* data_ptr_ = nullptr;
  std::unique_ptr<TensorImpl> tensor_impl_;
  std::vector<SizesType> sizes_;
  std::vector<StridesType> strides_;
  std::vector<DimOrderType> dim_order_;
  ScalarType dtype_;
};
} // namespace executor
} // namespace torch
