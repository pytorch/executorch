/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
// @nolint PATTERNLINT Ok to use stdlib for this optional library
#include <vector>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/assert.h>

#ifdef USE_ATEN_LIB
#include <torch/torch.h>
#endif

namespace executorch {
namespace extension {

/**
 * A tensor wrapper takes ownership of all the memory of the necessary metadata
 * for exec_aten::Tensor. Note that it doesn't own the data memory.
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
      exec_aten::ScalarType dtype)
      : dtype_(dtype), sizes_(sizes), data_ptr_(data) {
#ifdef USE_ATEN_LIB
    tensor_ = torch::from_blob(data, sizes, dtype_);
#else
    ssize_t dim = sizes.size();
    dim_order_.resize(dim);
    strides_.resize(dim);
    for (size_t i = 0; i < dim; ++i) {
      dim_order_[i] = i;
    }
    executorch::runtime::dim_order_to_stride_nocheck(
        sizes.data(), dim_order_.data(), dim, strides_.data());
    tensor_impl_ = std::make_unique<exec_aten::TensorImpl>(
        dtype_,
        dim,
        sizes_.data(),
        data_ptr_,
        dim_order_.data(),
        strides_.data(),
        executorch::runtime::TensorShapeDynamism::DYNAMIC_BOUND);
#endif
  }

  void resize(const std::vector<SizesType>& new_sizes) {
    ET_CHECK_MSG(
        new_sizes.size() == sizes_.size(),
        "Cannot change rank of a managed tensor");
    auto err = executorch::runtime::resize_tensor(
        this->get_aliasing_tensor(),
        exec_aten::ArrayRef<SizesType>(new_sizes.data(), new_sizes.size()));
    ET_CHECK(err == executorch::runtime::Error::Ok);
  }

  /**
   * Get the underlying Tensor object. This is assuming the copying is cheap.
   */
  exec_aten::Tensor get_aliasing_tensor() {
#ifdef USE_ATEN_LIB
    return tensor_;
#else
    return exec_aten::Tensor(tensor_impl_.get());
#endif
  }

 private:
  exec_aten::ScalarType dtype_;
  std::unique_ptr<exec_aten::TensorImpl> tensor_impl_;
  std::vector<SizesType> sizes_;
  std::vector<StridesType> strides_;
  std::vector<DimOrderType> dim_order_;
  void* data_ptr_ = nullptr;
#ifdef USE_ATEN_LIB
  exec_aten::Tensor tensor_;
#endif
};

} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::ManagedTensor;
} // namespace executor
} // namespace torch
