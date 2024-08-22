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
      : sizes_(sizes) {
#ifdef USE_ATEN_LIB
    tensor_ = torch::from_blob(data, sizes, dtype);
#else
    // Calculate strides.
    strides_ = std::vector<StridesType>(sizes_.size());
    if (sizes_.size() > 0) {
      strides_.back() = 1;
      for (size_t i = strides_.size() - 1; i > 0; --i) {
        strides_[i - 1] = strides_[i] * sizes_[i];
      }
    }

    // Allocate TensorImpl.
    tensor_impl_ = std::make_unique<exec_aten::TensorImpl>(
        dtype,
        sizes_.size(),
        sizes_.data(),
        data,
        /*dim_order=*/nullptr,
        strides_.data(),
        executorch::runtime::TensorShapeDynamism::DYNAMIC_BOUND);
#endif
  }

  void resize(const std::vector<SizesType>& new_sizes) {
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
  std::unique_ptr<exec_aten::TensorImpl> tensor_impl_;
  std::vector<SizesType> sizes_;
  std::vector<StridesType> strides_;
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
