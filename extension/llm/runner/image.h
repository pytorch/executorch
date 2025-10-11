/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple image struct.

#pragma once
#include <executorch/runtime/platform/compiler.h>
#include <cstdint>
#include <vector>

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace executorch {
namespace extension {
namespace llm {

// Assuming NCHW format
class ET_EXPERIMENTAL Image {
 public:
  // Constructor for uint8_t data
  Image(
      std::vector<uint8_t>&& data,
      int32_t width,
      int32_t height,
      int32_t channels)
      : Image(make_tensor_ptr(
            {channels, height, width},
            std::move(data),
            executorch::aten::ScalarType::Byte)) {}

  // Constructor for float data
  Image(
      std::vector<float>&& data,
      int32_t width,
      int32_t height,
      int32_t channels)
      : Image(make_tensor_ptr({channels, height, width}, std::move(data))) {}

  Image(executorch::extension::TensorPtr tensor) : tensor_(std::move(tensor)) {
    ET_CHECK_MSG(tensor_, "Null tensor");
    ET_CHECK_MSG(tensor_->dim() == 3, "Invalid tensor rank");
  }

  // Getters
  int32_t channels() const {
    return tensor_->size(0);
  }

  int32_t height() const {
    return tensor_->size(1);
  }

  int32_t width() const {
    return tensor_->size(2);
  }

  // Data access
  bool is_uint8() const {
    return tensor_->scalar_type() == ::executorch::aten::ScalarType::Byte;
  }

  bool is_float() const {
    return tensor_->scalar_type() == ::executorch::aten::ScalarType::Float;
  }

  std::vector<uint8_t> copy_uint8_data() const {
    ET_DCHECK_MSG(is_uint8(), "Image dtype is not uint8");
    auto data = tensor_->const_data_ptr<uint8_t>();
    return std::vector<uint8_t>(data, data + tensor_->numel());
  }

  std::vector<uint8_t> copy_uint8_data() {
    ET_DCHECK_MSG(is_uint8(), "Image dtype is not uint8");
    auto data = tensor_->const_data_ptr<uint8_t>();
    return std::vector<uint8_t>(data, data + tensor_->numel());
  }

  std::vector<float> copy_float_data() const {
    ET_DCHECK_MSG(is_float(), "Image dtype is not float");
    auto data = tensor_->const_data_ptr<float>();
    return std::vector<float>(data, data + tensor_->numel());
  }

  std::vector<float> copy_float_data() {
    ET_DCHECK_MSG(is_float(), "Image dtype is not float");
    auto data = tensor_->const_data_ptr<float>();
    return std::vector<float>(data, data + tensor_->numel());
  }

  executorch::runtime::Result<executorch::extension::TensorPtr> toTensor(
      bool with_batch = false) const {
    if (with_batch) {
      return make_tensor_ptr(
          *tensor_,
          {1, 
           executorch::aten::SizesType(tensor_->size(0)),
           executorch::aten::SizesType(tensor_->size(1)),
           executorch::aten::SizesType(tensor_->size(2))});
    }
    return tensor_;
  }

 private:
  executorch::extension::TensorPtr tensor_;
};

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::Image;
} // namespace executor
} // namespace torch
