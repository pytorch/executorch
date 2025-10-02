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
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace executorch {
namespace extension {
namespace llm {

class ET_EXPERIMENTAL Image {
 public:
  // Default constructor
  Image() : width_(0), height_(0), channels_(0) {}

  // Constructor for uint8_t data
  Image(
      std::vector<uint8_t>&& data,
      int32_t width,
      int32_t height,
      int32_t channels)
      : data_(std::move(data)),
        width_(width),
        height_(height),
        channels_(channels) {}

  // Constructor for float data
  Image(
      std::vector<float>&& data,
      int32_t width,
      int32_t height,
      int32_t channels)
      : data_(std::move(data)),
        width_(width),
        height_(height),
        channels_(channels) {}

  // Getters
  int32_t width() const {
    return width_;
  }
  int32_t height() const {
    return height_;
  }
  int32_t channels() const {
    return channels_;
  }

  // Data access
  bool is_uint8() const {
    return std::holds_alternative<std::vector<uint8_t>>(data_);
  }

  bool is_float() const {
    return std::holds_alternative<std::vector<float>>(data_);
  }

  const std::vector<uint8_t>& get_uint8_data() const& {
    return std::get<std::vector<uint8_t>>(data_);
  }

  std::vector<uint8_t>& get_uint8_data() & {
    return std::get<std::vector<uint8_t>>(data_);
  }

  const std::vector<float>& get_float_data() const& {
    return std::get<std::vector<float>>(data_);
  }

  std::vector<float>& get_float_data() & {
    return std::get<std::vector<float>>(data_);
  }

  executorch::runtime::Result<executorch::extension::TensorPtr> toTensor(
      bool with_batch = false) const {
    // Note: This creates a 3D tensor (CHW). The model might expect a 4D
    // tensor (NCHW). The caller should handle reshaping if needed.
    std::vector<executorch::aten::SizesType> sizes = {
        channels(), height(), width()};
    if (with_batch) {
      sizes.insert(sizes.begin(), 1);
    }
    if (is_float()) {
      return executorch::extension::from_blob(
          const_cast<float*>(get_float_data().data()),
          sizes,
          ::executorch::aten::ScalarType::Float);
    } else if (is_uint8()) {
      return executorch::extension::from_blob(
          const_cast<uint8_t*>(get_uint8_data().data()),
          sizes,
          ::executorch::aten::ScalarType::Byte);
    }
    ET_LOG(
        Error, "Image data is not initialized with uint8_t or float vector.");
    return ::executorch::runtime::Error::NotSupported;
  }

 private:
  // Assuming NCHW format
  std::variant<std::vector<uint8_t>, std::vector<float>> data_;
  int32_t width_;
  int32_t height_;
  int32_t channels_;
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
