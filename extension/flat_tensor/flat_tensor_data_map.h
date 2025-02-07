/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/named_data_map.h>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/tensor_layout.h>
#include <executorch/runtime/platform/compiler.h>

#include <utility>

// Forward declare flatbuffer types. This is a public header and must not
// include the generated flatbuffer header.
namespace flat_tensor_flatbuffer {
struct FlatTensor;
} // namespace flat_tensor_flatbuffer

namespace executorch {
namespace extension {

/**
 * A NamedDataMap implementation for FlatTensor-serialized data.
 */
class FlatTensorDataMap final : public executorch::runtime::NamedDataMap {
 public:
  /**
   * Creates a new DataMap that wraps FlatTensor data.
   *
   * @param[in] loader The DataLoader that wraps the FlatTensor file.
   * Note: the loader must outlive the FlatTensorDataMap instance.
   */
  static executorch::runtime::Result<FlatTensorDataMap> load(
      executorch::runtime::DataLoader* loader);

  ET_NODISCARD
  executorch::runtime::Result<const executorch::runtime::TensorLayout>
  get_metadata(const char* key) const override;
  ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> get_data(
      const char* key) const override;
  ET_NODISCARD executorch::runtime::Result<size_t>
  load_data_into(const char* key, void* buffer, size_t size) const override;

  ET_NODISCARD executorch::runtime::Result<size_t> get_num_keys()
      const override;
  ET_NODISCARD executorch::runtime::Result<const char*> get_key(
      size_t index) const override;

  FlatTensorDataMap(FlatTensorDataMap&&) noexcept = default;

  ~FlatTensorDataMap() override = default;

 private:
  FlatTensorDataMap(
      executorch::runtime::FreeableBuffer&& flat_tensor_data,
      const flat_tensor_flatbuffer::FlatTensor* flat_tensor,
      executorch::runtime::FreeableBuffer&& data_ro)
      : flat_tensor_data_(std::move(flat_tensor_data)),
        flat_tensor_(flat_tensor),
        data_ro_(std::move(data_ro)) {}

  // Not copyable or assignable.
  FlatTensorDataMap(const FlatTensorDataMap& rhs) = delete;
  FlatTensorDataMap& operator=(FlatTensorDataMap&& rhs) noexcept = delete;
  FlatTensorDataMap& operator=(const FlatTensorDataMap& rhs) = delete;

  // Serialized flat_tensor flatbuffer data.
  executorch::runtime::FreeableBuffer flat_tensor_data_;

  // Flatbuffer representation of the flat_tensor.
  const flat_tensor_flatbuffer::FlatTensor* flat_tensor_;

  // Loaded read-only tensor data.
  executorch::runtime::FreeableBuffer data_ro_;
};

} // namespace extension
} // namespace executorch
