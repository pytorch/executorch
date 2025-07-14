/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/named_data_map.h>

#include <executorch/extension/flat_tensor/serialize/flat_tensor_header.h>

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
class FlatTensorDataMap final
    : public executorch::ET_RUNTIME_NAMESPACE::NamedDataMap {
 public:
  /**
   * Creates a new DataMap that wraps FlatTensor data.
   *
   * @param[in] loader The DataLoader that wraps the FlatTensor file.
   * Note: the loader must outlive the FlatTensorDataMap instance.
   */
  static executorch::runtime::Result<FlatTensorDataMap> load(
      executorch::runtime::DataLoader* loader);

  /**
   * Retrieve the tensor_layout for the specified key.
   *
   * @param[in] key The name of the tensor to get metadata on.
   *
   * @return Error::NotFound if the key is not present.
   */
  ET_NODISCARD
  executorch::runtime::Result<
      const executorch::ET_RUNTIME_NAMESPACE::TensorLayout>
  get_tensor_layout(executorch::aten::string_view key) const override;

  /**
   * Retrieve read-only data for the specified key.
   *
   * @param[in] key The name of the tensor to get data on.
   *
   * @return error if the key is not present or data cannot be loaded.
   */
  ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> get_data(
      executorch::aten::string_view key) const override;

  /**
   * Loads the data of the specified tensor into the provided buffer.
   *
   * @param[in] key The name of the tensor to get the data of.
   * @param[in] buffer The buffer to load data into. Must point to at least
   * `size` bytes of memory.
   * @param[in] size The number of bytes to load.
   *
   * @returns an Error indicating if the load was successful.
   */
  ET_NODISCARD executorch::runtime::Error load_data_into(
      executorch::aten::string_view key,
      void* buffer,
      size_t size) const override;

  /**
   * @returns The number of keys in the map.
   */
  ET_NODISCARD executorch::runtime::Result<uint32_t> get_num_keys()
      const override;

  /**
   * @returns The key at the specified index, error if index out of bounds.
   */
  ET_NODISCARD executorch::runtime::Result<const char*> get_key(
      uint32_t index) const override;

  FlatTensorDataMap(FlatTensorDataMap&&) noexcept = default;

  ~FlatTensorDataMap() override = default;

 private:
  FlatTensorDataMap(
      const FlatTensorHeader& header,
      executorch::runtime::FreeableBuffer&& flat_tensor_data,
      const flat_tensor_flatbuffer::FlatTensor* flat_tensor,
      executorch::runtime::DataLoader* loader)
      : header_(header),
        flat_tensor_data_(std::move(flat_tensor_data)),
        flat_tensor_(flat_tensor),
        loader_(loader) {}

  // Not copyable or assignable.
  FlatTensorDataMap(const FlatTensorDataMap& rhs) = delete;
  FlatTensorDataMap& operator=(FlatTensorDataMap&& rhs) noexcept = delete;
  FlatTensorDataMap& operator=(const FlatTensorDataMap& rhs) = delete;

  // FlatTensor header, containing segment_base_offset and segment_data_size.
  const FlatTensorHeader header_;

  // Serialized flat_tensor flatbuffer data.
  executorch::runtime::FreeableBuffer flat_tensor_data_;

  // Flatbuffer representation of the flat_tensor.
  const flat_tensor_flatbuffer::FlatTensor* flat_tensor_;

  // Data loader, used to load segment data.
  executorch::runtime::DataLoader* loader_;
};

} // namespace extension
} // namespace executorch
