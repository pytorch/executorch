/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/named_data_map.h>

namespace executorch {
namespace runtime {
/**
 * A NamedDataMap implementation that wraps other NamedDataMaps.
 */
template <size_t N>
class MergedDataMap final
    : public executorch::ET_RUNTIME_NAMESPACE::NamedDataMap {
 public:
  /**
   * Creates a new NamedDataMap that takes in other data maps.
   *
   * @param[in] data_maps Array of NamedDataMap pointers to merge.
   * Note: the data maps must outlive the MergedDataMap instance.
   */
  static executorch::runtime::Result<MergedDataMap> load(
      const std::array<const NamedDataMap*, N>& data_maps) {
    std::array<const NamedDataMap*, N> valid_data_maps;
    size_t num_data_maps = 0;
    for (size_t i = 0; i < data_maps.size(); i++) {
      if (data_maps[i] != nullptr) {
        valid_data_maps[num_data_maps++] = data_maps[i];
      }
    }
    ET_CHECK_OR_RETURN_ERROR(
        num_data_maps > 0, InvalidArgument, "All provided data maps are null");

    // Check for duplicate keys.
    for (size_t i = 0; i < num_data_maps; i++) {
      for (size_t j = i + 1; j < num_data_maps; j++) {
        for (int k = 0; k < valid_data_maps[i]->get_num_keys().get(); k++) {
          const auto key = valid_data_maps[i]->get_key(k).get();
          ET_CHECK_OR_RETURN_ERROR(
              valid_data_maps[j]->get_tensor_layout(key).error() ==
                  executorch::runtime::Error::NotFound,
              InvalidArgument,
              "Duplicate key %s in data maps at index %zu and %zu",
              key,
              i,
              j);
        }
      }
    }
    return MergedDataMap<N>(std::move(valid_data_maps), num_data_maps);
  }

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
  get_tensor_layout(executorch::aten::string_view key) const override {
    for (size_t i = 0; i < num_data_maps_; i++) {
      auto layout = data_maps_[i]->get_tensor_layout(key);
      if (layout.ok()) {
        return layout.get();
      }
      if (layout.error() != executorch::runtime::Error::NotFound) {
        return layout.error();
      }
    }
    return executorch::runtime::Error::NotFound;
  }

  /**
   * Retrieve read-only data for the specified key.
   *
   * @param[in] key The name of the tensor to get data on.
   *
   * @return error if the key is not present or data cannot be loaded.
   */
  ET_NODISCARD
  executorch::runtime::Result<executorch::runtime::FreeableBuffer> get_data(
      executorch::aten::string_view key) const override {
    for (size_t i = 0; i < num_data_maps_; i++) {
      auto data = data_maps_[i]->get_data(key);
      if (data.error() != executorch::runtime::Error::NotFound) {
        return data;
      }
    }
    return executorch::runtime::Error::NotFound;
  }

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
      size_t size) const override {
    for (size_t i = 0; i < num_data_maps_; i++) {
      auto error = data_maps_[i]->load_data_into(key, buffer, size);
      if (error != executorch::runtime::Error::NotFound) {
        return error;
      }
    }
    return executorch::runtime::Error::NotFound;
  }

  /**
   * @returns The number of keys in the map.
   */
  ET_NODISCARD executorch::runtime::Result<uint32_t> get_num_keys()
      const override {
    uint32_t num_keys = 0;
    for (size_t i = 0; i < num_data_maps_; i++) {
      num_keys += data_maps_[i]->get_num_keys().get();
    }
    return num_keys;
  }

  /**
   * @returns The key at the specified index, error if index out of bounds.
   */
  ET_NODISCARD executorch::runtime::Result<const char*> get_key(
      uint32_t index) const override {
    uint32_t total_num_keys = get_num_keys().get();
    ET_CHECK_OR_RETURN_ERROR(
        index >= 0 && index < total_num_keys,
        InvalidArgument,
        "Index %u out of range of size %u",
        index,
        total_num_keys);
    for (size_t i = 0; i < num_data_maps_; i++) {
      auto num_keys = data_maps_[i]->get_num_keys().get();
      if (index < num_keys) {
        return data_maps_[i]->get_key(index);
      }
      index -= num_keys;
    }
    // Shouldn't reach here.
    return executorch::runtime::Error::Internal;
  }

  MergedDataMap(MergedDataMap&&) noexcept = default;

  ~MergedDataMap() override = default;

 private:
  MergedDataMap(
      const std::array<const NamedDataMap*, N>& data_maps,
      size_t num_data_maps)
      : data_maps_(data_maps), num_data_maps_(num_data_maps){};

  // Not copyable or assignable.
  MergedDataMap(const MergedDataMap& rhs) = delete;
  MergedDataMap& operator=(MergedDataMap&& rhs) noexcept = delete;
  MergedDataMap& operator=(const MergedDataMap& rhs) = delete;

  const std::array<const NamedDataMap*, N> data_maps_;
  const size_t num_data_maps_;
};

} // namespace runtime
} // namespace executorch
