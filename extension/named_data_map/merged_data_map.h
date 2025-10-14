/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/named_data_map.h>

#include <unordered_map>
#include <vector>

#ifdef USE_ATEN_LIB
#define ET_MERGED_DATA_MAP_NAMESPACE merged_data_map::aten
#else // !USE_ATEN_LIB
#define ET_MERGED_DATA_MAP_NAMESPACE merged_data_map
#endif // USE_ATEN_LIB

namespace executorch::extension {

namespace ET_MERGED_DATA_MAP_NAMESPACE {
/**
 * A NamedDataMap implementation that wraps other NamedDataMaps.
 */
class MergedDataMap final
    : public executorch::ET_RUNTIME_NAMESPACE::NamedDataMap {
 public:
  /**
   * Creates a new NamedDataMap that takes in other data maps.
   *
   * @param[in] data_maps vector of NamedDataMap pointers to merge.
   * Note: the data maps must outlive the MergedDataMap instance.
   */
  static executorch::runtime::Result<MergedDataMap>
  load(executorch::runtime::Span<
       const executorch::ET_RUNTIME_NAMESPACE::NamedDataMap*> named_data_maps);

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

  MergedDataMap(MergedDataMap&&) noexcept = default;

  ~MergedDataMap() override = default;

 private:
  MergedDataMap(
      std::vector<const executorch::ET_RUNTIME_NAMESPACE::NamedDataMap*>
          named_data_maps,
      std::unordered_map<std::string, uint32_t> key_to_map_index)
      : named_data_maps_(std::move(named_data_maps)),
        key_to_map_index_(std::move(key_to_map_index)) {}

  // Not copyable or assignable.
  MergedDataMap(const MergedDataMap& rhs) = delete;
  MergedDataMap& operator=(MergedDataMap&& rhs) noexcept = delete;
  MergedDataMap& operator=(const MergedDataMap& rhs) = delete;

  std::vector<const executorch::ET_RUNTIME_NAMESPACE::NamedDataMap*>
      named_data_maps_;

  // Map from key to index in the named_data_maps_ vector.
  std::unordered_map<std::string, uint32_t> key_to_map_index_;
};

} // namespace ET_MERGED_DATA_MAP_NAMESPACE
} // namespace executorch::extension
