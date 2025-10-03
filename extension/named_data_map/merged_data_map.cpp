/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/named_data_map/merged_data_map.h>

#include <executorch/runtime/core/data_loader.h>

#include <vector>

using executorch::aten::string_view;
using executorch::ET_RUNTIME_NAMESPACE::NamedDataMap;
using executorch::ET_RUNTIME_NAMESPACE::TensorLayout;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

namespace executorch {
namespace extension {

/*static*/ Result<MergedDataMap> MergedDataMap::load(
    std::vector<const NamedDataMap*> named_data_maps) {
  std::vector<const NamedDataMap*> valid_data_maps;
  for (size_t i = 0; i < named_data_maps.size(); i++) {
    if (named_data_maps[i] != nullptr &&
        named_data_maps[i]->get_num_keys().get() > 0) {
      valid_data_maps.push_back(named_data_maps[i]);
    }
  }
  ET_CHECK_OR_RETURN_ERROR(
      valid_data_maps.size() > 0,
      InvalidArgument,
      "No non-empty named data maps provided to merge");

  // Check for duplicate keys.
  std::unordered_map<std::string, uint32_t> key_to_map_index;
  for (uint32_t i = 0; i < valid_data_maps.size(); i++) {
    const auto cur_map = valid_data_maps[i];
    uint32_t num_keys = cur_map->get_num_keys().get();
    for (uint32_t j = 0; j < num_keys; ++j) {
      const auto cur_key = cur_map->get_key(j).get();
      ET_CHECK_OR_RETURN_ERROR(
          key_to_map_index.find(cur_key) == key_to_map_index.end(),
          InvalidArgument,
          "Duplicate key %s in named data maps at index %u and %u",
          cur_key,
          key_to_map_index.at(cur_key),
          i);
      key_to_map_index[cur_key] = i;
    }
  }
  return MergedDataMap(std::move(valid_data_maps), std::move(key_to_map_index));
}

ET_NODISCARD Result<const TensorLayout> MergedDataMap::get_tensor_layout(
    string_view key) const {
  ET_CHECK_OR_RETURN_ERROR(
      key_to_map_index_.find(key.data()) != key_to_map_index_.end(),
      NotFound,
      "Key %s not found in named data maps",
      key.data());

  return named_data_maps_.at(key_to_map_index_.at(key.data()))
      ->get_tensor_layout(key);
}

ET_NODISCARD
Result<FreeableBuffer> MergedDataMap::get_data(string_view key) const {
  ET_CHECK_OR_RETURN_ERROR(
      key_to_map_index_.find(key.data()) != key_to_map_index_.end(),
      NotFound,
      "Key %s not found in named data maps",
      key.data());
  return named_data_maps_.at(key_to_map_index_.at(key.data()))->get_data(key);
}

ET_NODISCARD Error MergedDataMap::load_data_into(
    string_view key,
    void* buffer,
    size_t size) const {
  ET_CHECK_OR_RETURN_ERROR(
      key_to_map_index_.find(key.data()) != key_to_map_index_.end(),
      NotFound,
      "Key %s not found in named data maps",
      key.data());
  return named_data_maps_.at(key_to_map_index_.at(key.data()))
      ->load_data_into(key, buffer, size);
}

ET_NODISCARD Result<uint32_t> MergedDataMap::get_num_keys() const {
  return key_to_map_index_.size();
}

ET_NODISCARD Result<const char*> MergedDataMap::get_key(uint32_t index) const {
  uint32_t total_num_keys = get_num_keys().get();
  ET_CHECK_OR_RETURN_ERROR(
      index >= 0 && index < total_num_keys,
      InvalidArgument,
      "Index %u out of range of size %u",
      index,
      total_num_keys);
  for (size_t i = 0; i < named_data_maps_.size(); i++) {
    auto num_keys = named_data_maps_[i]->get_num_keys().get();
    if (index < num_keys) {
      return named_data_maps_[i]->get_key(index);
    }
    index -= num_keys;
  }
  // Shouldn't reach here.
  return Error::Internal;
}
} // namespace extension
} // namespace executorch
