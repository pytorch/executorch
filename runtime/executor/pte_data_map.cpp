/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/pte_data_map.h>
#include <executorch/schema/program_generated.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {
namespace internal {

/* static */ Result<PteDataMap> PteDataMap::create(
    DataLoader* loader,
    size_t segment_base_offset,
    const flatbuffers::FlatbufferNamedData* named_data,
    const flatbuffers::FlatbufferDataSegment* segments) {
  ET_CHECK_OR_RETURN_ERROR(
      loader != nullptr && named_data != nullptr && segments != nullptr,
      InvalidArgument,
      "PteDataMap loader, named_data or segments is null; most likely the program does not have any named_data segments");
  return PteDataMap(loader, segment_base_offset, named_data, segments);
}

ET_NODISCARD
Result<FreeableBuffer> PteDataMap::get_data(
    executorch::aten::string_view key) const {
  for (uint32_t i = 0; i < named_data_->size(); i++) {
    const auto* named_data_item = named_data_->Get(i);
    ET_CHECK_OR_RETURN_ERROR(
        named_data_item != nullptr && named_data_item->key() != nullptr,
        InvalidArgument,
        "Searching for key %.*s: NamedData at index %d is null",
        static_cast<int>(key.size()),
        key.data(),
        i);
    const auto* named_data_key = named_data_item->key();
    if (named_data_key->size() == key.size() &&
        memcmp(named_data_key->data(), key.data(), key.size()) == 0) {
      // Get the segment index.
      size_t segment_index = named_data_item->segment_index();

      // Get the segment offset and size.
      ET_CHECK_OR_RETURN_ERROR(
          segment_index < segments_->size(),
          InvalidArgument,
          "Segment index %zu for key %.*s is out of range for segments size %u",
          segment_index,
          static_cast<int>(key.size()),
          key.data(),
          segments_->size());
      size_t segment_offset = segments_->Get(segment_index)->offset();
      size_t segment_size = segments_->Get(segment_index)->size();
      return loader_->load(
          /*offset=*/segment_base_offset_ + segment_offset,
          segment_size,
          DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::External));
    }
  }
  return Error::NotFound;
}

ET_NODISCARD Result<uint32_t> PteDataMap::get_num_keys() const {
  return named_data_->size();
}

ET_NODISCARD Result<const char*> PteDataMap::get_key(uint32_t index) const {
  ET_CHECK_OR_RETURN_ERROR(
      index < named_data_->size(),
      InvalidArgument,
      "Index out of range: named_data size is %u, received index %u",
      named_data_->size(),
      index);

  const auto* item = named_data_->Get(index);
  ET_CHECK_OR_RETURN_ERROR(
      item != nullptr && item->key() != nullptr,
      InvalidArgument,
      "NamedData at index %u is null",
      index);
  return item->key()->c_str();
}

} // namespace internal
} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
