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
namespace runtime {
namespace internal {

/* static */ executorch::runtime::Result<PteDataMap> PteDataMap::create(
    executorch::runtime::DataLoader* loader,
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
executorch::runtime::Result<executorch::runtime::FreeableBuffer>
PteDataMap::get_data(const char* key) const {
  for (size_t i = 0; i < named_data_->size(); i++) {
    ET_CHECK_OR_RETURN_ERROR(
        named_data_->Get(i) != nullptr && named_data_->Get(i)->key() != nullptr,
        InvalidArgument,
        "Searching for key %s: NamedData at index %zu is null",
        key,
        i);
    if (strcmp(named_data_->Get(i)->key()->c_str(), key) == 0) {
      // Get the segment index.
      size_t segment_index = named_data_->Get(i)->segment_index();

      // Get the segment offset and size.
      ET_CHECK_OR_RETURN_ERROR(
          segment_index < segments_->size(),
          InvalidArgument,
          "Segment index %zu for key %s is out of range for segments size %u",
          segment_index,
          key,
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

ET_NODISCARD executorch::runtime::Result<size_t> PteDataMap::get_num_keys()
    const {
  return named_data_->size();
}

ET_NODISCARD executorch::runtime::Result<const char*> PteDataMap::get_key(
    size_t index) const {
  ET_CHECK_OR_RETURN_ERROR(
      index < named_data_->size(),
      InvalidArgument,
      "Index out of range: named_data size is %u, received index %zu",
      named_data_->size(),
      index);

  ET_CHECK_OR_RETURN_ERROR(
      named_data_->Get(index) != nullptr &&
          named_data_->Get(index)->key() != nullptr,
      InvalidArgument,
      "NamedData at index %zu is null",
      index);
  return named_data_->Get(index)->key()->c_str();
}

} // namespace internal
} // namespace runtime
} // namespace executorch
