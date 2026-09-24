/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/pte_data_map.h>
#include <executorch/schema/extended_header.h>
#include <executorch/schema/program_generated.h>

#include <cstring>

#include <c10/util/safe_numerics.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {
namespace internal {
namespace {

Result<uint64_t> segment_alignment(uint64_t base, uint64_t offset) {
  uint64_t absolute_offset;
  ET_CHECK_OR_RETURN_ERROR(
      !c10::add_overflows(base, offset, &absolute_offset),
      InvalidProgram,
      "PTE segment offset overflows");
  return absolute_offset == 0
      ? 1
      : absolute_offset & (~absolute_offset + 1);
}

Result<uint64_t> aligned_segment_offset(
    uint64_t base,
    uint64_t next_offset,
    uint64_t alignment) {
  uint64_t absolute_offset;
  ET_CHECK_OR_RETURN_ERROR(
      !c10::add_overflows(base, next_offset, &absolute_offset),
      InvalidArgument,
      "Replacement segment offset overflows");
  const uint64_t mask = alignment - 1;
  ET_CHECK_OR_RETURN_ERROR(
      absolute_offset <= UINT64_MAX - mask,
      InvalidArgument,
      "Replacement segment alignment overflows");
  const uint64_t aligned_absolute_offset = (absolute_offset + mask) & ~mask;
  return aligned_absolute_offset - base;
}

} // namespace

/* static */ Result<PteDataMap> PteDataMap::create(
    DataLoader* loader,
    size_t segment_base_offset,
    const flatbuffers::FlatbufferNamedData* named_data,
    const flatbuffers::FlatbufferDataSegment* segments,
    Span<const uint8_t> program_data) {
  ET_CHECK_OR_RETURN_ERROR(
      loader != nullptr && named_data != nullptr && segments != nullptr,
      InvalidArgument,
      "PteDataMap loader, named_data or segments is null; most likely the program does not have any named_data segments");
  return PteDataMap(
      loader, segment_base_offset, named_data, segments, program_data);
}

ET_NODISCARD
Result<FreeableBuffer> PteDataMap::get_data(std::string_view key) const {
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
          DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Constant));
    }
  }
  return Error::NotFound;
}

Error PteDataMap::replace_data(
    Span<const Data> data,
    MemoryAllocator* temp_allocator) const {
  if (data.empty()) {
    return Error::Ok;
  }
  ET_CHECK_OR_RETURN_ERROR(
      loader_ != nullptr && segment_base_offset_ > 0 &&
          !program_data_.empty(),
      NotSupported,
      "Named data is not backed by replaceable PTE segments");
  ET_CHECK_OR_RETURN_ERROR(
      temp_allocator != nullptr,
      InvalidArgument,
      "Named data replacement requires a temporary allocator");

  auto* program_data = static_cast<uint8_t*>(temp_allocator->allocate(
      program_data_.size(), alignof(std::max_align_t)));
  ET_CHECK_OR_RETURN_ERROR(
      program_data != nullptr, MemoryAllocationFailed, "Could not copy PTE");
  std::memcpy(program_data, program_data_.data(), program_data_.size());
  auto* program = flatbuffers::GetMutableRoot<executorch_flatbuffer::Program>(
      program_data);
  auto* segments = program->mutable_segments();
  auto* named_data = program->mutable_named_data();
  ET_CHECK_OR_RETURN_ERROR(
      segments != nullptr && named_data != nullptr,
      InvalidProgram,
      "PTE has no named-data segments");

  int64_t* replacement_by_segment =
      temp_allocator->allocateList<int64_t>(segments->size());
  ET_CHECK_OR_RETURN_ERROR(
      replacement_by_segment != nullptr,
      MemoryAllocationFailed,
      "Could not allocate replacement index table");
  for (size_t i = 0; i < segments->size(); ++i) {
    replacement_by_segment[i] = -1;
  }
  for (size_t replacement_index = 0; replacement_index < data.size();
       ++replacement_index) {
    const auto& replacement = data[replacement_index];
    ET_CHECK_OR_RETURN_ERROR(
        replacement.bytes.data() != nullptr || replacement.bytes.empty(),
        InvalidArgument,
        "Replacement data for key %.*s is null",
        static_cast<int>(replacement.key.size()),
        replacement.key.data());
    if (replacement.alignment.has_value()) {
      const size_t alignment = *replacement.alignment;
      ET_CHECK_OR_RETURN_ERROR(
          alignment > 0 && (alignment & (alignment - 1)) == 0,
          InvalidArgument,
          "Replacement alignment must be a power of two");
    }

    size_t segment_index = segments->size();
    for (size_t i = 0; i < named_data->size(); ++i) {
      const auto* entry = named_data->Get(i);
      if (entry->key()->size() == replacement.key.size() &&
          (replacement.key.empty() ||
           std::memcmp(
               entry->key()->data(),
               replacement.key.data(),
               replacement.key.size()) == 0)) {
        segment_index = entry->segment_index();
        break;
      }
    }
    ET_CHECK_OR_RETURN_ERROR(
        segment_index < segments->size(),
        NotFound,
        "Named data key %.*s was not found",
        static_cast<int>(replacement.key.size()),
        replacement.key.data());
    const int64_t previous_index = replacement_by_segment[segment_index];
    if (previous_index >= 0) {
      const auto& previous = data[static_cast<size_t>(previous_index)];
      ET_CHECK_OR_RETURN_ERROR(
          previous.bytes.size() == replacement.bytes.size() &&
              previous.alignment == replacement.alignment &&
              (replacement.bytes.empty() ||
               std::memcmp(
                   previous.bytes.data(),
                   replacement.bytes.data(),
                   replacement.bytes.size()) == 0),
          InvalidArgument,
          "Aliased named data replacements do not match");
    } else {
      replacement_by_segment[segment_index] =
          static_cast<int64_t>(replacement_index);
    }
  }

  const size_t chunk_capacity = 1 + 2 * segments->size();
  auto* chunks =
      temp_allocator->allocateList<DataLoader::DataChunk>(chunk_capacity);
  ET_CHECK_OR_RETURN_ERROR(
      chunks != nullptr,
      MemoryAllocationFailed,
      "Could not allocate replacement chunks");
  size_t chunk_count = 0;
  chunks[chunk_count++] = DataLoader::DataChunk::from_buffer(
      program_data, program_data_.size());
  uint64_t next_offset = 0;
  for (size_t i = 0; i < segments->size(); ++i) {
    auto* segment = segments->GetMutableObject(i);
    const int64_t replacement_index = replacement_by_segment[i];
    const Data* replacement = replacement_index >= 0
        ? &data[static_cast<size_t>(replacement_index)]
        : nullptr;
    uint64_t alignment;
    if (replacement != nullptr && replacement->alignment.has_value()) {
      alignment = *replacement->alignment;
    } else {
      auto inferred_alignment =
          segment_alignment(segment_base_offset_, segment->offset());
      if (!inferred_alignment.ok()) {
        return inferred_alignment.error();
      }
      alignment = inferred_alignment.get();
    }
    auto aligned_offset =
        aligned_segment_offset(segment_base_offset_, next_offset, alignment);
    if (!aligned_offset.ok()) {
      return aligned_offset.error();
    }
    const uint64_t offset = aligned_offset.get();
    chunks[chunk_count++] =
        DataLoader::DataChunk::zero(offset - next_offset);
    const uint64_t size =
        replacement == nullptr ? segment->size() : replacement->bytes.size();
    if (replacement == nullptr) {
      chunks[chunk_count++] = DataLoader::DataChunk::from_loader(
          segment_base_offset_ + segment->offset(), segment->size());
    } else {
      chunks[chunk_count++] = DataLoader::DataChunk::from_buffer(
          replacement->bytes.data(), replacement->bytes.size());
    }
    ET_CHECK_OR_RETURN_ERROR(
        offset <= UINT64_MAX - size,
        InvalidArgument,
        "Replacement segment size overflows");
    next_offset = offset + size;
    ET_CHECK_OR_RETURN_ERROR(
        segment->mutate_offset(offset) && segment->mutate_size(size),
        InvalidProgram,
        "Could not update PTE segment metadata");
  }

  constexpr size_t kSegmentDataSizeOffset =
      ExtendedHeader::kHeaderOffset + 24;
  ET_CHECK_OR_RETURN_ERROR(
      program_data_.size() >= kSegmentDataSizeOffset + sizeof(uint64_t),
      InvalidProgram,
      "PTE extended header does not contain a segment data size");
  flatbuffers::WriteScalar<uint64_t>(
      program_data + kSegmentDataSizeOffset, next_offset);
  return loader_->replace_data(
      Span<const DataLoader::DataChunk>(chunks, chunk_count));
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
