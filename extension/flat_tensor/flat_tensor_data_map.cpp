/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>

#include <c10/util/safe_numerics.h>

#include <executorch/extension/flat_tensor/serialize/flat_tensor_generated.h>
#include <executorch/extension/flat_tensor/serialize/flat_tensor_header.h>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/compiler.h>

#include <cinttypes>
#include <cstring>
#include <vector>

using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::Span;

using executorch::aten::ScalarType;
using executorch::ET_RUNTIME_NAMESPACE::TensorLayout;
using executorch::runtime::DataLoader;

namespace executorch {
namespace extension {

namespace {
/**
 * FlatTensor data must be aligned to this value to properly parse it. Must be a
 * power of 2. Note that max_align_t is the alignment that malloc() and new
 * guarantee.
 */
constexpr size_t kMinimumAlignment = alignof(std::max_align_t);

bool is_aligned(const void* data) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(data);
  return addr % kMinimumAlignment == 0;
}

Result<const flat_tensor_flatbuffer::NamedData*> get_named_data(
    std::string_view key,
    const flatbuffers::Vector<
        flatbuffers::Offset<flat_tensor_flatbuffer::NamedData>>* named_data,
    const flatbuffers::Vector<
        flatbuffers::Offset<flat_tensor_flatbuffer::DataSegment>>* segments,
    uint64_t segment_end_offset) {
  // Linear search by name.
  if (named_data == nullptr) {
    return Error::NotFound;
  }
  for (flatbuffers::uoffset_t i = 0; i < named_data->size(); ++i) {
    if (key.size() == named_data->Get(i)->key()->size() &&
        std::strncmp(
            named_data->Get(i)->key()->c_str(),
            key.data(),
            named_data->Get(i)->key()->size()) == 0) {
      const auto* found = named_data->Get(i);
      // Validate the named_data.
      size_t segment_index = found->segment_index();
      ET_CHECK_OR_RETURN_ERROR(
          segment_index >= 0 && segment_index < segments->size(),
          InvalidExternalData,
          "Segment index %zu for key %.*s is out of bounds for segment size %d. Malformed PTD file.",
          segment_index,
          static_cast<int>(key.size()),
          key.data(),
          segments->size());
      // Validate the segment.
      uint64_t seg_end = 0;
      ET_CHECK_OR_RETURN_ERROR(
          !c10::add_overflows(
              static_cast<uint64_t>(segments->Get(segment_index)->offset()),
              static_cast<uint64_t>(segments->Get(segment_index)->size()),
              &seg_end) &&
              seg_end <= segment_end_offset,
          InvalidExternalData,
          "Invalid segment offset %" PRIu64
          " is larger than the segment_base_offset + segment_data_size %" PRIu64
          "; malformed PTD file.",
          segments->Get(segment_index)->offset(),
          segment_end_offset);
      return found;
    }
  }
  return Error::NotFound;
}

Result<uint64_t> get_segment_end_offset(const FlatTensorHeader& header) {
  uint64_t segment_end_offset = 0;
  ET_CHECK_OR_RETURN_ERROR(
      !c10::add_overflows(
          header.segment_base_offset,
          header.segment_data_size,
          &segment_end_offset),
      InvalidExternalData,
      "segment_base_offset %" PRIu64 " + segment_data_size %" PRIu64
      " overflows uint64_t; malformed PTD file.",
      header.segment_base_offset,
      header.segment_data_size);
  return segment_end_offset;
}

Result<uint64_t> segment_alignment(uint64_t base, uint64_t offset) {
  uint64_t absolute_offset;
  ET_CHECK_OR_RETURN_ERROR(
      !c10::add_overflows(base, offset, &absolute_offset),
      InvalidExternalData,
      "FlatTensor segment offset overflows");
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

Result<const TensorLayout> create_tensor_layout(
    const flat_tensor_flatbuffer::TensorLayout* tensor_layout) {
  ScalarType scalar_type =
      static_cast<ScalarType>(tensor_layout->scalar_type());
  const int dim = tensor_layout->sizes()->size();
  const auto serialized_sizes = tensor_layout->sizes()->data();
  const auto serialized_dim_order = tensor_layout->dim_order()->data();
  return TensorLayout::create(
      Span<const int32_t>(serialized_sizes, dim),
      Span<const uint8_t>(serialized_dim_order, dim),
      scalar_type);
}

} // namespace

ET_NODISCARD Result<const TensorLayout> FlatTensorDataMap::get_tensor_layout(
    std::string_view key) const {
  Result<uint64_t> segment_end_offset = get_segment_end_offset(header_);
  if (!segment_end_offset.ok()) {
    return segment_end_offset.error();
  }
  Result<const flat_tensor_flatbuffer::NamedData*> named_data = get_named_data(
      key,
      flat_tensor_->named_data(),
      flat_tensor_->segments(),
      segment_end_offset.get());
  if (!named_data.ok()) {
    return named_data.error();
  }
  return create_tensor_layout(named_data.get()->tensor_layout());
}

ET_NODISCARD Result<FreeableBuffer> FlatTensorDataMap::get_data(
    std::string_view key) const {
  Result<uint64_t> segment_end_offset = get_segment_end_offset(header_);
  if (!segment_end_offset.ok()) {
    return segment_end_offset.error();
  }
  Result<const flat_tensor_flatbuffer::NamedData*> named_data = get_named_data(
      key,
      flat_tensor_->named_data(),
      flat_tensor_->segments(),
      segment_end_offset.get());
  if (!named_data.ok()) {
    return named_data.error();
  }

  uint32_t segment_index = named_data.get()->segment_index();
  uint64_t segment_offset =
      flat_tensor_->segments()->Get(segment_index)->offset();
  uint64_t segment_size = flat_tensor_->segments()->Get(segment_index)->size();

  return loader_->load(
      /*offset=*/header_.segment_base_offset + segment_offset,
      segment_size,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Constant));
}

ET_NODISCARD Error FlatTensorDataMap::load_data_into(
    ET_UNUSED std::string_view key,
    ET_UNUSED void* buffer,
    ET_UNUSED size_t size) const {
  Result<uint64_t> segment_end_offset = get_segment_end_offset(header_);
  if (!segment_end_offset.ok()) {
    return segment_end_offset.error();
  }
  Result<const flat_tensor_flatbuffer::NamedData*> named_data = get_named_data(
      key,
      flat_tensor_->named_data(),
      flat_tensor_->segments(),
      segment_end_offset.get());
  if (!named_data.ok()) {
    return named_data.error();
  }

  uint32_t segment_index = named_data.get()->segment_index();
  uint64_t segment_offset =
      flat_tensor_->segments()->Get(segment_index)->offset();

  Result<const TensorLayout> tensor_layout =
      create_tensor_layout(named_data.get()->tensor_layout());

  if (!tensor_layout.ok()) {
    return tensor_layout.error();
  }

  ET_CHECK_OR_RETURN_ERROR(
      size <= tensor_layout.get().nbytes(),
      InvalidArgument,
      "Buffer size %zu is smaller than tensor size %zu",
      size,
      tensor_layout.get().nbytes());

  // Load mutable data.
  DataLoader::SegmentInfo info = DataLoader::SegmentInfo(
      DataLoader::SegmentInfo::Type::Mutable, 0, nullptr);
  return loader_->load_into(
      header_.segment_base_offset + segment_offset,
      tensor_layout.get().nbytes(),
      info,
      buffer);
}

ET_NODISCARD Result<uint32_t> FlatTensorDataMap::get_num_keys() const {
  return flat_tensor_->named_data()->size();
}

ET_NODISCARD Result<const char*> FlatTensorDataMap::get_key(
    uint32_t index) const {
  uint32_t num_keys = get_num_keys().get();
  ET_CHECK_OR_RETURN_ERROR(
      index >= 0 && index < num_keys,
      InvalidArgument,
      "Index %u out of range of size %u",
      index,
      num_keys);
  return flat_tensor_->named_data()->Get(index)->key()->c_str();
}

ET_NODISCARD Error FlatTensorDataMap::replace_data(
    Span<const Data> data,
    ET_UNUSED executorch::runtime::MemoryAllocator* temp_allocator) const {
  if (data.empty()) {
    return Error::Ok;
  }

  auto prefix = loader_->load(
      0,
      header_.segment_base_offset,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  if (!prefix.ok()) {
    return prefix.error();
  }
  std::vector<uint8_t> flat_tensor_data(header_.segment_base_offset);
  std::memcpy(
      flat_tensor_data.data(), prefix->data(), header_.segment_base_offset);
  auto* flat_tensor =
      flatbuffers::GetMutableRoot<flat_tensor_flatbuffer::FlatTensor>(
          flat_tensor_data.data());
  auto* segments = flat_tensor->mutable_segments();
  auto* named_data = flat_tensor->mutable_named_data();
  ET_CHECK_OR_RETURN_ERROR(
      segments != nullptr && named_data != nullptr,
      InvalidExternalData,
      "FlatTensor has no named-data segments");

  std::vector<int64_t> replacement_by_segment(segments->size(), -1);
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

  std::vector<DataLoader::DataChunk> chunks;
  chunks.reserve(1 + 2 * segments->size());
  chunks.push_back(DataLoader::DataChunk::from_buffer(
      flat_tensor_data.data(), flat_tensor_data.size()));
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
          segment_alignment(header_.segment_base_offset, segment->offset());
      if (!inferred_alignment.ok()) {
        return inferred_alignment.error();
      }
      alignment = inferred_alignment.get();
    }
    auto aligned_offset = aligned_segment_offset(
        header_.segment_base_offset, next_offset, alignment);
    if (!aligned_offset.ok()) {
      return aligned_offset.error();
    }
    const uint64_t offset = aligned_offset.get();
    chunks.push_back(DataLoader::DataChunk::zero(offset - next_offset));
    const uint64_t size =
        replacement == nullptr ? segment->size() : replacement->bytes.size();
    if (replacement == nullptr) {
      chunks.push_back(DataLoader::DataChunk::from_loader(
          header_.segment_base_offset + segment->offset(), segment->size()));
    } else {
      chunks.push_back(DataLoader::DataChunk::from_buffer(
          replacement->bytes.data(), replacement->bytes.size()));
    }
    ET_CHECK_OR_RETURN_ERROR(
        offset <= UINT64_MAX - size,
        InvalidArgument,
        "Replacement segment size overflows");
    next_offset = offset + size;
    ET_CHECK_OR_RETURN_ERROR(
        segment->mutate_offset(offset) && segment->mutate_size(size),
        InvalidExternalData,
        "Could not update FlatTensor segment metadata");
  }

  constexpr size_t kSegmentDataSizeOffset =
      FlatTensorHeader::kHeaderOffset + 32;
  ET_CHECK_OR_RETURN_ERROR(
      flat_tensor_data.size() >=
          kSegmentDataSizeOffset + sizeof(uint64_t),
      InvalidExternalData,
      "FlatTensor header does not contain a segment data size");
  flatbuffers::WriteScalar<uint64_t>(
      flat_tensor_data.data() + kSegmentDataSizeOffset, next_offset);
  return loader_->replace_data(
      Span<const DataLoader::DataChunk>(chunks.data(), chunks.size()));
}

/* static */ Result<FlatTensorDataMap> FlatTensorDataMap::load(
    DataLoader* loader) {
  // Check header.
  Result<FreeableBuffer> header = loader->load(
      /*offset=*/0,
      FlatTensorHeader::kNumHeadBytes,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  if (!header.ok()) {
    ET_LOG(Error, "Failed to load header.");
    return header.error();
  }
  Result<FlatTensorHeader> fh =
      FlatTensorHeader::Parse(header->data(), header->size());

  ET_CHECK_OR_RETURN_ERROR(
      fh.ok(),
      InvalidExternalData,
      "Failed to parse FlatTensor header with error code %u. File may be corrupt.",
      static_cast<uint32_t>(fh.error()));

  size_t expected_size = fh->segment_base_offset + fh->segment_data_size;
  size_t actual_size = loader->size().get();
  ET_CHECK_OR_RETURN_ERROR(
      expected_size <= actual_size,
      InvalidExternalData,
      "File size is too small; file may be corrupted or truncated. Expected %zu from flat_tensor header, received %zu from data loader",
      expected_size,
      actual_size);

  // Load flatbuffer data as a segment.
  Result<FreeableBuffer> flat_tensor_data = loader->load(
      /*offset=*/0,
      fh->flatbuffer_offset + fh->flatbuffer_size,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  if (!flat_tensor_data.ok()) {
    ET_LOG(Error, "Failed to load flat_tensor data.");
    return flat_tensor_data.error();
  }

  // Make sure magic matches.
  if (!flat_tensor_flatbuffer::FlatTensorBufferHasIdentifier(
          flat_tensor_data->data())) {
    ET_LOG(
        Error,
        "FlatTensor identifier '%.4s' != expected '%.4s'",
        flatbuffers::GetBufferIdentifier(flat_tensor_data->data()),
        flat_tensor_flatbuffer::FlatTensorIdentifier());
    return Error::InvalidExternalData;
  }

  // The flatbuffer data must start at an aligned address to ensure internal
  // alignment of flatbuffer fields.
  ET_CHECK_OR_RETURN_ERROR(
      is_aligned(flat_tensor_data->data()),
      InvalidArgument,
      "FlatTensor data 0x%p must be aligned to %zu",
      flat_tensor_data->data(),
      kMinimumAlignment);

  // Verify that the root table offset is within bounds before dereferencing it.
  // GetFlatTensor() below reads the root offset from the first bytes of the
  // buffer and returns a pointer at buf + root_offset; reading any field then
  // walks that table's vtable. Nothing here runs full flatbuffer verification,
  // so a corrupt offset would otherwise let version()/named_data()/segments()
  // dereference memory outside the buffer.
  //
  // Minimum size: root offset + file identifier, i.e. the flatbuffer header
  // before the FlatTensor header begins. The identifier check above already
  // implies at least this many bytes, but check explicitly so the bound below
  // cannot underflow.
  constexpr size_t kMinBufferSize = FlatTensorHeader::kHeaderOffset;
  ET_CHECK_OR_RETURN_ERROR(
      flat_tensor_data->size() >= kMinBufferSize,
      InvalidExternalData,
      "FlatTensor data size %zu is too small (minimum %zu)",
      flat_tensor_data->size(),
      kMinBufferSize);
  uint32_t root_offset =
      flatbuffers::ReadScalar<flatbuffers::uoffset_t>(flat_tensor_data->data());
  // The root table is at buf + root_offset. It must not point into the header
  // and must leave room for at least a vtable offset (soffset_t) at its
  // position.
  ET_CHECK_OR_RETURN_ERROR(
      root_offset >= kMinBufferSize &&
          root_offset <=
              flat_tensor_data->size() - sizeof(flatbuffers::soffset_t),
      InvalidExternalData,
      "FlatTensor root table offset %u is invalid for data size %zu",
      root_offset,
      flat_tensor_data->size());

  // Get pointer to root of flatbuffer table.
  const flat_tensor_flatbuffer::FlatTensor* flat_tensor =
      flat_tensor_flatbuffer::GetFlatTensor(flat_tensor_data->data());

  // The file identifier above ("FT01") is bumped by convention only on a
  // backward-incompatible schema change, so it selects a schema family. The
  // version is the finer gate within that family: a file written by a newer
  // exporter is refused here instead of being misread field by field. Older
  // files stay loadable because the schema only grows by appending optional
  // fields.
  ET_CHECK_OR_RETURN_ERROR(
      flat_tensor->version() <= kMaxSupportedSchemaVersion,
      InvalidExternalData,
      "FlatTensor schema version %u is newer than the highest this runtime "
      "supports (%u). Export the data with an older ExecuTorch, or update the "
      "runtime.",
      flat_tensor->version(),
      kMaxSupportedSchemaVersion);

  // Validate flat_tensor.
  ET_CHECK_OR_RETURN_ERROR(
      flat_tensor->named_data() != nullptr,
      InvalidExternalData,
      "FlatTensor named_data is nullptr, malformed PTD file.");

  ET_CHECK_OR_RETURN_ERROR(
      flat_tensor->segments() != nullptr,
      InvalidExternalData,
      "FlatTensor segments is nullptr, malformed PTD file.");

  return FlatTensorDataMap(
      fh.get(), std::move(flat_tensor_data.get()), flat_tensor, loader);
}

} // namespace extension
} // namespace executorch
