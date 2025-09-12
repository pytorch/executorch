/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>

#include <executorch/extension/flat_tensor/serialize/flat_tensor_generated.h>
#include <executorch/extension/flat_tensor/serialize/flat_tensor_header.h>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/compiler.h>

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
    executorch::aten::string_view key,
    const flatbuffers::Vector<
        flatbuffers::Offset<flat_tensor_flatbuffer::NamedData>>* named_data,
    const flatbuffers::Vector<
        flatbuffers::Offset<flat_tensor_flatbuffer::DataSegment>>* segments,
    size_t segment_end_offset) {
  // Linear search by name.
  if (named_data == nullptr) {
    return Error::NotFound;
  }
  for (int i = 0; i < named_data->size(); i++) {
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
      ET_CHECK_OR_RETURN_ERROR(
          segments->Get(segment_index)->offset() < segment_end_offset,
          InvalidExternalData,
          "Invalid segment offset %" PRIu64
          " is larger than the segment_base_offset + segment_data_size %" PRIu64
          "; malformed PTD file.",
          segments->Get(segment_index)->offset(),
          static_cast<uint64_t>(segment_end_offset));
      return found;
    }
  }
  return Error::NotFound;
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
    executorch::aten::string_view key) const {
  Result<const flat_tensor_flatbuffer::NamedData*> named_data = get_named_data(
      key,
      flat_tensor_->named_data(),
      flat_tensor_->segments(),
      header_.segment_base_offset + header_.segment_data_size);
  if (!named_data.ok()) {
    return named_data.error();
  }
  return create_tensor_layout(named_data.get()->tensor_layout());
}

ET_NODISCARD Result<FreeableBuffer> FlatTensorDataMap::get_data(
    executorch::aten::string_view key) const {
  Result<const flat_tensor_flatbuffer::NamedData*> named_data = get_named_data(
      key,
      flat_tensor_->named_data(),
      flat_tensor_->segments(),
      header_.segment_base_offset + header_.segment_data_size);
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
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::External));
}

ET_NODISCARD Error FlatTensorDataMap::load_data_into(
    ET_UNUSED executorch::aten::string_view key,
    ET_UNUSED void* buffer,
    ET_UNUSED size_t size) const {
  Result<const flat_tensor_flatbuffer::NamedData*> named_data = get_named_data(
      key,
      flat_tensor_->named_data(),
      flat_tensor_->segments(),
      header_.segment_base_offset + header_.segment_data_size);
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

/* static */ Result<FlatTensorDataMap> FlatTensorDataMap::load(
    DataLoader* loader) {
  // Check header.
  Result<FreeableBuffer> header = loader->load(
      /*offset=*/0,
      FlatTensorHeader::kNumHeadBytes,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::External));
  if (!header.ok()) {
    ET_LOG(Error, "Failed to load header.");
    return header.error();
  }
  Result<FlatTensorHeader> fh =
      FlatTensorHeader::Parse(header->data(), header->size());
  if (fh.error() == Error::NotFound) {
    // No header, throw error.
    ET_LOG(Error, "No FlatTensorHeader found.");
    return fh.error();
  } else if (fh.error() != Error::Ok) {
    // corruption, throw error.
    ET_LOG(Error, "Flat tensor header may be corrupt.");
    return fh.error();
  }

  // Load flatbuffer data as a segment.
  Result<FreeableBuffer> flat_tensor_data = loader->load(
      /*offset=*/0,
      fh->flatbuffer_offset + fh->flatbuffer_size,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::External));
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

  // Get pointer to root of flatbuffer table.
  const flat_tensor_flatbuffer::FlatTensor* flat_tensor =
      flat_tensor_flatbuffer::GetFlatTensor(flat_tensor_data->data());

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
