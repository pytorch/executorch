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
using executorch::runtime::DataLoader;
using executorch::runtime::TensorLayout;

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

Result<const flat_tensor_flatbuffer::TensorMetadata*> get_flat_tensor_metadata(
    const char* key,
    const flatbuffers::Vector<
        flatbuffers::Offset<flat_tensor_flatbuffer::TensorMetadata>>* tensors) {
  // Linear search by name.
  for (int i = 0; i < tensors->size(); i++) {
    if (std::strcmp(tensors->Get(i)->fully_qualified_name()->c_str(), key) ==
        0) {
      const auto* metadata = tensors->Get(i);
      ET_CHECK_OR_RETURN_ERROR(
          metadata->segment_index() >= 0 && metadata->offset() >= 0,
          InvalidExternalData,
          "Invalid segment_index %d or offset %lu; malformed PTD file.",
          metadata->segment_index(),
          metadata->offset());
      return metadata;
    }
  }
  return Error::NotFound;
}

Result<const TensorLayout> create_tensor_layout(
    const flat_tensor_flatbuffer::TensorMetadata* tensor_metadata) {
  ScalarType scalar_type =
      static_cast<ScalarType>(tensor_metadata->scalar_type());
  const int dim = tensor_metadata->sizes()->size();
  const auto serialized_sizes = tensor_metadata->sizes()->data();
  const auto serialized_dim_order = tensor_metadata->dim_order()->data();
  return TensorLayout::create(
      Span<const int32_t>(serialized_sizes, dim),
      Span<const uint8_t>(serialized_dim_order, dim),
      scalar_type);
}

} // namespace

ET_NODISCARD Result<const TensorLayout> FlatTensorDataMap::get_metadata(
    const char* key) const {
  Result<const flat_tensor_flatbuffer::TensorMetadata*> metadata_res =
      get_flat_tensor_metadata(key, flat_tensor_->tensors());
  if (!metadata_res.ok()) {
    return metadata_res.error();
  }
  return create_tensor_layout(metadata_res.get());
}

ET_NODISCARD Result<FreeableBuffer> FlatTensorDataMap::get_data(
    const char* key) const {
  Result<const flat_tensor_flatbuffer::TensorMetadata*> metadata =
      get_flat_tensor_metadata(key, flat_tensor_->tensors());
  if (!metadata.ok()) {
    return metadata.error();
  }
  Result<const TensorLayout> tensor_layout =
      create_tensor_layout(metadata.get());
  if (!tensor_layout.ok()) {
    return tensor_layout.error();
  }

  // Load constant data.
  int segment_offset =
      flat_tensor_->segments()->Get(metadata.get()->segment_index())->offset();
  return loader_->load(
      header_.segment_base_offset + segment_offset + metadata.get()->offset(),
      tensor_layout.get().nbytes(),
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::External));
}

ET_NODISCARD Result<size_t> FlatTensorDataMap::load_data_into(
    ET_UNUSED const char* key,
    ET_UNUSED void* buffer,
    ET_UNUSED size_t size) const {
  // Get metadata to get nbytes.
  Result<const flat_tensor_flatbuffer::TensorMetadata*> metadata =
      get_flat_tensor_metadata(key, flat_tensor_->tensors());
  if (!metadata.ok()) {
    return metadata.error();
  }
  Result<const TensorLayout> tensor_layout =
      create_tensor_layout(metadata.get());
  if (!tensor_layout.ok()) {
    return tensor_layout.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      size < tensor_layout.get().nbytes(),
      InvalidArgument,
      "Buffer size %zu is smaller than tensor size %zu",
      size,
      tensor_layout.get().nbytes())

  int segment_offset =
      flat_tensor_->segments()->Get(metadata.get()->segment_index())->offset();
  DataLoader::SegmentInfo info = DataLoader::SegmentInfo(
      DataLoader::SegmentInfo::Type::Mutable, 0, nullptr);

  return loader_->load_into(
      header_.segment_base_offset + segment_offset + metadata.get()->offset(),
      tensor_layout.get().nbytes(),
      info,
      buffer);
}

ET_NODISCARD Result<size_t> FlatTensorDataMap::get_num_keys() const {
  return flat_tensor_->tensors()->size();
}

ET_NODISCARD Result<const char*> FlatTensorDataMap::get_key(
    size_t index) const {
  if (index < 0 || index >= flat_tensor_->tensors()->size()) {
    return Error::InvalidArgument;
  }
  return flat_tensor_->tensors()->Get(index)->fully_qualified_name()->c_str();
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

  return FlatTensorDataMap(
      fh.get(), std::move(flat_tensor_data.get()), flat_tensor, loader);
}

} // namespace extension
} // namespace executorch
