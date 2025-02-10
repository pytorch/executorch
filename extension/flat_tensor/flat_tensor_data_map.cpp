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
      // TODO(T214294528): Support multiple segments in FlatTensor.
      if (tensors->Get(i)->segment_index() != 0) {
        return Error::InvalidExternalData;
      }
      return tensors->Get(i);
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
  auto tensor_metadata = flat_tensor_->tensors();

  Result<const flat_tensor_flatbuffer::TensorMetadata*> metadata_res =
      get_flat_tensor_metadata(key, tensor_metadata);
  if (!metadata_res.ok()) {
    return metadata_res.error();
  }
  const auto metadata = metadata_res.get();
  if (metadata->segment_index() < 0 || metadata->offset() < 0) {
    // Invalid segment_index/offset; malformed PTD file.
    return Error::InvalidExternalData;
  }

  Result<const TensorLayout> tensor_layout_res = create_tensor_layout(metadata);
  if (!tensor_layout_res.ok()) {
    return tensor_layout_res.error();
  }

  // This FreeableBuffer doesn't own the underlying data, and will not free it,
  // which is why the free function is a nullptr.
  // TODO(T214294528): Remove data_ro_ and instead load the data here, letting
  // FreeableBuffer own it.
  return FreeableBuffer(
      static_cast<const uint8_t*>(data_ro_.data()) + metadata->offset(),
      tensor_layout_res.get().nbytes(),
      nullptr);
}

ET_NODISCARD Result<size_t> FlatTensorDataMap::load_data_into(
    ET_UNUSED const char* key,
    ET_UNUSED void* buffer,
    ET_UNUSED size_t size) const {
  return Error::NotImplemented;
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
  // Load data map.
  size_t flatbuffer_offset = 0;
  size_t flatbuffer_size = 0;
  size_t segment_base_offset = 0;
  size_t segment_data_size = 0;
  {
    // Check header.
    Result<FreeableBuffer> header = loader->load(
        /*offset=*/0,
        FlatTensorHeader::kNumHeadBytes,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::External));
    if (!header.ok()) {
      return header.error();
    }
    Result<FlatTensorHeader> fh =
        FlatTensorHeader::Parse(header->data(), header->size());
    if (fh.ok()) {
      // The header has the data map size.
      flatbuffer_offset = fh->flatbuffer_offset;
      flatbuffer_size = fh->flatbuffer_size;
      segment_base_offset = fh->segment_base_offset;
      segment_data_size = fh->segment_data_size;
    } else if (fh.error() == Error::NotFound) {
      // No header, throw error.
      ET_LOG(Error, "No FlatTensorHeader found.");
      return fh.error();
    } else {
      // corruption, throw error.
      ET_LOG(Error, "Flat tensor header may be corrupt.");
      return fh.error();
    }
  }

  // Load flatbuffer data as a segment.
  Result<FreeableBuffer> flat_tensor_data = loader->load(
      /*offset=*/0,
      flatbuffer_offset + flatbuffer_size,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::External));
  if (!flat_tensor_data.ok()) {
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

  // Validate flatbuffer data.
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(flat_tensor_data->data()),
      flat_tensor_data->size());
  bool ok = flat_tensor_flatbuffer::VerifyFlatTensorBuffer(verifier);
  ET_CHECK_OR_RETURN_ERROR(
      ok,
      InvalidExternalData,
      "Verification failed; data may be truncated or corrupt");

  // Get pointer to tensor metadata.
  const auto* s_tensor_metadata = flat_tensor->tensors();
  if (s_tensor_metadata == nullptr) {
    ET_LOG(Error, "FlatTensor has no tensor metadata.");
    return Error::InvalidExternalData;
  }

  // Load constant data.
  const auto* s_data_segment = flat_tensor->segments();

  // TODO(T214294528): Support multiple segments in FlatTensor.
  if (s_data_segment->size() != 1) {
    ET_LOG(
        Error,
        "FlatTensor has %u segments, only 1 supported.",
        s_data_segment->size());
  }
  // First segment size should be <= the total segment data size.
  int segment_size = s_data_segment->Get(0)->size();
  int segment_offset = s_data_segment->Get(0)->offset();
  if (segment_size > segment_data_size) {
    ET_LOG(
        Error,
        "FlatTensor segment size %d > segment data size %zu",
        segment_size,
        segment_data_size);
  }

  Result<FreeableBuffer> data_ro = loader->load(
      /*offset=*/segment_base_offset + segment_offset,
      segment_size,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::External));
  if (!data_ro.ok()) {
    return data_ro.error();
  }

  return FlatTensorDataMap(
      std::move(flat_tensor_data.get()), flat_tensor, std::move(data_ro.get()));
}

} // namespace extension
} // namespace executorch
