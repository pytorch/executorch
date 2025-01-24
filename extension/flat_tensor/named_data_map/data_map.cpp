/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/flat_tensor/named_data_map/data_map.h>
#include <executorch/extension/flat_tensor/serialize/flat_tensor_header.h>
#include <executorch/extension/flat_tensor/serialize/schema_generated.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/compiler.h>

#include <tuple>
#include <unordered_map>

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

bool IsAligned(const void* data) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(data);
  return addr % kMinimumAlignment == 0;
}
} // namespace

ET_NODISCARD Result<const TensorLayout> DataMap::get_metadata(
    const char* fqn) const {
  auto result = _name_to_tensor.find(fqn);
  if (result == _name_to_tensor.end()) {
    return Error::InvalidArgument;
  }
  // value is a tuple of (segment_index, offset, tensor_layout)
  return std::get<2>(result->second);
}

ET_NODISCARD Result<FreeableBuffer> DataMap::get_data(const char* fqn) const {
  auto result = _name_to_tensor.find(fqn);
  if (result == _name_to_tensor.end()) {
    return Error::InvalidArgument;
  }
  int offset = std::get<1>(result->second);
  TensorLayout tensor = std::get<2>(result->second);

  const uint8_t* data = static_cast<const uint8_t*>(_data_ro.data()) + offset;
  return FreeableBuffer(data, tensor.nbytes(), nullptr);
}

ET_NODISCARD Result<size_t>
DataMap::load_data_into(const char* fqn, size_t size, void* buffer) const {
  return Error::NotImplemented;
}

ET_NODISCARD Result<size_t> DataMap::get_num_keys() const {
  return _name_to_tensor.size();
}

ET_NODISCARD Result<const char*> DataMap::get_key(size_t index) const {
  if (index < 0 || index >= _name_to_tensor.size()) {
    return Error::InvalidArgument;
  }

  auto iter = _name_to_tensor.begin();
  for (int i = 0; i < index; ++i) {
    ++iter;
  }
  return iter->first.c_str();
}

/* static */ Result<DataMap> DataMap::load(DataLoader* loader) {
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
      /*offset=*/flatbuffer_offset,
      flatbuffer_size,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::External));
  if (!flat_tensor_data.ok()) {
    return flat_tensor_data.error();
  }

  // Make sure magic matches.
  if (!flat_tensor::FlatTensorBufferHasIdentifier(flat_tensor_data->data())) {
    ET_LOG(
        Error,
        "FlatTensor identifier '%.4s' != expected '%.4s'",
        flatbuffers::GetBufferIdentifier(flat_tensor_data->data()),
        flat_tensor::FlatTensorIdentifier());
    return Error::InvalidExternalData;
  }

  // The flatbuffer data must start at an aligned address to ensure internal
  // alignment of flatbuffer fields.
  ET_CHECK_OR_RETURN_ERROR(
      IsAligned(flat_tensor_data->data()),
      InvalidArgument,
      "FlatTensor data 0x%p must be aligned to %zu",
      flat_tensor_data->data(),
      kMinimumAlignment);

  // Get pointer to root of flatbuffer table.
  const flat_tensor::FlatTensor* flat_tensor =
      flat_tensor::GetFlatTensor(flat_tensor_data->data());

  // Get pointer to tensor metadata.
  const auto* s_tensor_metadata = flat_tensor->tensors();
  assert(s_tensor_metadata != nullptr);

  std::unordered_map<std::string, std::tuple<int, int, TensorLayout>>
      name_to_tensor = {};
  for (int i = 0; i < s_tensor_metadata->size(); i++) {
    // Create TensorLayouts.
    ScalarType scalar_type =
        static_cast<ScalarType>(s_tensor_metadata->Get(i)->scalar_type());
    const int dim = s_tensor_metadata->Get(i)->sizes()->size();

    const auto serialized_sizes = s_tensor_metadata->Get(i)->sizes()->data();
    const auto serialized_dim_order =
        s_tensor_metadata->Get(i)->dim_order()->data();
    TensorLayout tensor_layout = TensorLayout(
        scalar_type,
        Span<const int32_t>(serialized_sizes, dim),
        Span<const uint8_t>(serialized_dim_order, dim));

    int segment_index = s_tensor_metadata->Get(i)->segment_index();
    int offset = s_tensor_metadata->Get(i)->offset();
    std::string fqn = s_tensor_metadata->Get(i)->fully_qualified_name()->str();

    auto val = std::make_tuple(segment_index, offset, tensor_layout);
    name_to_tensor.insert({fqn, std::move(val)});
  }

  // Load constant data.
  const auto* s_data_segment = flat_tensor->segments();

  // Only support one segment for now.
  assert(s_data_segment->size() == 1);
  // First segment offset should be 0.
  int segment_offset = s_data_segment->Get(0)->offset();
  assert(segment_offset == 0);
  // First segment size should be <= the total segment data size.
  int segment_size = s_data_segment->Get(0)->size();
  assert(segment_size <= segment_data_size);

  Result<FreeableBuffer> _data_ro = loader->load(
      /*offset=*/segment_base_offset + segment_offset,
      segment_size,
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::External));
  if (!_data_ro.ok()) {
    return _data_ro.error();
  }

  return DataMap(
      std::move(flat_tensor_data.get()),
      std::move(name_to_tensor),
      std::move(_data_ro.get()));
}

DataMap::~DataMap() {}

} // namespace extension
} // namespace executorch
