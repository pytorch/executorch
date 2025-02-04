/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNHeader.h>

#include <cstring>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#pragma clang diagnostic ignored "-Wdeprecated"

namespace executorch {
namespace backends {
namespace xnnpack {
namespace delegate {

using executorch::runtime::Error;
using executorch::runtime::Result;

namespace {
/// Interprets the 8 bytes at `data` as a little-endian uint64_t.
uint64_t GetUInt64LE(const uint8_t* data) {
  return (uint64_t)data[0] | ((uint64_t)data[1] << 8) |
      ((uint64_t)data[2] << 16) | ((uint64_t)data[3] << 24) |
      ((uint64_t)data[4] << 32) | ((uint64_t)data[5] << 40) |
      ((uint64_t)data[6] << 48) | ((uint64_t)data[7] << 56);
}

/// Interprets the 4 bytes at `data` as a little-endian uint32_t.
uint32_t GetUInt32LE(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
      ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

} // namespace

Result<XNNHeader> XNNHeader::Parse(const void* data, size_t size) {
  const uint8_t* header_data = (const uint8_t*)data;

  if (size < XNNHeader::kMinSize) {
    return Error::InvalidArgument;
  }

  const uint8_t* magic_start = header_data + XNNHeader::kMagicOffset;
  if (std::memcmp(magic_start, XNNHeader::kMagic, XNNHeader::kMagicSize) != 0) {
    return Error::NotFound;
  }

  uint32_t flatbuffer_offset =
      GetUInt32LE(header_data + XNNHeader::kFlatbufferDataOffsetOffset);

  uint32_t flatbuffer_size =
      GetUInt32LE(header_data + XNNHeader::kFlatbufferDataSizeOffset);

  uint32_t constant_data_offset =
      GetUInt32LE(header_data + XNNHeader::kConstantDataOffsetOffset);

  uint64_t constant_data_size =
      GetUInt64LE(header_data + XNNHeader::kConstantDataSizeOffset);

  return XNNHeader{
      flatbuffer_offset,
      flatbuffer_size,
      constant_data_offset,
      constant_data_size};
}

// Define storage for the static.
constexpr char XNNHeader::kMagic[kMagicSize];

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
