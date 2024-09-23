//
//  Copyright (c) 2024 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSDelegateHeader.h>

#include <cstring>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

/// Interprets the 8 bytes at `data` as a little-endian uint64_t.
uint64_t getUInt64LE(const uint8_t* data) {
  return (uint64_t)data[0] | ((uint64_t)data[1] << 8) |
      ((uint64_t)data[2] << 16) | ((uint64_t)data[3] << 24) |
      ((uint64_t)data[4] << 32) | ((uint64_t)data[5] << 40) |
      ((uint64_t)data[6] << 48) | ((uint64_t)data[7] << 56);
}

Result<MPSDelegateHeader> MPSDelegateHeader::Parse(const void* data, size_t size) {
  const uint8_t* header_data = (const uint8_t*)data;

  if (size < MPSDelegateHeader::kMinSize) {
    return Error::InvalidArgument;
  }

  const uint8_t* magic_start = header_data + MPSDelegateHeader::kMagicOffset;
  if (std::memcmp(magic_start, MPSDelegateHeader::kMagic, MPSDelegateHeader::kMagicSize) != 0) {
    return Error::NotFound;
  }

  uint64_t constant_data_offset = getUInt64LE(header_data + MPSDelegateHeader::kConstantDataSegmentOffset);
  uint64_t constant_data_size = getUInt64LE(header_data + MPSDelegateHeader::kConstantDataSizeOffset);
  uint64_t flatbuffer_offset = MPSDelegateHeader::kFlatbufferDataOffsetOffset;
  uint64_t flatbuffer_size = size - flatbuffer_offset;

  return MPSDelegateHeader{
      constant_data_offset,
      constant_data_size,
      flatbuffer_offset,
      flatbuffer_size};
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
