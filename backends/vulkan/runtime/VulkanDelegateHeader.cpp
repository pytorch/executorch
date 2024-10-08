/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/VulkanDelegateHeader.h>

#include <cstring>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#pragma clang diagnostic ignored "-Wdeprecated"

namespace executorch {
namespace backends {
namespace vulkan {

using executorch::runtime::Error;
using executorch::runtime::Result;

namespace {

struct ByteSlice {
  size_t offset;
  size_t size;
};

constexpr size_t kExpectedSize = 30;
constexpr char kExpectedMagic[4] = {'V', 'H', '0', '0'};

constexpr ByteSlice kMagic = {4, 4};
constexpr ByteSlice kHeaderSize = {8, 2};
constexpr ByteSlice kFlatbufferOffset = {10, 4};
constexpr ByteSlice kFlatbufferSize = {14, 4};
constexpr ByteSlice kBytesOffset = {18, 4};
constexpr ByteSlice kBytesSize = {22, 8};

} // namespace

/// Interprets the 8 bytes at `data` as a little-endian uint64_t.
uint64_t getUInt64LE(const uint8_t* data) {
  return (uint64_t)data[0] | ((uint64_t)data[1] << 8) |
      ((uint64_t)data[2] << 16) | ((uint64_t)data[3] << 24) |
      ((uint64_t)data[4] << 32) | ((uint64_t)data[5] << 40) |
      ((uint64_t)data[6] << 48) | ((uint64_t)data[7] << 56);
}

/// Interprets the 4 bytes at `data` as a little-endian uint32_t.
uint32_t getUInt32LE(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
      ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

/// Interprets the 2 bytes at `data` as a little-endian uint32_t.
uint32_t getUInt16LE(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8);
}

bool VulkanDelegateHeader::is_valid() const {
  if (header_size < kExpectedSize) {
    return false;
  }
  if (flatbuffer_offset < header_size) {
    return false;
  }
  if (flatbuffer_size == 0) {
    return false;
  }
  if (bytes_offset < flatbuffer_offset + flatbuffer_size) {
    return false;
  }
  if (bytes_size < 0) {
    return false;
  }

  return true;
}

Result<VulkanDelegateHeader> VulkanDelegateHeader::parse(const void* data) {
  const uint8_t* header_data = (const uint8_t*)data;

  const uint8_t* magic_start = header_data + kMagic.offset;
  if (std::memcmp(magic_start, kExpectedMagic, kMagic.size) != 0) {
    return Error::NotFound;
  }

  VulkanDelegateHeader header = VulkanDelegateHeader{
      getUInt16LE(header_data + kHeaderSize.offset),
      getUInt32LE(header_data + kFlatbufferOffset.offset),
      getUInt32LE(header_data + kFlatbufferSize.offset),
      getUInt32LE(header_data + kBytesOffset.offset),
      getUInt64LE(header_data + kBytesSize.offset),
  };

  if (!header.is_valid()) {
    return Error::InvalidArgument;
  }

  return header;
}

} // namespace vulkan
} // namespace backends
} // namespace executorch
