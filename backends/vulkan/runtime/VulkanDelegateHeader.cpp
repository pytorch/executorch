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

namespace torch {
namespace executor {
namespace vulkan {

namespace {

struct ByteSlice {
  size_t offset;
  size_t size;
};

constexpr size_t kExpectedSize = 42;
constexpr char kExpectedMagic[4] = {'V', 'K', 'D', 'G'};

constexpr ByteSlice kMagic = {4, 4};
constexpr ByteSlice kHeaderSize = {8, 2};
constexpr ByteSlice kFlatbufferOffset = {10, 4};
constexpr ByteSlice kFlatbufferSize = {14, 4};
constexpr ByteSlice kConstantsOffset = {18, 4};
constexpr ByteSlice kConstantsSize = {22, 8};
constexpr ByteSlice kShadersOffset = {30, 4};
constexpr ByteSlice kShadersSize = {34, 8};

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

/// Interprets the 2 bytes at `data` as a little-endian uint32_t.
uint32_t GetUInt16LE(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8);
}

} // namespace

Result<VulkanDelegateHeader> VulkanDelegateHeader::Parse(const void* data) {
  const uint8_t* header_data = (const uint8_t*)data;

  const uint8_t* magic_start = header_data + kMagic.offset;
  if (std::memcmp(magic_start, kExpectedMagic, kMagic.size) != 0) {
    return Error::NotFound;
  }

  const uint32_t header_size = GetUInt16LE(header_data + kHeaderSize.offset);
  if (header_size < kExpectedSize) {
    return Error::InvalidArgument;
  }

  return VulkanDelegateHeader{
      GetUInt32LE(header_data + kFlatbufferOffset.offset),
      GetUInt32LE(header_data + kFlatbufferSize.offset),
      GetUInt32LE(header_data + kConstantsOffset.offset),
      GetUInt64LE(header_data + kConstantsSize.offset),
      GetUInt32LE(header_data + kShadersOffset.offset),
      GetUInt32LE(header_data + kShadersSize.offset),
  };
}

} // namespace vulkan
} // namespace executor
} // namespace torch
