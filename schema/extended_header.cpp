/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/schema/extended_header.h>

#include <cinttypes>
#include <cstring>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#pragma clang diagnostic ignored "-Wdeprecated"

namespace executorch {
namespace runtime {

namespace {

/// The expected location of the header length field relative to the beginning
/// of the header.
static constexpr size_t kHeaderLengthOffset = ExtendedHeader::kMagicSize;

/// The expected location of the program_size field relative to the beginning of
/// the header.
static constexpr size_t kHeaderProgramSizeOffset =
    kHeaderLengthOffset + sizeof(uint32_t);

/// The expected location of the segment_base_offset field relative to the
/// beginning of the header.
static constexpr size_t kHeaderSegmentBaseOffsetOffset =
    kHeaderProgramSizeOffset + sizeof(uint64_t);

/**
 * The size of the header that covers the fields known of by this version of
 * the code. It's ok for a header to be larger as long as the fields stay in
 * the same place, but this code will ignore any new fields.
 */
static constexpr size_t kMinimumHeaderLength =
    kHeaderSegmentBaseOffsetOffset + sizeof(uint64_t);

/// Interprets the 4 bytes at `data` as a little-endian uint32_t.
uint32_t GetUInt32LE(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
      ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

/// Interprets the 8 bytes at `data` as a little-endian uint64_t.
uint64_t GetUInt64LE(const uint8_t* data) {
  return (uint64_t)data[0] | ((uint64_t)data[1] << 8) |
      ((uint64_t)data[2] << 16) | ((uint64_t)data[3] << 24) |
      ((uint64_t)data[4] << 32) | ((uint64_t)data[5] << 40) |
      ((uint64_t)data[6] << 48) | ((uint64_t)data[7] << 56);
}

} // namespace

/* static */ Result<ExtendedHeader> ExtendedHeader::Parse(
    const void* data,
    size_t size) {
  if (size < ExtendedHeader::kNumHeadBytes) {
    return Error::InvalidArgument;
  }
  const uint8_t* header =
      reinterpret_cast<const uint8_t*>(data) + kHeaderOffset;

  // Check magic bytes.
  if (std::memcmp(header, ExtendedHeader::kMagic, ExtendedHeader::kMagicSize) !=
      0) {
    return Error::NotFound;
  }

  // Check header length.
  uint32_t header_length = GetUInt32LE(header + kHeaderLengthOffset);
  if (header_length < kMinimumHeaderLength) {
    ET_LOG(
        Error,
        "Extended header length %" PRIu32 " < %zu",
        header_length,
        kMinimumHeaderLength);
    return Error::InvalidProgram;
  }

  // The header is present and apparently valid.
  return ExtendedHeader{
      /*program_size=*/GetUInt64LE(header + kHeaderProgramSizeOffset),
      /*segment_base_offset=*/
      GetUInt64LE(header + kHeaderSegmentBaseOffsetOffset),
  };
}

// Define storage for the static.
constexpr char ExtendedHeader::kMagic[kMagicSize];

} // namespace runtime
} // namespace executorch
