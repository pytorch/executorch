/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstring>

namespace executorch {
namespace backends {
namespace tensorrt {

/**
 * Magic bytes identifying a TensorRT blob.
 * "TR01" = TensorRT version 1 format with I/O metadata.
 */
constexpr char kTensorRTMagic[4] = {'T', 'R', '0', '1'};

/**
 * Header size in bytes (32 bytes, 16-byte aligned).
 */
constexpr uint32_t kHeaderSize = 32;

/**
 * TensorRT blob header structure.
 *
 * Layout (little-endian, 32 bytes total):
 *   magic (4 bytes) - "TR01"
 *   metadata_offset (4 bytes) - offset to metadata JSON from start
 *   metadata_size (4 bytes) - size of metadata JSON in bytes
 *   engine_offset (4 bytes) - offset to engine data from start
 *   engine_size (8 bytes) - size of engine data in bytes
 *   reserved (8 bytes) - for future use
 */
struct TensorRTBlobHeader {
  char magic[4];
  uint32_t metadata_offset;
  uint32_t metadata_size;
  uint32_t engine_offset;
  uint64_t engine_size;
  uint8_t reserved[8];

  /**
   * Check if this is a valid TensorRT blob header.
   *
   * @return true if magic bytes match "TR01".
   */
  bool is_valid() const {
    return std::memcmp(magic, kTensorRTMagic, 4) == 0;
  }
};

static_assert(sizeof(TensorRTBlobHeader) == 32, "Header must be 32 bytes");

/**
 * Parse a TensorRT blob header from raw bytes.
 *
 * @param data Pointer to blob data (must be at least kHeaderSize bytes).
 * @param data_size Size of data buffer in bytes.
 * @param out_header Output header structure.
 * @return true if header was parsed successfully.
 */
inline bool parse_blob_header(
    const void* data,
    size_t data_size,
    TensorRTBlobHeader& out_header) {
  if (data == nullptr || data_size < kHeaderSize) {
    return false;
  }

  std::memcpy(&out_header, data, sizeof(TensorRTBlobHeader));
  return out_header.is_valid();
}

/**
 * Get a pointer to the engine data within a blob.
 *
 * @param data Pointer to blob data.
 * @param data_size Size of data buffer in bytes.
 * @param header Parsed header from parse_blob_header().
 * @param out_engine Output pointer to engine data.
 * @param out_engine_size Output size of engine data.
 * @return true if engine data was located successfully.
 */
inline bool get_engine_from_blob(
    const void* data,
    size_t data_size,
    const TensorRTBlobHeader& header,
    const void*& out_engine,
    size_t& out_engine_size) {
  if (data == nullptr || !header.is_valid()) {
    return false;
  }

  const size_t end_offset = header.engine_offset + header.engine_size;
  if (end_offset > data_size) {
    return false;
  }

  out_engine = static_cast<const uint8_t*>(data) + header.engine_offset;
  out_engine_size = static_cast<size_t>(header.engine_size);
  return true;
}

/**
 * Get a pointer to the metadata JSON within a blob.
 *
 * @param data Pointer to blob data.
 * @param data_size Size of data buffer in bytes.
 * @param header Parsed header from parse_blob_header().
 * @param out_metadata Output pointer to metadata JSON.
 * @param out_metadata_size Output size of metadata JSON.
 * @return true if metadata was located successfully.
 */
inline bool get_metadata_from_blob(
    const void* data,
    size_t data_size,
    const TensorRTBlobHeader& header,
    const void*& out_metadata,
    size_t& out_metadata_size) {
  if (data == nullptr || !header.is_valid()) {
    return false;
  }

  const size_t end_offset = header.metadata_offset + header.metadata_size;
  if (end_offset > data_size) {
    return false;
  }

  out_metadata = static_cast<const uint8_t*>(data) + header.metadata_offset;
  out_metadata_size = static_cast<size_t>(header.metadata_size);
  return true;
}

} // namespace tensorrt
} // namespace backends
} // namespace executorch
