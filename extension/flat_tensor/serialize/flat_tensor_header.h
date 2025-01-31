/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/result.h>

namespace executorch {
namespace extension {

/**
 * A FlatTensor header found at the beginning of a flat_tensor-serialized blob.
 */
struct FlatTensorHeader {
  /**
   * To find the header, callers should provide at least this many bytes of the
   * head of the serialized FlatTensor data.
   */
  static constexpr size_t kNumHeadBytes = 64;

  /**
   * The offset into the serialized FlatTensor data where the FlatTensor
   * header should begin.
   */
  static constexpr size_t kHeaderOffset = 8;

  /**
   * The magic bytes that identify the header. This should be in sync with
   * the magic in executorch/extension/flat_tensor/serialize/serialize.py
   *
   * This is the canonical definition of the expected value. If the header
   * layout ever changes in a compatibility-breaking way, increment the digits
   * in the magic. But, doing so will prevent older binaries from recognizing
   * the presence of the header. The compatibility-preserving way to make
   * changes is to increase the header's length field and add new fields at the
   * end.
   */
  static constexpr size_t kMagicSize = 4;
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr char kMagic[kMagicSize] = {'F', 'H', '0', '1'};

  /// The expected length of the header, in bytes.
  static constexpr uint32_t kHeaderExpectedLength =
      // Header magic
      4
      // Header length
      + 4
      // Flatbuffer offset
      + 8
      // Flatbuffer data size
      + 8
      // Segment base offset
      + 8
      // Data size
      + 8;

  /**
   * Look for and parse a FlatTensorHeader in the provided data.
   *
   * @param[in] data The contents of the beginning of the serialized binary
   *     FlatTensor data, starting at offset 0 (i.e., the head of the file).
   * @param[in] size Length of `data` in bytes. Must be >= kNumHeadBytes or this
   *     call will fail.
   *
   * @returns a FlatTensorHeader if the header was found and is valid. Returns
   * an error if size was too short, if the header was not found, or if the
   *     header appeared to be corrupt.
   */
  static runtime::Result<FlatTensorHeader> Parse(const void* data, size_t size);

  /// Offset of the FlatTensor flatbuffer in the serialized binary.
  uint64_t flatbuffer_offset;

  /// Size of the flatbuffer in bytes.
  uint64_t flatbuffer_size;

  /// The offset in bytes of the first segment.
  uint64_t segment_base_offset;

  /// Size of all the segment data, in bytes.
  uint64_t segment_data_size;
};

} // namespace extension
} // namespace executorch
