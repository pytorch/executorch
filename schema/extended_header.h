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
namespace runtime {

/**
 * An extended, ExecuTorch-specific header that may be embedded in the
 * serialized Program data header.
 *
 * For details see //executorch/docs/source/pte-file-format.md
 */
struct ExtendedHeader {
  /**
   * To find the header, callers should provide at least this many bytes of the
   * head of the serialized Program data.
   */
  static constexpr size_t kNumHeadBytes = 64;

  /**
   * The offset into the Program serialized program data where the extended
   * header should begin.
   */
  static constexpr size_t kHeaderOffset = 8;

  /**
   * The magic bytes that identify the header.
   *
   * This is the canonical definition of the expected value. If the header
   * layout ever changes in a compatibility-breaking way, increment the digits
   * in the magic. But, doing so will prevent older binaries from recognizing
   * the presence of the header. The compatibility-preserving way to make
   * changes is to increase the header's length field and add new fields at the
   * end.
   */
  static constexpr size_t kMagicSize = 4;
  static constexpr char kMagic[kMagicSize] = {'e', 'h', '0', '0'};

  /**
   * Look for and parse an ExtendedHeader in the provided data.
   *
   * @param[in] data The contents of the beginning of the serialized binary
   *     Program data, starting at offset 0 (i.e., the head of the file).
   * @param[in] size Length of `data` in bytes. Must be >= kNumHeadBytes or this
   *     call will fail.
   *
   * @returns an ExtendedHeader if the header was found and is valid. Returns an
   *     error if size was too short, if the header was not found, or if the
   *     header appeared to be corrupt.
   */
  static Result<ExtendedHeader> Parse(const void* data, size_t size);

  /**
   * The size in bytes of the Program flatbuffer data, starting from offset
   * zero.
   */
  uint64_t program_size;

  /**
   * The offset in bytes of the first segment, if present. Zero if no segment
   * is present.
   */
  uint64_t segment_base_offset;
};

} // namespace runtime
} // namespace executorch
