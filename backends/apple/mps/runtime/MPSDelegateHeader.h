//
//  Copyright (c) 2024 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#pragma once

#include <executorch/runtime/core/result.h>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

/**
 * MPS-header that is embedded before the flatbuffer payload
 *
 */
struct MPSDelegateHeader {
  /**
   * The minimum size of the MPSDelegateHeader. The caller should provide at
   * least this many bytes of the head of the serialized MPS Data
   */
  static constexpr size_t kMinSize = 30;

  /**
   * The magic offset. This offset is the same as the offset for flatbuffer
   * header so we will be able to check if the header is is either the
   * flatbuffer head or the wrapper header we introduce here
   */
  static constexpr size_t kMagicOffset = 4;

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
  static constexpr char kMagic[kMagicSize] = {'M', 'P', '0', '0'};

  /**
   * The size in bytes of the header length. We store 2 bytes for the header
   * length
   */
  static constexpr size_t kHeaderLengthSize = 2;

  /**
   * The expected location of the header length field relative to the beginning
   * of the header.
   */
  static constexpr size_t kHeaderLengthOffset =
      MPSDelegateHeader::kMagicOffset + MPSDelegateHeader::kMagicSize;

  /*
   * The expected location of the constant data offset field relative to the
   * beginning of the header.
   */
  static constexpr size_t kConstantDataSegmentOffset = kHeaderLengthOffset;

  /*
   * The expected location of the constant data size field relative to the
   * beginning of the header.
   */
  static constexpr size_t kConstantDataSizeOffset =
      kConstantDataSegmentOffset + sizeof(uint64_t);

  /**
   * The expected location of the flatbuffer data offset field relative to the
   * beginning of the header.
   */
  static constexpr size_t kFlatbufferDataOffsetOffset =
      kConstantDataSizeOffset + sizeof(uint64_t);

  /**
   * Look for and parse an ExtendedHeader in the provided data.
   *
   * @param[in] data The contents of the beginning of the serialized binary
   *     Program data, starting at offset 0 (i.e., the head of the file).
   * @param[in] size Length of `data` in bytes.
   *
   * @returns an MPSHeader if the header was found and is valid. Returns an
   *     error if size was too short, if the header was not found, or if the
   *     header appeared to be corrupt.
   */
  static Result<MPSDelegateHeader> Parse(const void* data, size_t size);

  /**
   * The offset in bytes to the beginning of the constant data.
   */
  uint64_t constant_data_offset;
  /**
   * The size in bytes of the constant data.
   */
  uint64_t constant_data_size;
  /**
   * The offset in bytes to the beginning of the flatbuffer data.
   */
  uint64_t flatbuffer_offset;
  /**
   * The size in bytes of the flatbuffer data.
   */
  uint64_t flatbuffer_size;
};

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
