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
namespace backends {
namespace xnnpack {
namespace delegate {

/**
 * An extended XNNPACK-header that is embeded before the flatbuffer payload
 *
 */
struct XNNHeader {
  /**
   * The minimum size of the XNNHeader. The caller should provide at least this
   * many bytes of the head of the serialized XNNPACK Data
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
  static constexpr char kMagic[kMagicSize] = {'X', 'H', '0', '0'};

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
      XNNHeader::kMagicOffset + XNNHeader::kMagicSize;

  /**
   * The expected location of the flatbuffer data offset field relative to the
   * beginning of the header.
   */
  static constexpr size_t kFlatbufferDataOffsetOffset =
      kHeaderLengthOffset + sizeof(uint16_t);

  /**
   * The expected location of the flatbuffer data size field relative to the
   * beginning of the header.
   */
  static constexpr size_t kFlatbufferDataSizeOffset =
      kFlatbufferDataOffsetOffset + sizeof(uint32_t);

  /*
   * The expected location of the constant data offset field relative to the
   * beginning of the header.
   */
  static constexpr size_t kConstantDataOffsetOffset =
      kFlatbufferDataSizeOffset + sizeof(uint32_t);

  /*
   * The expected location of the constant data size field relative to the
   * beginning of the header.
   */
  static constexpr size_t kConstantDataSizeOffset =
      kConstantDataOffsetOffset + sizeof(uint32_t);

  /**
   * Look for and parse an ExtendedHeader in the provided data.
   *
   * @param[in] data The contents of the beginning of the serialized binary
   *     Program data, starting at offset 0 (i.e., the head of the file).
   * @param[in] size Length of `data` in bytes.
   *
   * @returns an XNNHeader if the header was found and is valid. Returns an
   *     error if size was too short, if the header was not found, or if the
   *     header appeared to be corrupt.
   */
  static executorch::runtime::Result<XNNHeader> Parse(
      const void* data,
      size_t size);

  /**
   * The offset in bytes to the beginning of the flatbuffer data.
   */
  uint32_t flatbuffer_offset;
  /**
   * The size in bytes of the flatbuffer data.
   */
  uint32_t flatbuffer_size;

  /**
   * The offset in bytes to the beginning of the constant data.
   */
  uint32_t constant_data_offset;
  /**
   * The size in bytes of the constant data.
   */
  uint64_t constant_data_size;
};

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
