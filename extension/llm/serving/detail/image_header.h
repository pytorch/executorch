/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/runner/image.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>

namespace executorch::extension::llm::serving::detail {

// Allocation-free source-size preflight, not a codec validator. The CPU hook
// must still reject corrupt streams, enforce scratch limits, and own output.
inline bool bounded_image_header(
    const EncodedImage& image,
    std::size_t max_dimension,
    std::size_t max_pixels) {
  const auto& data = image.data;
  const auto bounded = [&](std::uint32_t width, std::uint32_t height) {
    return width != 0 && height != 0 && width <= max_dimension &&
        height <= max_dimension && width <= max_pixels / height;
  };
  if (image.mime_type == "image/png") {
    constexpr std::uint8_t signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (data.size() < 33 ||
        !std::equal(std::begin(signature), std::end(signature), data.begin())) {
      return false;
    }
    const auto u32 = [&](std::size_t offset) {
      return (std::uint32_t{data[offset]} << 24) |
          (std::uint32_t{data[offset + 1]} << 16) |
          (std::uint32_t{data[offset + 2]} << 8) | data[offset + 3];
    };
    // IHDR must be the first chunk and contain its complete 13-byte payload
    // and CRC. CRC and pixel-data validation belong to the actual decoder.
    return u32(8) == 13 && data[12] == 'I' && data[13] == 'H' &&
        data[14] == 'D' && data[15] == 'R' && bounded(u32(16), u32(20));
  }
  if (image.mime_type != "image/jpeg" || data.size() < 4 || data[0] != 0xff ||
      data[1] != 0xd8) {
    return false;
  }
  const auto u16 = [&](std::size_t offset) {
    return (std::uint32_t{data[offset]} << 8) | data[offset + 1];
  };
  std::size_t offset = 2;
  while (offset < data.size()) {
    if (data[offset++] != 0xff) {
      return false;
    }
    while (offset < data.size() && data[offset] == 0xff) {
      ++offset;
    }
    if (offset == data.size()) {
      return false;
    }
    const auto marker = data[offset++];
    if (marker == 0 || marker == 0xd8 || marker == 0xd9 || marker == 0xda) {
      return false;
    }
    // Standalone TEM/restart markers have no length field.
    if (marker == 1 || (marker >= 0xd0 && marker <= 0xd7)) {
      continue;
    }
    if (data.size() - offset < 2) {
      return false;
    }
    const auto length = u16(offset);
    if (length < 2 || length > data.size() - offset) {
      return false;
    }
    const bool frame = marker >= 0xc0 && marker <= 0xcf && marker != 0xc4 &&
        marker != 0xc8 && marker != 0xcc;
    if (frame) {
      return length >= 8 && data[offset + 7] != 0 && data[offset + 7] <= 4 &&
          length == 8 + 3u * data[offset + 7] &&
          bounded(u16(offset + 5), u16(offset + 3));
    }
    offset += length;
  }
  return false;
}

} // namespace executorch::extension::llm::serving::detail
