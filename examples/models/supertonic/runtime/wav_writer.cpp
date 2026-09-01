/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "wav_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace supertonic {
namespace {

void write_u16(std::ostream& output, uint16_t value) {
  const char bytes[2] = {
      static_cast<char>(value & 0xff), static_cast<char>((value >> 8) & 0xff)};
  output.write(bytes, sizeof(bytes));
}

void write_u32(std::ostream& output, uint32_t value) {
  const char bytes[4] = {
      static_cast<char>(value & 0xff),
      static_cast<char>((value >> 8) & 0xff),
      static_cast<char>((value >> 16) & 0xff),
      static_cast<char>((value >> 24) & 0xff)};
  output.write(bytes, sizeof(bytes));
}

int16_t pcm16(float sample) {
  if (!std::isfinite(sample)) {
    sample = 0.0f;
  }
  sample = std::clamp(sample, -1.0f, 1.0f);
  return sample < 0.0f ? static_cast<int16_t>(std::lround(sample * 32768.0f))
                       : static_cast<int16_t>(std::lround(sample * 32767.0f));
}

} // namespace

WavLayout
validate_wav_layout(size_t sample_count, int sample_rate, int channels) {
  if (sample_rate <= 0) {
    throw std::invalid_argument("sample rate must be positive");
  }
  if (channels <= 0 ||
      channels > static_cast<int>(std::numeric_limits<uint16_t>::max())) {
    throw std::invalid_argument(
        "channels must be in the representable WAV range");
  }
  if (sample_count % static_cast<size_t>(channels) != 0) {
    throw std::invalid_argument(
        "sample count must contain a whole number of frames");
  }
  const uint64_t block_align =
      static_cast<uint64_t>(channels) * sizeof(int16_t);
  if (block_align > std::numeric_limits<uint16_t>::max()) {
    throw std::overflow_error("WAV block align is unrepresentable");
  }
  const uint64_t byte_rate = static_cast<uint64_t>(sample_rate) * block_align;
  if (byte_rate > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("WAV byte rate is unrepresentable");
  }
  if (sample_count >
      (std::numeric_limits<uint32_t>::max() - 36) / sizeof(int16_t)) {
    throw std::overflow_error("WAV data size is unrepresentable");
  }
  const uint32_t data_bytes =
      static_cast<uint32_t>(sample_count * sizeof(int16_t));
  return {
      data_bytes,
      static_cast<uint32_t>(36 + data_bytes),
      static_cast<uint16_t>(block_align),
      static_cast<uint32_t>(byte_rate)};
}

bool write_pcm16_wav(
    const std::string& path,
    const std::vector<float>& samples,
    int sample_rate,
    int channels) {
  WavLayout layout;
  try {
    layout = validate_wav_layout(samples.size(), sample_rate, channels);
  } catch (const std::exception&) {
    return false;
  }
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    return false;
  }
  output.write("RIFF", 4);
  write_u32(output, layout.riff_size);
  output.write("WAVEfmt ", 8);
  write_u32(output, 16);
  write_u16(output, 1);
  write_u16(output, static_cast<uint16_t>(channels));
  write_u32(output, static_cast<uint32_t>(sample_rate));
  write_u32(output, layout.byte_rate);
  write_u16(output, layout.block_align);
  write_u16(output, 16);
  output.write("data", 4);
  write_u32(output, layout.data_bytes);
  for (float sample : samples) {
    write_u16(output, static_cast<uint16_t>(pcm16(sample)));
  }
  return output.good();
}

} // namespace supertonic
