/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "wav_writer.h"

#include <algorithm>

namespace voxtral_tts {
namespace {

void write_u16(std::ofstream& file, std::uint16_t value) {
  file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_u32(std::ofstream& file, std::uint32_t value) {
  file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

} // namespace

WavWriter::WavWriter(const std::string& path, int sample_rate, int num_channels)
    : file_(path, std::ios::binary),
      sample_rate_(sample_rate),
      num_channels_(num_channels) {
  if (file_.is_open()) {
    WriteHeaderPlaceholder();
  }
}

WavWriter::~WavWriter() {
  Close();
}

bool WavWriter::IsOpen() const {
  return file_.is_open() && !closed_;
}

bool WavWriter::Write(const float* samples, std::size_t frame_count) {
  if (!IsOpen() || samples == nullptr) {
    return false;
  }

  const std::size_t sample_count =
      frame_count * static_cast<std::size_t>(num_channels_);
  for (std::size_t i = 0; i < sample_count; ++i) {
    const float clipped = std::clamp(samples[i], -1.0f, 1.0f);
    auto pcm = static_cast<std::int16_t>(clipped * 32767.0f);
    file_.write(reinterpret_cast<const char*>(&pcm), sizeof(pcm));
  }

  data_bytes_ +=
      static_cast<std::uint32_t>(sample_count * sizeof(std::int16_t));
  return file_.good();
}

bool WavWriter::Close() {
  if (!file_.is_open() || closed_) {
    return true;
  }
  closed_ = true;
  const bool ok = FinalizeHeader();
  file_.close();
  return ok;
}

void WavWriter::WriteHeaderPlaceholder() {
  const std::uint16_t bits_per_sample = 16;
  const std::uint32_t byte_rate = static_cast<std::uint32_t>(
      sample_rate_ * num_channels_ * bits_per_sample / 8);
  const std::uint16_t block_align =
      static_cast<std::uint16_t>(num_channels_ * bits_per_sample / 8);

  file_.write("RIFF", 4);
  write_u32(file_, 0);
  file_.write("WAVE", 4);
  file_.write("fmt ", 4);
  write_u32(file_, 16);
  write_u16(file_, 1); // PCM
  write_u16(file_, static_cast<std::uint16_t>(num_channels_));
  write_u32(file_, static_cast<std::uint32_t>(sample_rate_));
  write_u32(file_, byte_rate);
  write_u16(file_, block_align);
  write_u16(file_, bits_per_sample);
  file_.write("data", 4);
  write_u32(file_, 0);
}

bool WavWriter::FinalizeHeader() {
  if (!file_.good()) {
    return false;
  }
  const std::uint32_t riff_size = 36 + data_bytes_;
  file_.seekp(4, std::ios::beg);
  write_u32(file_, riff_size);
  file_.seekp(40, std::ios::beg);
  write_u32(file_, data_bytes_);
  file_.seekp(0, std::ios::end);
  return file_.good();
}

} // namespace voxtral_tts
