/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple WAV file loader.

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::extension::llm {
// See https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
constexpr uint16_t kWavFormatPcm = 0x0001;
constexpr uint16_t kWavFormatIeeeFloat = 0x0003;

constexpr float kOneOverIntMax = 1 / static_cast<float>(INT32_MAX);
constexpr float kOneOverShortMax = 1 / static_cast<float>(INT16_MAX);

struct WavHeader {
  /* RIFF Chunk Descriptor */
  uint8_t RIFF[4];
  uint32_t ChunkSize;
  uint8_t WAVE[4];
  /* "fmt" sub-chunk */
  uint8_t fmt[4];
  uint32_t Subchunk1Size;
  uint16_t AudioFormat;
  uint16_t NumOfChan;
  uint32_t SamplesPerSec;
  uint32_t bytesPerSec;
  uint16_t blockAlign;
  uint16_t bitsPerSample;
  /* "data" sub-chunk */
  uint32_t dataOffset;
  uint32_t Subchunk2Size;
};

namespace detail {

// Safe little-endian reads via memcpy (no strict aliasing violation,
// no alignment requirement). WAV files are always little-endian; these
// helpers assume the host is also little-endian (x86, ARM).
inline uint16_t read_le16(const char* p) {
  uint16_t val;
  std::memcpy(&val, p, sizeof(val));
  return val;
}

inline uint32_t read_le32(const char* p) {
  uint32_t val;
  std::memcpy(&val, p, sizeof(val));
  return val;
}

/// Read the entire file into a byte buffer. Returns an empty vector on failure.
inline std::vector<char> read_file(const std::string& fp) {
  std::ifstream file(fp, std::ios::binary);
  if (!file.is_open()) {
    return {};
  }
  file.seekg(0, std::ios::end);
  auto pos = file.tellg();
  if (pos < 0 || !file.good()) {
    return {};
  }
  size_t file_size = static_cast<size_t>(pos);
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(file_size);
  file.read(buffer.data(), file_size);
  if (static_cast<size_t>(file.gcount()) != file_size) {
    return {};
  }
  return buffer;
}

} // namespace detail

/// Parse a WAV header from a raw byte buffer.
/// Returns nullptr if the buffer does not contain a valid WAV file.
inline std::unique_ptr<WavHeader> load_wav_header(
    const char* data,
    size_t data_size) {
  bool has_riff = false;
  bool has_wave = false;

  if (data_size >= 4 && std::memcmp(data, "RIFF", 4) == 0) {
    has_riff = true;
  }

  if (data_size >= 12 && std::memcmp(data + 8, "WAVE", 4) == 0) {
    has_wave = true;
  }

  bool is_wav_file = has_riff && has_wave;
  if (!is_wav_file) {
    return nullptr;
  }

  // Minimum size: 12 (RIFF+size+WAVE) + 24 (fmt chunk header + min fields) = 36
  // We need at least up to bitsPerSample (offset 34, 2 bytes) = 36 bytes.
  constexpr size_t kMinHeaderSize = 36;
  if (data_size < kMinHeaderSize) {
    ET_LOG(
        Error,
        "WAV header detected but file is too small (%zu bytes) to contain a complete header",
        data_size);
    return nullptr;
  }

  auto header = std::make_unique<WavHeader>();

  // Parse RIFF chunk descriptor (bytes 0-11)
  std::memcpy(header->RIFF, data, 4);
  header->ChunkSize = detail::read_le32(data + 4);
  std::memcpy(header->WAVE, data + 8, 4);

  // Parse fmt sub-chunk (bytes 12-35)
  std::memcpy(header->fmt, data + 12, 4);
  header->Subchunk1Size = detail::read_le32(data + 16);
  header->AudioFormat = detail::read_le16(data + 20);
  header->NumOfChan = detail::read_le16(data + 22);
  header->SamplesPerSec = detail::read_le32(data + 24);
  header->bytesPerSec = detail::read_le32(data + 28);
  header->blockAlign = detail::read_le16(data + 32);
  header->bitsPerSample = detail::read_le16(data + 34);

  // Find "data" sub-chunk (may not be immediately after fmt if extra
  // chunks like LIST or JUNK are present). Start after fmt to avoid
  // false-matching "data" inside earlier chunk payloads.
  size_t data_offset = 0;
  size_t search_start = 12 + 8 + header->Subchunk1Size;
  for (size_t i = search_start; i + 4 < data_size; i++) {
    if (std::memcmp(data + i, "data", 4) == 0) {
      data_offset = i;
      break;
    }
  }

  if (data_offset == 0) {
    ET_LOG(
        Error,
        "WAV file structure is invalid, missing Subchunk2ID 'data' field.");
    return nullptr;
  }

  // Validate that we can safely read the Subchunk2Size (4 bytes at
  // data_offset + 4) and that the data starts at data_offset + 8
  if (data_offset + 8 > data_size) {
    ET_LOG(
        Error,
        "WAV file structure is invalid: data chunk header extends beyond file bounds (offset %zu, file size %zu)",
        data_offset,
        data_size);
    return nullptr;
  }

  // Use memcpy instead of reinterpret_cast to avoid strict aliasing violation
  header->Subchunk2Size = detail::read_le32(data + data_offset + 4);
  header->dataOffset = static_cast<uint32_t>(data_offset + 8);

  ET_LOG(Info, "WAV header detected, getting raw audio data.");
  ET_LOG(
      Info,
      "RIFF Header: %c%c%c%c",
      header->RIFF[0],
      header->RIFF[1],
      header->RIFF[2],
      header->RIFF[3]);
  ET_LOG(Info, "Chunk Size: %d", header->ChunkSize);
  ET_LOG(
      Info,
      "WAVE Header: %c%c%c%c",
      header->WAVE[0],
      header->WAVE[1],
      header->WAVE[2],
      header->WAVE[3]);
  ET_LOG(
      Info,
      "Format Header: %c%c%c%c",
      header->fmt[0],
      header->fmt[1],
      header->fmt[2],
      header->fmt[3]);
  ET_LOG(Info, "Format Chunk Size: %d", header->Subchunk1Size);
  ET_LOG(Info, "Audio Format: %d", header->AudioFormat);
  ET_LOG(Info, "Number of Channels: %d", header->NumOfChan);
  ET_LOG(Info, "Sample Rate: %d", header->SamplesPerSec);
  ET_LOG(Info, "Byte Rate: %d", header->bytesPerSec);
  ET_LOG(Info, "Block Align: %d", header->blockAlign);
  ET_LOG(Info, "Bits per Sample: %d", header->bitsPerSample);
  ET_LOG(Info, "Subchunk2Size: %d", header->Subchunk2Size);

  return header;
}

/// Parse a WAV header from a file path.
inline std::unique_ptr<WavHeader> load_wav_header(const std::string& fp) {
  std::vector<char> buffer = detail::read_file(fp);
  if (buffer.empty()) {
    ET_CHECK_MSG(false, "Failed to open WAV file: %s", fp.c_str());
  }
  return load_wav_header(buffer.data(), buffer.size());
}

/// Load and decode audio samples from a WAV file, returning normalized floats.
/// Reads the file only once.
inline std::vector<float> load_wav_audio_data(const std::string& fp) {
  std::vector<char> buffer = detail::read_file(fp);
  if (buffer.empty()) {
    ET_CHECK_MSG(false, "Failed to open WAV file: %s", fp.c_str());
  }

  auto header = load_wav_header(buffer.data(), buffer.size());

  if (header == nullptr) {
    ET_CHECK_MSG(false, "WAV header not detected in file: %s", fp.c_str());
  }

  const char* data = buffer.data();
  size_t data_offset = header->dataOffset;
  size_t data_size = header->Subchunk2Size;
  int bits_per_sample = header->bitsPerSample;
  int audio_format = header->AudioFormat;

  // Validate that the claimed data size does not exceed the buffer bounds.
  if (data_offset > buffer.size()) {
    ET_CHECK_MSG(
        false,
        "Invalid WAV file: data offset (%zu) exceeds file size (%zu)",
        data_offset,
        buffer.size());
  }
  if (data_size > buffer.size() - data_offset) {
    ET_CHECK_MSG(
        false,
        "Invalid WAV file: claimed data size (%zu) exceeds available data (%zu bytes from offset %zu)",
        data_size,
        buffer.size() - data_offset,
        data_offset);
  }

  if (audio_format != kWavFormatPcm && audio_format != kWavFormatIeeeFloat) {
    ET_CHECK_MSG(
        false,
        "Unsupported audio format: 0x%04X. Only PCM (0x%04X) and IEEE Float (0x%04X) are supported.",
        audio_format,
        kWavFormatPcm,
        kWavFormatIeeeFloat);
  }

  std::vector<float> audio_data;

  if (bits_per_sample == 32) {
    size_t num_samples = data_size / 4;

    if (audio_format == kWavFormatIeeeFloat) {
      // IEEE float format - memcpy avoids strict aliasing / alignment issues
      audio_data.resize(num_samples);
      std::memcpy(
          audio_data.data(), data + data_offset, num_samples * sizeof(float));
    } else {
      // PCM integer format - normalize from int32
      audio_data.resize(num_samples);
      for (size_t i = 0; i < num_samples; ++i) {
        int32_t sample;
        std::memcpy(
            &sample, data + data_offset + i * sizeof(int32_t), sizeof(int32_t));
        audio_data[i] =
            static_cast<float>(static_cast<double>(sample) * kOneOverIntMax);
      }
    }
  } else if (bits_per_sample == 16) {
    size_t num_samples = data_size / 2;
    audio_data.resize(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
      int16_t sample;
      std::memcpy(
          &sample, data + data_offset + i * sizeof(int16_t), sizeof(int16_t));
      audio_data[i] =
          static_cast<float>(static_cast<double>(sample) * kOneOverShortMax);
    }
  } else {
    ET_CHECK_MSG(
        false,
        "Unsupported bits per sample: %d. Only support 32 and 16.",
        bits_per_sample);
  }

  ET_LOG(
      Info,
      "Loaded %zu audio samples from WAV file: %s",
      audio_data.size(),
      fp.c_str());

  return audio_data;
}

} // namespace executorch::extension::llm
