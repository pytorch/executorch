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
#include <stdexcept>
#include <string>
#include <vector>

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::extension::llm {

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

inline std::unique_ptr<WavHeader> load_wav_header(const std::string& fp) {
  std::ifstream file(fp, std::ios::binary);
  if (!file.is_open()) {
    ET_CHECK_MSG(false, "Failed to open WAV file: %s", fp.c_str());
  }

  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(file_size);
  file.read(buffer.data(), file_size);
  file.close();

  const char* data = buffer.data();
  size_t data_size = buffer.size();

  bool has_riff = false;
  bool has_wave = false;

  if (data_size >= 4 && std::memcmp(data, "RIFF", 4) == 0) {
    has_riff = true;
  }

  if (data_size >= 12 && std::memcmp(data + 8, "WAVE", 4) == 0) {
    has_wave = true;
  }

  bool is_wav_file = has_riff && has_wave;
  std::unique_ptr<WavHeader> header;

  if (is_wav_file) {
    header = std::make_unique<WavHeader>();
    size_t default_header_size = sizeof(WavHeader);

    size_t data_offset = 0;
    for (size_t i = 0; i + 4 < data_size; i++) {
      if (std::memcmp(data + i, "data", 4) == 0) {
        data_offset = i;
        break;
      }
    }

    if (data_size >= default_header_size) {
      std::memcpy(
          reinterpret_cast<char*>(header.get()), data, default_header_size);

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

      if (data_offset != 0) {
        header->Subchunk2Size =
            *reinterpret_cast<const int32_t*>(data + data_offset + 4);
        ET_LOG(Info, "Subchunk2Size: %d", header->Subchunk2Size);
        header->dataOffset = static_cast<uint32_t>(data_offset + 8);
      } else {
        ET_LOG(
            Error,
            "WAV file structure is invalid, missing Subchunk2ID 'data' field.");
        throw std::runtime_error("Invalid WAV file structure");
      }
    } else {
      ET_CHECK_MSG(
          false,
          "WAV header detected but file is too small to contain a complete header");
    }
  }

  return header;
}

inline std::vector<float> load_wav_audio_data(const std::string& fp) {
  std::ifstream file(fp, std::ios::binary);
  if (!file.is_open()) {
    ET_CHECK_MSG(false, "Failed to open WAV file: %s", fp.c_str());
  }

  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(file_size);
  file.read(buffer.data(), file_size);
  file.close();

  auto header = load_wav_header(fp);

  if (header.get() == nullptr) {
    ET_CHECK_MSG(false, "WAV header not detected in file: %s", fp.c_str());
  }

  const char* data = buffer.data();
  size_t data_offset = header->dataOffset;
  size_t data_size = header->Subchunk2Size;
  int bits_per_sample = header->bitsPerSample;

  std::vector<float> audio_data;

  if (bits_per_sample == 32) {
    size_t num_samples = data_size / 4;
    audio_data.resize(num_samples);
    const int32_t* input_buffer =
        reinterpret_cast<const int32_t*>(data + data_offset);

    for (size_t i = 0; i < num_samples; ++i) {
      audio_data[i] = static_cast<float>(
          static_cast<double>(input_buffer[i]) * kOneOverIntMax);
    }
  } else if (bits_per_sample == 16) {
    size_t num_samples = data_size / 2;
    audio_data.resize(num_samples);
    const int16_t* input_buffer =
        reinterpret_cast<const int16_t*>(data + data_offset);

    for (size_t i = 0; i < num_samples; ++i) {
      audio_data[i] = static_cast<float>(
          static_cast<double>(input_buffer[i]) * kOneOverShortMax);
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
