/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/testing_util/temp_file.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using executorch::extension::llm::kOneOverIntMax;
using executorch::extension::llm::kOneOverShortMax;
using executorch::extension::llm::kWavFormatIeeeFloat;
using executorch::extension::llm::load_wav_audio_data;
using executorch::extension::llm::load_wav_header;
using executorch::extension::llm::WavHeader;
using executorch::extension::testing::TempFile;

namespace {

// WAV file format constants
constexpr uint32_t kWavHeaderSizeBeforeData = 36;
constexpr uint32_t kWavHeaderSizeWithData = 44;

// Test fixture to ensure PAL initialization
class WavLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure PAL is initialized before tests run
    executorch::runtime::runtime_init();
  }
};

void append_bytes(std::vector<uint8_t>& out, const char* literal) {
  out.insert(out.end(), literal, literal + 4);
}

void append_le16(std::vector<uint8_t>& out, uint16_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xFF));
  out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
}

void append_le32(std::vector<uint8_t>& out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xFF));
  out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
  out.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
  out.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
}

void append_float(std::vector<uint8_t>& out, float value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  for (size_t i = 0; i < sizeof(float); ++i) {
    out.push_back(bytes[i]);
  }
}

std::vector<uint8_t> make_pcm_wav_bytes(
    int bits_per_sample,
    const std::vector<int32_t>& samples,
    uint16_t num_channels = 1,
    uint32_t sample_rate = 16000) {
  const auto bytes_per_sample = static_cast<size_t>(bits_per_sample / 8);
  const auto subchunk2_size =
      static_cast<uint32_t>(samples.size() * bytes_per_sample);
  const uint32_t byte_rate = sample_rate * num_channels * bytes_per_sample;
  const uint16_t block_align = num_channels * bytes_per_sample;
  const auto chunk_size = kWavHeaderSizeBeforeData + subchunk2_size;

  std::vector<uint8_t> bytes;
  bytes.reserve(kWavHeaderSizeWithData + subchunk2_size);

  append_bytes(bytes, "RIFF");
  append_le32(bytes, chunk_size);
  append_bytes(bytes, "WAVE");
  append_bytes(bytes, "fmt ");
  append_le32(bytes, 16); // PCM
  append_le16(bytes, 1); // AudioFormat PCM
  append_le16(bytes, num_channels);
  append_le32(bytes, sample_rate);
  append_le32(bytes, byte_rate);
  append_le16(bytes, block_align);
  append_le16(bytes, static_cast<uint16_t>(bits_per_sample));
  append_bytes(bytes, "data");
  append_le32(bytes, subchunk2_size);

  for (int32_t sample : samples) {
    const uint32_t encoded =
        static_cast<uint32_t>(static_cast<int32_t>(sample));
    for (size_t byte_idx = 0; byte_idx < bytes_per_sample; ++byte_idx) {
      bytes.push_back(static_cast<uint8_t>((encoded >> (8 * byte_idx)) & 0xFF));
    }
  }

  return bytes;
}

std::vector<uint8_t> make_float_wav_bytes(
    const std::vector<float>& samples,
    uint16_t num_channels = 1,
    uint32_t sample_rate = 16000) {
  const auto bytes_per_sample = sizeof(float);
  const auto subchunk2_size =
      static_cast<uint32_t>(samples.size() * bytes_per_sample);
  const uint32_t byte_rate = sample_rate * num_channels * bytes_per_sample;
  const uint16_t block_align = num_channels * bytes_per_sample;
  const auto chunk_size = kWavHeaderSizeBeforeData + subchunk2_size;

  std::vector<uint8_t> bytes;
  bytes.reserve(kWavHeaderSizeWithData + subchunk2_size);

  append_bytes(bytes, "RIFF");
  append_le32(bytes, chunk_size);
  append_bytes(bytes, "WAVE");
  append_bytes(bytes, "fmt ");
  append_le32(bytes, 16);
  append_le16(bytes, 3); // AudioFormat IEEE Float
  append_le16(bytes, num_channels);
  append_le32(bytes, sample_rate);
  append_le32(bytes, byte_rate);
  append_le16(bytes, block_align);
  append_le16(bytes, 32); // bits per sample
  append_bytes(bytes, "data");
  append_le32(bytes, subchunk2_size);

  for (float sample : samples) {
    append_float(bytes, sample);
  }

  return bytes;
}

std::vector<uint8_t> make_wav_bytes_with_format(
    uint16_t audio_format,
    int bits_per_sample,
    const std::vector<uint8_t>& sample_data,
    uint16_t num_channels = 1,
    uint32_t sample_rate = 16000) {
  const auto bytes_per_sample = static_cast<size_t>(bits_per_sample / 8);
  const auto subchunk2_size = static_cast<uint32_t>(sample_data.size());
  const uint32_t byte_rate = sample_rate * num_channels * bytes_per_sample;
  const uint16_t block_align = num_channels * bytes_per_sample;
  const auto chunk_size = kWavHeaderSizeBeforeData + subchunk2_size;

  std::vector<uint8_t> bytes;
  bytes.reserve(kWavHeaderSizeWithData + subchunk2_size);

  append_bytes(bytes, "RIFF");
  append_le32(bytes, chunk_size);
  append_bytes(bytes, "WAVE");
  append_bytes(bytes, "fmt ");
  append_le32(bytes, 16);
  append_le16(bytes, audio_format);
  append_le16(bytes, num_channels);
  append_le32(bytes, sample_rate);
  append_le32(bytes, byte_rate);
  append_le16(bytes, block_align);
  append_le16(bytes, static_cast<uint16_t>(bits_per_sample));
  append_bytes(bytes, "data");
  append_le32(bytes, subchunk2_size);

  bytes.insert(bytes.end(), sample_data.begin(), sample_data.end());

  return bytes;
}

} // namespace

TEST_F(WavLoaderTest, LoadHeaderParsesPcmMetadata) {
  const std::vector<uint8_t> wav_bytes =
      make_pcm_wav_bytes(16, {0, 32767, -32768});
  TempFile file(wav_bytes.data(), wav_bytes.size());

  std::unique_ptr<WavHeader> header = load_wav_header(file.path());
  ASSERT_NE(header, nullptr);

  EXPECT_EQ(header->AudioFormat, 1);
  EXPECT_EQ(header->NumOfChan, 1);
  EXPECT_EQ(header->SamplesPerSec, 16000);
  EXPECT_EQ(header->bitsPerSample, 16);
  EXPECT_EQ(header->blockAlign, 2);
  EXPECT_EQ(header->bytesPerSec, 32000);
  EXPECT_EQ(header->dataOffset, 44);
  EXPECT_EQ(header->Subchunk2Size, 6);
}

TEST_F(WavLoaderTest, LoadAudioData16BitNormalizesSamples) {
  const std::vector<int32_t> samples = {0, 32767, -32768};
  const std::vector<uint8_t> wav_bytes = make_pcm_wav_bytes(16, samples);
  TempFile file(wav_bytes.data(), wav_bytes.size());

  std::vector<float> audio = load_wav_audio_data(file.path());
  ASSERT_EQ(audio.size(), samples.size());

  EXPECT_NEAR(audio[0], 0.0f, 1e-6f);
  EXPECT_NEAR(audio[1], 32767.0f * kOneOverShortMax, 1e-6f);
  EXPECT_NEAR(audio[2], -32768.0f * kOneOverShortMax, 1e-6f);
}

TEST_F(WavLoaderTest, LoadAudioData32BitNormalizesSamples) {
  const std::vector<int32_t> samples = {
      0,
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::min()};
  const std::vector<uint8_t> wav_bytes = make_pcm_wav_bytes(32, samples);
  TempFile file(wav_bytes.data(), wav_bytes.size());

  std::vector<float> audio = load_wav_audio_data(file.path());
  ASSERT_EQ(audio.size(), samples.size());

  EXPECT_NEAR(audio[0], 0.0f, 1e-8f);
  EXPECT_NEAR(
      audio[1],
      static_cast<float>(static_cast<double>(samples[1]) * kOneOverIntMax),
      1e-6f);
  EXPECT_NEAR(
      audio[2],
      static_cast<float>(static_cast<double>(samples[2]) * kOneOverIntMax),
      1e-6f);
}

TEST_F(WavLoaderTest, LoadHeaderReturnsNullWhenMagicMissing) {
  const std::string bogus_contents = "not a wav file";
  TempFile file(bogus_contents);

  std::unique_ptr<WavHeader> header = load_wav_header(file.path());
  EXPECT_EQ(header, nullptr);
}

TEST_F(WavLoaderTest, LoadAudioDataFloatFormatReadsDirectly) {
  const std::vector<float> samples = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f};
  const std::vector<uint8_t> wav_bytes = make_float_wav_bytes(samples);
  TempFile file(wav_bytes.data(), wav_bytes.size());

  std::unique_ptr<WavHeader> header = load_wav_header(file.path());
  ASSERT_NE(header, nullptr);
  EXPECT_EQ(header->AudioFormat, kWavFormatIeeeFloat);
  EXPECT_EQ(header->bitsPerSample, 32);

  std::vector<float> audio = load_wav_audio_data(file.path());
  ASSERT_EQ(audio.size(), samples.size());

  for (size_t i = 0; i < samples.size(); ++i) {
    EXPECT_FLOAT_EQ(audio[i], samples[i]);
  }
}

TEST_F(WavLoaderTest, LoadAudioDataRejectsUnsupportedFormat) {
  const std::vector<uint8_t> sample_data = {0, 0, 0, 0};
  const std::vector<uint8_t> wav_bytes =
      make_wav_bytes_with_format(0x0006, 16, sample_data);
  TempFile file(wav_bytes.data(), wav_bytes.size());

  EXPECT_DEATH(
      { load_wav_audio_data(file.path()); }, "Unsupported audio format");
}
