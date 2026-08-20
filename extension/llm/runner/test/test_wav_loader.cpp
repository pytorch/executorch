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
#include <executorch/test/utils/DeathTest.h>

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
  ET_EXPECT_DEATH(
      { load_wav_audio_data(file.path()); }, "Unsupported audio format");
}

// Helper function to create a malformed WAV file with inflated Subchunk2Size
std::vector<uint8_t> make_malformed_wav_with_inflated_size(
    uint32_t claimed_data_size,
    uint32_t actual_data_size,
    uint16_t num_channels = 1,
    uint32_t sample_rate = 16000) {
  const size_t bytes_per_sample = sizeof(float);
  const uint32_t byte_rate = sample_rate * num_channels * bytes_per_sample;
  const uint16_t block_align = num_channels * bytes_per_sample;
  // Use claimed size in ChunkSize to match what a malicious file would have
  const auto chunk_size = kWavHeaderSizeBeforeData + claimed_data_size;

  std::vector<uint8_t> bytes;
  bytes.reserve(kWavHeaderSizeWithData + actual_data_size);

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
  append_le32(bytes, claimed_data_size); // Inflated size in header

  // Only write actual_data_size bytes of audio data
  for (uint32_t i = 0; i < actual_data_size; ++i) {
    bytes.push_back(0);
  }

  return bytes;
}

// Security test: Verify that WAV files with inflated Subchunk2Size are rejected
// This prevents heap-buffer-overflow from malicious files claiming more data
// than actually present.
TEST_F(WavLoaderTest, LoadAudioDataRejectsInflatedSubchunk2Size) {
  // Create a 48-byte WAV file that claims to contain 1 MB of audio data
  // (similar to the proof-of-concept in the security report)
  const uint32_t claimed_size = 1024 * 1024; // 1 MB claimed
  const uint32_t actual_size = 4; // Only 4 bytes of actual data
  const std::vector<uint8_t> wav_bytes =
      make_malformed_wav_with_inflated_size(claimed_size, actual_size);
  TempFile file(wav_bytes.data(), wav_bytes.size());

  ET_EXPECT_DEATH(
      { load_wav_audio_data(file.path()); }, "claimed data size.*exceeds");
}

// Security test: Verify that WAV files with data chunk at end of file and
// inflated data size are rejected
TEST_F(WavLoaderTest, LoadAudioDataRejectsDataOffsetBeyondBounds) {
  // Create a minimal WAV header but truncate it so data offset points beyond
  // file
  std::vector<uint8_t> bytes;
  append_bytes(bytes, "RIFF");
  append_le32(bytes, 100); // ChunkSize
  append_bytes(bytes, "WAVE");
  append_bytes(bytes, "fmt ");
  append_le32(bytes, 16);
  append_le16(bytes, 3); // IEEE Float
  append_le16(bytes, 1); // channels
  append_le32(bytes, 16000); // sample rate
  append_le32(bytes, 64000); // byte rate
  append_le16(bytes, 4); // block align
  append_le16(bytes, 32); // bits per sample
  append_bytes(bytes, "data");
  append_le32(bytes, 1000); // Claim 1000 bytes but provide none
  // No actual audio data follows - file ends here

  TempFile file(bytes.data(), bytes.size());

  ET_EXPECT_DEATH(
      { load_wav_audio_data(file.path()); }, "claimed data size.*exceeds");
}

// Security test: Verify boundary condition where claimed size exactly matches
// available data
TEST_F(WavLoaderTest, LoadAudioDataAcceptsExactSizeMatch) {
  // Create a valid WAV file where Subchunk2Size exactly matches the data
  const std::vector<float> samples = {0.5f, -0.5f};
  const std::vector<uint8_t> wav_bytes = make_float_wav_bytes(samples);
  TempFile file(wav_bytes.data(), wav_bytes.size());

  // This should succeed without any issues
  std::vector<float> audio = load_wav_audio_data(file.path());
  ASSERT_EQ(audio.size(), samples.size());
  EXPECT_FLOAT_EQ(audio[0], 0.5f);
  EXPECT_FLOAT_EQ(audio[1], -0.5f);
}

// === Additional tests for strict-aliasing fix (T262564446) ===

// Test that the buffer-based overload of load_wav_header works correctly
TEST_F(WavLoaderTest, LoadHeaderFromBufferParsesCorrectly) {
  const std::vector<uint8_t> wav_bytes =
      make_pcm_wav_bytes(32, {100, -100, 0, INT32_MAX});
  TempFile file(wav_bytes.data(), wav_bytes.size());

  // Load via file path (original API)
  std::unique_ptr<WavHeader> header_from_file = load_wav_header(file.path());

  // Load via buffer (new overload)
  std::unique_ptr<WavHeader> header_from_buf = load_wav_header(
      reinterpret_cast<const char*>(wav_bytes.data()), wav_bytes.size());

  ASSERT_NE(header_from_file, nullptr);
  ASSERT_NE(header_from_buf, nullptr);

  // Both should produce identical results
  EXPECT_EQ(header_from_file->AudioFormat, header_from_buf->AudioFormat);
  EXPECT_EQ(header_from_file->NumOfChan, header_from_buf->NumOfChan);
  EXPECT_EQ(header_from_file->SamplesPerSec, header_from_buf->SamplesPerSec);
  EXPECT_EQ(header_from_file->bitsPerSample, header_from_buf->bitsPerSample);
  EXPECT_EQ(header_from_file->blockAlign, header_from_buf->blockAlign);
  EXPECT_EQ(header_from_file->bytesPerSec, header_from_buf->bytesPerSec);
  EXPECT_EQ(header_from_file->dataOffset, header_from_buf->dataOffset);
  EXPECT_EQ(header_from_file->Subchunk2Size, header_from_buf->Subchunk2Size);
  EXPECT_EQ(header_from_file->ChunkSize, header_from_buf->ChunkSize);
  EXPECT_EQ(header_from_file->Subchunk1Size, header_from_buf->Subchunk1Size);
}

// Test that unaligned buffers are handled correctly (the core strict-aliasing
// scenario: data pointer not aligned to uint32_t boundary).
TEST_F(WavLoaderTest, LoadHeaderHandlesUnalignedBuffer) {
  const std::vector<uint8_t> wav_bytes =
      make_pcm_wav_bytes(16, {1000, -1000, 0});

  // Create a buffer with 1-byte offset to guarantee misalignment
  std::vector<char> unaligned_buf(wav_bytes.size() + 1);
  std::memcpy(unaligned_buf.data() + 1, wav_bytes.data(), wav_bytes.size());

  // Parse from the unaligned pointer - this would crash with reinterpret_cast
  // on alignment-sensitive platforms, but works fine with memcpy.
  const char* unaligned_data = unaligned_buf.data() + 1;
  std::unique_ptr<WavHeader> header =
      load_wav_header(unaligned_data, wav_bytes.size());

  ASSERT_NE(header, nullptr);
  EXPECT_EQ(header->AudioFormat, 1); // PCM
  EXPECT_EQ(header->NumOfChan, 1);
  EXPECT_EQ(header->SamplesPerSec, 16000);
  EXPECT_EQ(header->bitsPerSample, 16);
  EXPECT_EQ(header->Subchunk2Size, 6); // 3 samples * 2 bytes
}

// Test WAV file with extra chunks between fmt and data (e.g., LIST chunk).
// The old code would mis-parse this because the naive memcpy of 44 bytes
// would place wrong data into the struct fields.
TEST_F(WavLoaderTest, LoadHeaderWithExtraChunkBetweenFmtAndData) {
  // Build a WAV with a "LIST" chunk inserted between fmt and data
  std::vector<uint8_t> bytes;

  // RIFF header
  append_bytes(bytes, "RIFF");
  uint32_t placeholder_chunk_size = 0; // will fill later
  size_t chunk_size_offset = bytes.size();
  append_le32(bytes, placeholder_chunk_size);
  append_bytes(bytes, "WAVE");

  // fmt chunk
  append_bytes(bytes, "fmt ");
  append_le32(bytes, 16); // Subchunk1Size
  append_le16(bytes, 1); // PCM
  append_le16(bytes, 2); // stereo
  append_le32(bytes, 44100); // sample rate
  append_le32(bytes, 44100 * 2 * 2); // byte rate
  append_le16(bytes, 4); // block align
  append_le16(bytes, 16); // bits per sample

  // Extra LIST chunk (commonly found in WAV files from editors)
  append_bytes(bytes, "LIST");
  append_le32(bytes, 26); // chunk size
  // 26 bytes of arbitrary metadata
  const char* list_data = "INFOIART\x0e\x00\x00\x00Test Artist\x00\x00";
  bytes.insert(bytes.end(), list_data, list_data + 26);

  // data chunk
  append_bytes(bytes, "data");
  uint32_t data_size = 8; // 2 stereo samples * 2 bytes each
  append_le32(bytes, data_size);
  // Sample data: L=1000, R=-1000, L=500, R=-500
  append_le16(bytes, 1000);
  append_le16(bytes, static_cast<uint16_t>(-1000));
  append_le16(bytes, 500);
  append_le16(bytes, static_cast<uint16_t>(-500));

  // Fix up ChunkSize
  uint32_t total_chunk_size = static_cast<uint32_t>(bytes.size() - 8);
  std::memcpy(bytes.data() + chunk_size_offset, &total_chunk_size, 4);

  TempFile file(bytes.data(), bytes.size());

  std::unique_ptr<WavHeader> header = load_wav_header(file.path());
  ASSERT_NE(header, nullptr);

  EXPECT_EQ(header->AudioFormat, 1);
  EXPECT_EQ(header->NumOfChan, 2);
  EXPECT_EQ(header->SamplesPerSec, 44100);
  EXPECT_EQ(header->bitsPerSample, 16);
  EXPECT_EQ(header->blockAlign, 4);
  EXPECT_EQ(header->Subchunk2Size, data_size);
  // dataOffset should point past the LIST chunk, to the actual audio data
  EXPECT_GT(header->dataOffset, 44u);
}

// Test single-read optimization: verify load_wav_audio_data still works
// correctly (this exercises the refactored single-pass code path).
TEST_F(WavLoaderTest, LoadAudioDataSinglePassReading) {
  // Use a larger sample set to ensure the full pipeline works
  std::vector<float> samples(1000);
  for (int i = 0; i < 1000; ++i) {
    samples[i] = static_cast<float>(i) / 1000.0f - 0.5f;
  }
  const std::vector<uint8_t> wav_bytes = make_float_wav_bytes(samples);
  TempFile file(wav_bytes.data(), wav_bytes.size());

  std::vector<float> audio = load_wav_audio_data(file.path());
  ASSERT_EQ(audio.size(), samples.size());

  for (size_t i = 0; i < samples.size(); ++i) {
    EXPECT_FLOAT_EQ(audio[i], samples[i]);
  }
}

// Test that 32-bit PCM with memcpy produces correct results at boundaries
TEST_F(WavLoaderTest, LoadAudioData32BitPcmBoundaryValues) {
  const std::vector<int32_t> samples = {
      0,
      1,
      -1,
      32767,
      -32768,
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::max() / 2};
  const std::vector<uint8_t> wav_bytes = make_pcm_wav_bytes(32, samples);
  TempFile file(wav_bytes.data(), wav_bytes.size());

  std::vector<float> audio = load_wav_audio_data(file.path());
  ASSERT_EQ(audio.size(), samples.size());

  for (size_t i = 0; i < samples.size(); ++i) {
    float expected =
        static_cast<float>(static_cast<double>(samples[i]) * kOneOverIntMax);
    EXPECT_NEAR(audio[i], expected, 1e-6f)
        << "Mismatch at sample " << i << " (value=" << samples[i] << ")";
  }
}
