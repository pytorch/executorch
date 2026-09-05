/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/arm/runtime/VelaBinStream.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string_view>
#include <vector>

#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/tensor_layout.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using executorch::backends::arm::kVelaExternalBlockReference;
using executorch::backends::arm::vela_bin_read;
using executorch::backends::arm::VelaHandles;
using executorch::backends::arm::VelaIO;
using executorch::backends::arm::VelaIOs;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::NamedDataMap;
using executorch::runtime::Result;
using executorch::runtime::TensorLayout;

namespace {

constexpr std::string_view kExternalKey =
    "g123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
constexpr std::string_view kSecondExternalKey =
    "h123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

void append_u32(std::vector<uint8_t>& output, uint32_t value) {
  for (size_t i = 0; i < sizeof(value); ++i) {
    output.push_back((value >> (8 * i)) & 0xff);
  }
}

void append_fixed_string(
    std::vector<uint8_t>& output,
    std::string_view value,
    size_t size) {
  output.insert(output.end(), value.begin(), value.end());
  output.resize(output.size() + size - value.size(), 0);
}

void append_block(
    std::vector<uint8_t>& output,
    std::string_view name,
    const std::vector<uint8_t>& payload = {},
    uint8_t external = 0,
    uint8_t reserved = 0) {
  append_fixed_string(output, name, 16);
  append_u32(output, static_cast<uint32_t>(payload.size()));
  output.push_back(external);
  output.resize(output.size() + 11, reserved);
  output.insert(output.end(), payload.begin(), payload.end());
  output.resize((output.size() + 15) & ~size_t{15}, 0);
}

std::vector<uint8_t> stream_with_external_key(
    std::string_view block_name,
    std::string_view key = kExternalKey) {
  std::vector<uint8_t> output;
  append_block(output, "vela_bin_stream");
  append_block(
      output,
      block_name,
      std::vector<uint8_t>(key.begin(), key.end()),
      kVelaExternalBlockReference);
  append_block(output, "vela_end_stream");
  return output;
}

class FakeNamedDataMap final : public NamedDataMap {
 public:
  struct Entry {
    std::string_view key;
    const void* data;
    size_t size;
  };

  FakeNamedDataMap(std::string_view key, const void* data, size_t size)
      : entries_{{key, data, size}} {}

  explicit FakeNamedDataMap(std::initializer_list<Entry> entries)
      : entries_(entries) {}

  Result<const TensorLayout> get_tensor_layout(
      std::string_view) const override {
    return Error::NotFound;
  }

  Result<FreeableBuffer> get_data(std::string_view key) const override {
    const auto entry = std::find_if(
        entries_.begin(), entries_.end(), [key](const Entry& candidate) {
          return key == candidate.key;
        });
    if (entry == entries_.end()) {
      return Error::NotFound;
    }
    return FreeableBuffer(entry->data, entry->size, nullptr);
  }

  Error load_data_into(std::string_view, void*, size_t) const override {
    return Error::NotImplemented;
  }

  Result<uint32_t> get_num_keys() const override {
    return static_cast<uint32_t>(entries_.size());
  }

  Result<const char*> get_key(uint32_t index) const override {
    return index < entries_.size()
        ? Result<const char*>(entries_[index].key.data())
        : Result<const char*>(Error::NotFound);
  }

 private:
  std::vector<Entry> entries_;
};

} // namespace

TEST(VelaExternalBlocksTest, ReadsInlineAndExternalBlocksInOnePass) {
  alignas(16) const std::array<uint8_t, 16> command{
      'C', 'O', 'P', '1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<uint8_t> weights{1, 2, 3, 4};
  std::vector<uint8_t> stream;
  append_block(stream, "vela_bin_stream");
  append_block(
      stream,
      "cmd_data",
      std::vector<uint8_t>(kExternalKey.begin(), kExternalKey.end()),
      kVelaExternalBlockReference);
  append_block(stream, "weight_data", weights);
  append_block(stream, "vela_end_stream");
  FakeNamedDataMap data_map(kExternalKey, command.data(), command.size());
  VelaHandles handles{};

  ASSERT_EQ(
      vela_bin_read(
          reinterpret_cast<const char*>(stream.data()),
          stream.size(),
          &data_map,
          &handles),
      Error::Ok);
  EXPECT_EQ(handles.cmd_data, reinterpret_cast<const char*>(command.data()));
  EXPECT_EQ(handles.cmd_data_size, command.size());
  ASSERT_NE(handles.weight_data, nullptr);
  EXPECT_EQ(handles.weight_data_size, weights.size());
  EXPECT_EQ(
      std::memcmp(handles.weight_data, weights.data(), weights.size()), 0);
}

TEST(VelaExternalBlocksTest, RejectsInvalidExternalCommandStream) {
  alignas(16) const std::array<uint8_t, 4> opaque_command{'B', 'A', 'D', '1'};
  const auto stream = stream_with_external_key("cmd_data");
  FakeNamedDataMap data_map(
      kExternalKey, opaque_command.data(), opaque_command.size());
  VelaHandles handles{};

  EXPECT_EQ(
      vela_bin_read(
          reinterpret_cast<const char*>(stream.data()),
          stream.size(),
          &data_map,
          &handles),
      Error::InvalidProgram);
}

TEST(VelaExternalBlocksTest, RejectsExternalKeyWithWrongSize) {
  alignas(16) const std::array<uint8_t, 4> command{'C', 'O', 'P', '1'};
  const auto stream = stream_with_external_key("cmd_data", "not-a-hash");
  FakeNamedDataMap data_map("not-a-hash", command.data(), command.size());
  VelaHandles handles{};

  EXPECT_EQ(
      vela_bin_read(
          reinterpret_cast<const char*>(stream.data()),
          stream.size(),
          &data_map,
          &handles),
      Error::InvalidProgram);
}

TEST(VelaExternalBlocksTest, ReadsInlineBlocksWithoutNamedData) {
  const std::vector<uint8_t> command{
      'C', 'O', 'P', '1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint8_t> stream;
  append_block(stream, "vela_bin_stream");
  append_block(stream, "cmd_data", command);
  append_block(stream, "vela_end_stream");
  VelaHandles handles{};

  ASSERT_EQ(
      vela_bin_read(
          reinterpret_cast<const char*>(stream.data()),
          stream.size(),
          nullptr,
          &handles),
      Error::Ok);
  ASSERT_NE(handles.cmd_data, nullptr);
  EXPECT_EQ(handles.cmd_data_size, command.size());
  EXPECT_EQ(std::memcmp(handles.cmd_data, command.data(), command.size()), 0);
}

TEST(VelaExternalBlocksTest, ResolvesEveryExternalBlockType) {
  alignas(16) const std::array<uint8_t, 16> weight_data{
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  alignas(16) const std::array<uint8_t, 4> scratch_size{32, 0, 0, 0};
  alignas(16) const std::array<uint8_t, 4> empty_ios{0, 0, 0, 0};
  struct ExternalBlockCase {
    std::string_view name;
    const uint8_t* data;
    size_t size;
  };
  const std::array<ExternalBlockCase, 4> cases{{
      {"weight_data", weight_data.data(), weight_data.size()},
      {"scratch_size", scratch_size.data(), scratch_size.size()},
      {"inputs", empty_ios.data(), empty_ios.size()},
      {"outputs", empty_ios.data(), empty_ios.size()},
  }};

  for (const auto& test_case : cases) {
    const auto stream = stream_with_external_key(test_case.name);
    FakeNamedDataMap data_map(kExternalKey, test_case.data, test_case.size);
    VelaHandles handles{};
    ASSERT_EQ(
        vela_bin_read(
            reinterpret_cast<const char*>(stream.data()),
            stream.size(),
            &data_map,
            &handles),
        Error::Ok)
        << test_case.name;
    if (test_case.name == "weight_data") {
      EXPECT_EQ(
          handles.weight_data, reinterpret_cast<const char*>(test_case.data));
      EXPECT_EQ(handles.weight_data_size, test_case.size);
    } else if (test_case.name == "scratch_size") {
      EXPECT_EQ(handles.scratch_data_size, 32);
    } else if (test_case.name == "inputs") {
      ASSERT_NE(handles.inputs, nullptr);
      EXPECT_EQ(handles.inputs->count, 0);
    } else {
      ASSERT_NE(handles.outputs, nullptr);
      EXPECT_EQ(handles.outputs->count, 0);
    }
  }
}

TEST(VelaExternalBlocksTest, ResolvesMultipleExternalBlocksIndependently) {
  // The payload must be large enough to hold `count` VelaIO entries, otherwise
  // vela_bin_read rejects it. count is the first int of the buffer.
  alignas(16) std::array<uint8_t, sizeof(VelaIOs) + sizeof(VelaIO)>
      inputs_data{};
  inputs_data[0] = 1; // count = 1
  alignas(16) std::array<uint8_t, sizeof(VelaIOs) + 2 * sizeof(VelaIO)>
      outputs_data{};
  outputs_data[0] = 2; // count = 2
  std::vector<uint8_t> stream;
  append_block(stream, "vela_bin_stream");
  append_block(
      stream,
      "inputs",
      std::vector<uint8_t>(kExternalKey.begin(), kExternalKey.end()),
      kVelaExternalBlockReference);
  append_block(
      stream,
      "outputs",
      std::vector<uint8_t>(
          kSecondExternalKey.begin(), kSecondExternalKey.end()),
      kVelaExternalBlockReference);
  append_block(stream, "vela_end_stream");
  FakeNamedDataMap data_map({
      {kExternalKey, inputs_data.data(), inputs_data.size()},
      {kSecondExternalKey, outputs_data.data(), outputs_data.size()},
  });
  VelaHandles handles{};

  ASSERT_EQ(
      vela_bin_read(
          reinterpret_cast<const char*>(stream.data()),
          stream.size(),
          &data_map,
          &handles),
      Error::Ok);
  ASSERT_EQ(handles.inputs, reinterpret_cast<VelaIOs*>(inputs_data.data()));
  EXPECT_EQ(handles.inputs->count, 1);
  ASSERT_EQ(handles.outputs, reinterpret_cast<VelaIOs*>(outputs_data.data()));
  EXPECT_EQ(handles.outputs->count, 2);
}

TEST(VelaExternalBlocksTest, RejectsUnknownExternalBlock) {
  executorch::runtime::runtime_init();
  alignas(16) const std::array<uint8_t, 4> payload{1, 2, 3, 4};
  const auto stream = stream_with_external_key("unsupported");
  FakeNamedDataMap data_map(kExternalKey, payload.data(), payload.size());
  VelaHandles handles{};

  EXPECT_EQ(
      vela_bin_read(
          reinterpret_cast<const char*>(stream.data()),
          stream.size(),
          &data_map,
          &handles),
      Error::InvalidProgram);
}

// A block header that is truncated by the end of the buffer must be rejected
// instead of being read out of bounds.
TEST(VelaExternalBlocksTest, RejectsTruncatedBlockHeader) {
  std::vector<uint8_t> stream;
  append_block(stream, "vela_bin_stream");
  // Fewer than sizeof(VelaBinBlock) trailing bytes: not a full header.
  stream.insert(stream.end(), 8, 0);
  VelaHandles handles{};

  EXPECT_EQ(
      vela_bin_read(
          reinterpret_cast<const char*>(stream.data()),
          stream.size(),
          nullptr,
          &handles),
      Error::InvalidProgram);
}

// A block whose declared size field exceeds the bytes remaining in the buffer
// must be rejected before its payload (or the cursor advance) runs off the end.
TEST(VelaExternalBlocksTest, RejectsBlockSizeLargerThanBuffer) {
  std::vector<uint8_t> stream;
  append_block(stream, "vela_bin_stream");
  // Hand-write a header claiming a 4096-byte payload but supply only one byte.
  append_fixed_string(stream, "weight_data", 16);
  append_u32(stream, 4096u);
  stream.push_back(0); // external
  stream.resize(stream.size() + 11, 0); // reserved
  stream.push_back(0xAA); // single real payload byte
  VelaHandles handles{};

  EXPECT_EQ(
      vela_bin_read(
          reinterpret_cast<const char*>(stream.data()),
          stream.size(),
          nullptr,
          &handles),
      Error::InvalidProgram);
}

// An inputs/outputs block whose declared count is larger than the payload can
// hold must be rejected, so that io[i] is never read out of bounds at execute.
TEST(VelaExternalBlocksTest, RejectsIoCountExceedingPayload) {
  std::vector<uint8_t> stream;
  append_block(stream, "vela_bin_stream");
  // count = 1 but no VelaIO entry follows (only the 4-byte count field).
  append_block(stream, "inputs", std::vector<uint8_t>{1, 0, 0, 0});
  append_block(stream, "vela_end_stream");
  VelaHandles handles{};

  EXPECT_EQ(
      vela_bin_read(
          reinterpret_cast<const char*>(stream.data()),
          stream.size(),
          nullptr,
          &handles),
      Error::InvalidProgram);
}
