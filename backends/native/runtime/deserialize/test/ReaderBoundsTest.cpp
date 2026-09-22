// cppcheck-suppress-file syntaxError

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/DeserializeError.h>
#include <executorch/backends/native/runtime/deserialize/Json.h>
#include <executorch/backends/native/runtime/deserialize/Limits.h>
#include <executorch/backends/native/runtime/deserialize/SafeTensorsReader.h>
#include <executorch/backends/native/runtime/deserialize/ZipReader.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace ptn {
namespace {

template <typename F>
void expect_resource_limit(F&& operation, const std::string& message) {
  try {
    operation();
    FAIL() << "expected ResourceLimitError";
  } catch (const ResourceLimitError& error) {
    EXPECT_EQ(error.what(), message);
  } catch (...) {
    FAIL() << "expected ResourceLimitError";
  }
}

template <typename T>
void write_le(std::vector<uint8_t>& bytes, size_t offset, T value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

std::vector<uint8_t> make_safetensors(
    const std::string& header,
    size_t payload_size = 0) {
  std::vector<uint8_t> bytes(sizeof(uint64_t) + header.size() + payload_size);
  const uint64_t header_size = header.size();
  std::memcpy(bytes.data(), &header_size, sizeof(header_size));
  std::memcpy(bytes.data() + sizeof(header_size), header.data(), header.size());
  return bytes;
}

TEST(SafeTensorsReaderTest, Open_ValidScalar_ReportsPayloadSize) {
  const auto bytes = make_safetensors(
      R"({"weight":{"dtype":"F32","shape":[],"data_offsets":[0,4]}})",
      /*payload_size=*/4);

  const SafeTensorsReader reader = SafeTensorsReader::open(bytes);

  ASSERT_NE(reader.find("weight"), nullptr);
  EXPECT_EQ(reader.total_bytes(), 4);
}

TEST(SafeTensorsReaderTest, Open_ExcessiveDimension_ThrowsResourceLimit) {
  const auto bytes = make_safetensors(
      R"({"weight":{"dtype":"F64","shape":[9223372036854775807],"data_offsets":[0,0]}})");

  expect_resource_limit(
      [&]() { static_cast<void>(SafeTensorsReader::open(bytes)); },
      "safetensors: entry 'weight' exceeds dimension limit");
}

TEST(SafeTensorsReaderTest, Open_ExcessiveRank_ThrowsResourceLimit) {
  std::string dimensions;
  for (size_t i = 0; i <= detail::kMaxTensorRank; ++i) {
    dimensions += i == 0 ? "1" : ",1";
  }
  const auto bytes = make_safetensors(
      "{\"weight\":{\"dtype\":\"F32\",\"shape\":[" + dimensions +
      "],\"data_offsets\":[0,0]}}");

  expect_resource_limit(
      [&]() { static_cast<void>(SafeTensorsReader::open(bytes)); },
      "safetensors: entry 'weight' exceeds tensor rank limit");
}

TEST(SafeTensorsReaderTest, Open_ElementCountOverflow_ThrowsResourceLimit) {
  const auto bytes = make_safetensors(
      R"({"weight":{"dtype":"F64","shape":[2147483647,2147483647,2147483647],"data_offsets":[0,0]}})");

  expect_resource_limit(
      [&]() { static_cast<void>(SafeTensorsReader::open(bytes)); },
      "safetensors: entry 'weight' element count overflows");
}

TEST(SafeTensorsReaderTest, Open_OverlappingRanges_Throws) {
  const auto bytes = make_safetensors(
      R"({"first":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"second":{"dtype":"F32","shape":[],"data_offsets":[0,4]}})",
      /*payload_size=*/4);

  EXPECT_THROW(SafeTensorsReader::open(bytes), std::runtime_error);
}

TEST(JsonParserTest, ParseDocument_DuplicateObjectKey_Throws) {
  EXPECT_THROW(parse_json(R"({"key":1,"key":2})"), std::runtime_error);
}

std::vector<uint8_t> make_zip(
    const std::string& name,
    const std::string& payload,
    uint32_t crc = 0x352441c2,
    bool use_descriptor = false) {
  constexpr uint32_t kEocdSignature = 0x06054b50;
  constexpr uint32_t kCentralDirectorySignature = 0x02014b50;
  constexpr uint32_t kLocalHeaderSignature = 0x04034b50;
  constexpr uint32_t kDataDescriptorSignature = 0x08074b50;
  constexpr uint16_t kEntryCount = 1;

  const size_t descriptor_size = use_descriptor ? 16 : 0;
  const size_t local_size = 30 + name.size() + payload.size() + descriptor_size;
  const size_t central_size = 46 + name.size();
  std::vector<uint8_t> bytes(local_size + central_size + 22);

  write_le(bytes, /*offset=*/0, kLocalHeaderSignature);
  write_le<uint16_t>(bytes, /*offset=*/4, /*value=*/20);
  write_le<uint16_t>(bytes, /*offset=*/6, use_descriptor ? 8 : 0);
  write_le(bytes, /*offset=*/14, use_descriptor ? 0 : crc);
  write_le<uint32_t>(
      bytes,
      /*offset=*/18,
      use_descriptor ? 0 : static_cast<uint32_t>(payload.size()));
  write_le<uint32_t>(
      bytes,
      /*offset=*/22,
      use_descriptor ? 0 : static_cast<uint32_t>(payload.size()));
  write_le<uint16_t>(bytes, /*offset=*/26, static_cast<uint16_t>(name.size()));
  std::memcpy(bytes.data() + 30, name.data(), name.size());
  std::memcpy(bytes.data() + 30 + name.size(), payload.data(), payload.size());
  if (use_descriptor) {
    const size_t descriptor = 30 + name.size() + payload.size();
    write_le(bytes, descriptor, kDataDescriptorSignature);
    write_le(bytes, descriptor + 4, crc);
    write_le<uint32_t>(
        bytes, descriptor + 8, static_cast<uint32_t>(payload.size()));
    write_le<uint32_t>(
        bytes, descriptor + 12, static_cast<uint32_t>(payload.size()));
  }

  const size_t central = local_size;
  write_le(bytes, central, kCentralDirectorySignature);
  write_le<uint16_t>(bytes, central + 4, /*value=*/20);
  write_le<uint16_t>(bytes, central + 6, /*value=*/20);
  write_le<uint16_t>(bytes, central + 8, use_descriptor ? 8 : 0);
  write_le(bytes, central + 16, crc);
  write_le<uint32_t>(
      bytes, central + 20, static_cast<uint32_t>(payload.size()));
  write_le<uint32_t>(
      bytes, central + 24, static_cast<uint32_t>(payload.size()));
  write_le<uint16_t>(bytes, central + 28, static_cast<uint16_t>(name.size()));
  std::memcpy(bytes.data() + central + 46, name.data(), name.size());

  const size_t eocd = central + central_size;
  write_le(bytes, eocd, kEocdSignature);
  write_le(bytes, eocd + 8, kEntryCount);
  write_le(bytes, eocd + 10, kEntryCount);
  write_le<uint32_t>(bytes, eocd + 12, static_cast<uint32_t>(central_size));
  write_le<uint32_t>(bytes, eocd + 16, static_cast<uint32_t>(central));
  return bytes;
}

TEST(ZipReaderTest, Open_ValidStoredMember_ReturnsPayload) {
  const auto bytes = make_zip("member", "abc");

  const ZipReader reader = ZipReader::open(bytes);

  EXPECT_EQ(reader.read("member"), (std::vector<uint8_t>{'a', 'b', 'c'}));
}

TEST(ZipReaderTest, Open_ValidDataDescriptor_ReturnsPayload) {
  const auto bytes =
      make_zip("member", "abc", /*crc=*/0x352441c2, /*use_descriptor=*/true);

  const ZipReader reader = ZipReader::open(bytes);

  EXPECT_EQ(reader.read("member"), (std::vector<uint8_t>{'a', 'b', 'c'}));
}

TEST(ZipReaderTest, Verify_BadPayloadCrc_Throws) {
  const auto bytes = make_zip("member", "abd");

  EXPECT_THROW(
      {
        const ZipReader reader = ZipReader::open(bytes);
        reader.verify("member");
      },
      std::runtime_error);
}

TEST(ZipReaderTest, Open_LocalNameDisagrees_Throws) {
  auto bytes = make_zip("member", "abc");
  bytes[30] = 'M';

  EXPECT_THROW(ZipReader::open(bytes), std::runtime_error);
}

TEST(ZipReaderTest, Open_LocalMethodDisagrees_Throws) {
  auto bytes = make_zip("member", "abc");
  bytes[8] = 1;

  EXPECT_THROW(ZipReader::open(bytes), std::runtime_error);
}

TEST(ZipReaderTest, Open_TruncatedCentralDirectory_Throws) {
  std::vector<uint8_t> bytes(22);
  constexpr uint32_t kEocdSignature = 0x06054b50;
  constexpr uint16_t kEntryCount = 1;
  constexpr uint32_t kCentralDirectorySize = 46;
  write_le(bytes, /*offset=*/0, kEocdSignature);
  write_le(bytes, /*offset=*/8, kEntryCount);
  write_le(bytes, /*offset=*/10, kEntryCount);
  write_le(bytes, /*offset=*/12, kCentralDirectorySize);

  EXPECT_THROW(ZipReader::open(bytes), std::runtime_error);
}

} // namespace
} // namespace ptn
