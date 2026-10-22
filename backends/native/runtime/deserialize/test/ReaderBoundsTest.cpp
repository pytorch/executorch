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
#include <executorch/backends/native/runtime/deserialize/test/PackageTestData.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace ptn {
namespace {

using testing::make_safetensors;
using testing::write_le;

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

std::vector<uint8_t> make_streaming_zip(
    const std::vector<testing::StoredMember>& members) {
  constexpr uint32_t kEocdSignature = 0x06054b50;
  constexpr uint32_t kCentralDirectorySignature = 0x02014b50;
  constexpr uint32_t kLocalHeaderSignature = 0x04034b50;
  constexpr uint32_t kDataDescriptorSignature = 0x08074b50;
  constexpr uint16_t kDataDescriptorFlag = 8;
  struct Entry {
    const testing::StoredMember* member;
    uint32_t crc;
    uint32_t local_offset;
  };

  std::vector<uint8_t> bytes;
  std::vector<Entry> entries;
  for (const testing::StoredMember& member : members) {
    const size_t offset = bytes.size();
    const size_t record_size =
        30 + member.name.size() + member.payload.size() + 16;
    bytes.resize(offset + record_size);
    const uint32_t crc =
        testing::crc32(member.payload.data(), member.payload.size());
    write_le(bytes, offset, kLocalHeaderSignature);
    write_le<uint16_t>(bytes, offset + 4, /*value=*/20);
    write_le(bytes, offset + 6, kDataDescriptorFlag);
    write_le<uint16_t>(
        bytes, offset + 26, static_cast<uint16_t>(member.name.size()));
    std::memcpy(
        bytes.data() + offset + 30, member.name.data(), member.name.size());
    const size_t payload = offset + 30 + member.name.size();
    std::memcpy(
        bytes.data() + payload, member.payload.data(), member.payload.size());
    const size_t descriptor = payload + member.payload.size();
    write_le(bytes, descriptor, kDataDescriptorSignature);
    write_le(bytes, descriptor + 4, crc);
    write_le<uint32_t>(
        bytes, descriptor + 8, static_cast<uint32_t>(member.payload.size()));
    write_le<uint32_t>(
        bytes, descriptor + 12, static_cast<uint32_t>(member.payload.size()));
    entries.push_back(Entry{&member, crc, static_cast<uint32_t>(offset)});
  }

  const size_t central = bytes.size();
  for (const Entry& entry : entries) {
    const size_t offset = bytes.size();
    bytes.resize(offset + 46 + entry.member->name.size());
    write_le(bytes, offset, kCentralDirectorySignature);
    write_le<uint16_t>(bytes, offset + 4, /*value=*/20);
    write_le<uint16_t>(bytes, offset + 6, /*value=*/20);
    write_le(bytes, offset + 8, kDataDescriptorFlag);
    write_le(bytes, offset + 16, entry.crc);
    write_le<uint32_t>(
        bytes,
        offset + 20,
        static_cast<uint32_t>(entry.member->payload.size()));
    write_le<uint32_t>(
        bytes,
        offset + 24,
        static_cast<uint32_t>(entry.member->payload.size()));
    write_le<uint16_t>(
        bytes, offset + 28, static_cast<uint16_t>(entry.member->name.size()));
    write_le(bytes, offset + 42, entry.local_offset);
    std::memcpy(
        bytes.data() + offset + 46,
        entry.member->name.data(),
        entry.member->name.size());
  }

  const size_t central_size = bytes.size() - central;
  const size_t eocd = bytes.size();
  bytes.resize(eocd + 22);
  write_le(bytes, eocd, kEocdSignature);
  write_le<uint16_t>(bytes, eocd + 8, static_cast<uint16_t>(members.size()));
  write_le<uint16_t>(bytes, eocd + 10, static_cast<uint16_t>(members.size()));
  write_le<uint32_t>(bytes, eocd + 12, static_cast<uint32_t>(central_size));
  write_le<uint32_t>(bytes, eocd + 16, static_cast<uint32_t>(central));
  return bytes;
}

std::vector<uint8_t> make_zip64_descriptor() {
  constexpr uint32_t kEocdSignature = 0x06054b50;
  constexpr uint32_t kCentralDirectorySignature = 0x02014b50;
  constexpr uint32_t kLocalHeaderSignature = 0x04034b50;
  constexpr uint32_t kDataDescriptorSignature = 0x08074b50;
  constexpr uint32_t kZip64Sentinel = 0xffffffffu;
  constexpr uint16_t kZip64ExtraId = 1;
  constexpr uint16_t kDataDescriptorFlag = 8;
  const std::string name = "member";
  const std::vector<uint8_t> payload{'a', 'b', 'c'};
  const uint32_t crc = testing::crc32(payload.data(), payload.size());
  constexpr size_t kZip64ExtraSize = 20;
  constexpr size_t kZip64DescriptorSize = 24;
  const size_t local_size = 30 + name.size() + kZip64ExtraSize +
      payload.size() + kZip64DescriptorSize;
  const size_t central_size = 46 + name.size();
  std::vector<uint8_t> bytes(local_size + central_size + 22);

  write_le(bytes, /*offset=*/0, kLocalHeaderSignature);
  write_le<uint16_t>(bytes, /*offset=*/4, /*value=*/45);
  write_le(bytes, /*offset=*/6, kDataDescriptorFlag);
  write_le(bytes, /*offset=*/18, kZip64Sentinel);
  write_le(bytes, /*offset=*/22, kZip64Sentinel);
  write_le<uint16_t>(bytes, /*offset=*/26, static_cast<uint16_t>(name.size()));
  write_le<uint16_t>(bytes, /*offset=*/28, /*value=*/kZip64ExtraSize);
  std::memcpy(bytes.data() + 30, name.data(), name.size());
  const size_t extra = 30 + name.size();
  write_le(bytes, extra, kZip64ExtraId);
  write_le<uint16_t>(bytes, extra + 2, /*value=*/16);
  write_le<uint64_t>(bytes, extra + 4, /*value=*/0);
  write_le<uint64_t>(bytes, extra + 12, /*value=*/0);
  const size_t payload_offset = extra + kZip64ExtraSize;
  std::memcpy(bytes.data() + payload_offset, payload.data(), payload.size());
  const size_t descriptor = payload_offset + payload.size();
  write_le(bytes, descriptor, kDataDescriptorSignature);
  write_le(bytes, descriptor + 4, crc);
  write_le<uint64_t>(bytes, descriptor + 8, payload.size());
  write_le<uint64_t>(bytes, descriptor + 16, payload.size());

  const size_t central = local_size;
  write_le(bytes, central, kCentralDirectorySignature);
  write_le<uint16_t>(bytes, central + 4, /*value=*/45);
  write_le<uint16_t>(bytes, central + 6, /*value=*/45);
  write_le(bytes, central + 8, kDataDescriptorFlag);
  write_le(bytes, central + 16, crc);
  write_le<uint32_t>(
      bytes, central + 20, static_cast<uint32_t>(payload.size()));
  write_le<uint32_t>(
      bytes, central + 24, static_cast<uint32_t>(payload.size()));
  write_le<uint16_t>(bytes, central + 28, static_cast<uint16_t>(name.size()));
  std::memcpy(bytes.data() + central + 46, name.data(), name.size());

  const size_t eocd = central + central_size;
  write_le(bytes, eocd, kEocdSignature);
  write_le<uint16_t>(bytes, eocd + 8, /*value=*/1);
  write_le<uint16_t>(bytes, eocd + 10, /*value=*/1);
  write_le<uint32_t>(bytes, eocd + 12, static_cast<uint32_t>(central_size));
  write_le<uint32_t>(bytes, eocd + 16, static_cast<uint32_t>(central));
  return bytes;
}

class TemporaryZipFile final {
 public:
  explicit TemporaryZipFile(const std::vector<uint8_t>& bytes)
      : path_(
            std::string(::testing::TempDir()) + "/reader_bounds_" +
            std::to_string(next_id_.fetch_add(1)) + ".zip") {
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
  }

  ~TemporaryZipFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  const std::string& path() const {
    return path_;
  }

 private:
  static std::atomic<uint64_t> next_id_;
  std::string path_;
};

std::atomic<uint64_t> TemporaryZipFile::next_id_{0};

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

TEST(ZipReaderTest, Open_MultipleStreamingMembers_ReturnsPayloads) {
  const auto bytes = make_streaming_zip({
      {"first", {'a'}},
      {"second", {'b'}},
  });

  const ZipReader reader = ZipReader::open(bytes);

  EXPECT_EQ(reader.read("first"), (std::vector<uint8_t>{'a'}));
  EXPECT_EQ(reader.read("second"), (std::vector<uint8_t>{'b'}));
}

TEST(ZipReaderTest, Open_Zip64DataDescriptor_ReturnsPayload) {
  const auto bytes = make_zip64_descriptor();

  const ZipReader reader = ZipReader::open(bytes);

  EXPECT_EQ(reader.read("member"), (std::vector<uint8_t>{'a', 'b', 'c'}));
}

TEST(ZipReaderTest, Read_SharedReaderSupportsConcurrentCalls) {
  const std::vector<uint8_t> payload(64 * 1024, /*value=*/0x5a);
  const std::vector<uint8_t> bytes = testing::make_zip({{"member", payload}});
  const ZipReader reader = ZipReader::open(bytes);
  constexpr size_t kThreadCount = 8;
  std::atomic<bool> start = false;
  std::array<std::exception_ptr, kThreadCount> errors{};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      try {
        for (size_t iteration = 0; iteration < 32; ++iteration) {
          if (reader.read("member") != payload) {
            throw std::runtime_error("concurrent ZIP read returned bad data");
          }
        }
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (std::thread& thread : threads) {
    thread.join();
  }

  for (const std::exception_ptr& error : errors) {
    EXPECT_EQ(error, nullptr);
  }
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

TEST(ZipReaderTest, Open_LocalCrcDisagrees_Throws) {
  auto bytes = make_zip("member", "abc");
  write_le<uint32_t>(bytes, /*offset=*/14, /*value=*/0);

  EXPECT_THROW(ZipReader::open(bytes), std::runtime_error);
}

TEST(ZipReaderTest, Open_LocalSizeDisagrees_Throws) {
  auto bytes = make_zip("member", "abc");
  write_le<uint32_t>(bytes, /*offset=*/18, /*value=*/2);

  EXPECT_THROW(ZipReader::open(bytes), std::runtime_error);
}

TEST(ZipReaderTest, Open_MultiDiskArchive_Throws) {
  auto bytes = make_zip("member", "abc");
  const size_t eocd = bytes.size() - 22;
  write_le<uint16_t>(bytes, eocd + 4, /*value=*/1);

  EXPECT_THROW(ZipReader::open(bytes), std::runtime_error);
}

TEST(ZipReaderTest, Open_DuplicateMemberName_Throws) {
  const auto bytes = testing::make_zip({
      {"member", {'a'}},
      {"member", {'b'}},
  });

  EXPECT_THROW(ZipReader::open(bytes), std::runtime_error);
}

TEST(ZipReaderTest, Open_InconsistentLocalOffset_Throws) {
  auto bytes = testing::make_zip({
      {"first", {'a'}},
      {"other", {'a'}},
  });
  constexpr size_t kLocalRecordSize = 30 + 5 + 1;
  constexpr size_t kCentralEntrySize = 46 + 5;
  constexpr size_t kSecondCentral = 2 * kLocalRecordSize + kCentralEntrySize;
  write_le<uint32_t>(bytes, kSecondCentral + 42, /*value=*/0);

  EXPECT_THROW(ZipReader::open(bytes), std::runtime_error);
}

TEST(ZipReaderTest, Open_PathAppliesMetadataValidation) {
  auto bytes = make_zip("member", "abc");
  bytes[30] = 'M';
  const TemporaryZipFile file(bytes);

  EXPECT_THROW(ZipReader::open(file.path()), std::runtime_error);
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
