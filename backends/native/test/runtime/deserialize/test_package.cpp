// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/Package.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <zip.h>

namespace ptn {
namespace {

std::vector<uint8_t> make_safetensors(
    std::string_view header,
    std::string_view data) {
  const uint64_t header_size = header.size();
  std::vector<uint8_t> bytes(sizeof(header_size) + header.size() + data.size());
  std::memcpy(bytes.data(), &header_size, sizeof(header_size));
  std::memcpy(bytes.data() + sizeof(header_size), header.data(), header.size());
  std::memcpy(
      bytes.data() + sizeof(header_size) + header.size(),
      data.data(),
      data.size());
  return bytes;
}

ByteSpan as_bytes(std::string_view value) {
  return ByteSpan(reinterpret_cast<const uint8_t*>(value.data()), value.size());
}

struct ZipDiscard {
  void operator()(zip_t* archive) const noexcept {
    zip_discard(archive);
  }
};

class TempPackage {
 private:
  const std::string program_ = "program";
  const std::string aliases_ = R"({"tied_weight":"weight"})";
  std::filesystem::path path_;

 public:
  TempPackage() {
    path_ = std::filesystem::temp_directory_path() /
        ("ptn_package_" + std::to_string(reinterpret_cast<uintptr_t>(this)) +
         ".ptn");
    int error = 0;
    std::unique_ptr<zip_t, ZipDiscard> archive(
        zip_open(path_.string().c_str(), ZIP_CREATE | ZIP_TRUNCATE, &error));
    if (archive == nullptr) {
      throw std::runtime_error("failed to create test package");
    }
    const std::vector<uint8_t> tensors = make_safetensors(
        R"({"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"bias":{"dtype":"I16","shape":[1],"data_offsets":[4,6]}})",
        "dataxy");
    add(archive.get(), kProgramEntry, as_bytes(program_));
    add(archive.get(), kSafeTensorsEntry, ByteSpan(tensors));
    add(archive.get(), kAliasesEntry, as_bytes(aliases_));
    if (zip_close(archive.get()) != 0) {
      throw std::runtime_error("failed to close test package");
    }
    archive.release();
  }

  ~TempPackage() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  std::string path() const {
    return path_.string();
  }

 private:
  static void add(zip_t* archive, const char* name, ByteSpan bytes) {
    zip_source_t* source =
        zip_source_buffer(archive, bytes.data(), bytes.size(), 0);
    if (source == nullptr) {
      throw std::runtime_error("failed to create test package source");
    }
    const zip_int64_t index =
        zip_file_add(archive, name, source, ZIP_FL_ENC_UTF_8);
    if (index < 0) {
      zip_source_free(source);
      throw std::runtime_error("failed to add test package member");
    }
    if (zip_set_file_compression(archive, index, ZIP_CM_STORE, 0) != 0) {
      throw std::runtime_error("failed to store test package member");
    }
  }
};

// cppcheck-suppress-begin syntaxError
TEST(PackageTest, LoadsConstantsOnDemand) {
  const TempPackage file;
  const Package package = Package::load(file.path());

  EXPECT_EQ(package.owner_keys(), (std::vector<std::string>{"weight", "bias"}));
  EXPECT_EQ(package.constant_bytes(), 6);

  const std::optional<ConstantInfo> info = package.constant_info("tied_weight");
  ASSERT_TRUE(info);
  EXPECT_NE(info->package_id, 0);
  EXPECT_EQ(info->dtype, kByte);
  EXPECT_EQ(*info->sizes, (std::vector<int64_t>{4}));
  EXPECT_EQ(info->nbytes, 4);
  EXPECT_EQ(info->owner, "weight");

  std::array<uint8_t, 4> destination{};
  EXPECT_TRUE(package.load_constant_into("weight", destination));
  EXPECT_EQ(destination, (std::array<uint8_t, 4>{'d', 'a', 't', 'a'}));

  const std::optional<OwnedBytes> acquired =
      package.acquire_constant("tied_weight");
  ASSERT_TRUE(acquired);
  EXPECT_TRUE(std::ranges::equal(acquired->span(), destination));
  EXPECT_NO_THROW(package.verify_constants());
}

TEST(PackageTest, ReportsMissingConstantsAndWrongDestinations) {
  const TempPackage file;
  const Package package = Package::load(file.path());

  EXPECT_EQ(package.constant_info("missing"), std::nullopt);
  EXPECT_EQ(package.acquire_constant("missing"), std::nullopt);
  std::array<uint8_t, 1> destination{};
  EXPECT_FALSE(package.load_constant_into("missing", destination));
  EXPECT_THROW(
      package.load_constant_into("weight", destination), std::runtime_error);
}

TEST(PackageTest, SupportsCallerOwnedArchiveBytes) {
  const TempPackage file;
  Package package = Package::load(file.path());
  package = Package::load(OwnedBytes::from_file(file.path(), false));

  const std::optional<OwnedBytes> weight = package.acquire_constant("weight");
  ASSERT_TRUE(weight);
  EXPECT_TRUE(std::ranges::equal(
      weight->span(), (std::array<uint8_t, 4>{'d', 'a', 't', 'a'})));
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
