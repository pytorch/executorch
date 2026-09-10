// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/ZipReader.h>

#include <array>
#include <cstdint>
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

struct ZipDiscard {
  void operator()(zip_t* archive) const noexcept {
    zip_discard(archive);
  }
};

class TempZip {
 public:
  TempZip() {
    path_ = std::filesystem::temp_directory_path() /
        ("ptn_zip_reader_" + std::to_string(reinterpret_cast<uintptr_t>(this)) +
         ".zip");
    int error = 0;
    std::unique_ptr<zip_t, ZipDiscard> archive(
        zip_open(path_.string().c_str(), ZIP_CREATE | ZIP_TRUNCATE, &error));
    if (archive == nullptr) {
      throw std::runtime_error("failed to create test zip");
    }
    add(archive.get(), "program.ptg", "program");
    add(archive.get(), "program.safetensors", "0123456789");
    if (zip_close(archive.get()) != 0) {
      throw std::runtime_error("failed to close test zip");
    }
    archive.release();
  }

  ~TempZip() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  const std::string path() const {
    return path_.string();
  }

 private:
  static void add(zip_t* archive, const char* name, std::string_view bytes) {
    zip_source_t* source =
        zip_source_buffer(archive, bytes.data(), bytes.size(), 0);
    if (source == nullptr) {
      throw std::runtime_error("failed to create test zip source");
    }
    const zip_int64_t index =
        zip_file_add(archive, name, source, ZIP_FL_ENC_UTF_8);
    if (index < 0) {
      zip_source_free(source);
      throw std::runtime_error("failed to add test zip member");
    }
    if (zip_set_file_compression(archive, index, ZIP_CM_STORE, 0) != 0) {
      throw std::runtime_error("failed to store test zip member");
    }
  }

  std::filesystem::path path_;
};

// cppcheck-suppress-begin syntaxError
TEST(ZipReaderTest, ReadsStoredMemberRanges) {
  const TempZip file;
  ZipReader zip = ZipReader::open(file.path());

  EXPECT_EQ(
      zip.names(),
      (std::vector<std::string>{"program.ptg", "program.safetensors"}));
  EXPECT_EQ(zip.member_size("program.safetensors"), 10);
  EXPECT_EQ(zip.member_size("missing"), std::nullopt);

  std::array<uint8_t, 4> bytes{};
  zip.read_into("program.safetensors", 3, MutableByteSpan(bytes));
  EXPECT_EQ(bytes, (std::array<uint8_t, 4>{'3', '4', '5', '6'}));
  EXPECT_EQ(
      zip.read("program.ptg"),
      (std::vector<uint8_t>{'p', 'r', 'o', 'g', 'r', 'a', 'm'}));
}

TEST(ZipReaderTest, RejectsInvalidRanges) {
  const TempZip file;
  ZipReader zip = ZipReader::open(file.path());

  std::array<uint8_t, 4> bytes{};
  EXPECT_THROW(
      zip.read_into("program.safetensors", 8, MutableByteSpan(bytes)),
      std::runtime_error);
  EXPECT_THROW(
      zip.read_into("missing", 0, MutableByteSpan(bytes)), std::runtime_error);
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
