// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace ptn {
namespace {

class OwnedBytesTest : public ::testing::Test {
 private:
  std::vector<std::filesystem::path> paths_;

 protected:
  std::string temp_path(std::string_view suffix) {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
        (std::string("owned_bytes_") + info->name() + std::string(suffix));
    std::error_code error;
    std::filesystem::remove(path, error);
    paths_.push_back(path);
    return path.string();
  }

  void write_file(const std::string& path, std::string_view contents) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file);
    file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    ASSERT_TRUE(file);
  }

  void TearDown() override {
    for (const std::filesystem::path& path : paths_) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  }
};

static_assert(!std::is_copy_constructible_v<OwnedBytes>);
static_assert(!std::is_copy_assignable_v<OwnedBytes>);
static_assert(std::is_nothrow_move_constructible_v<OwnedBytes>);
static_assert(std::is_nothrow_move_assignable_v<OwnedBytes>);

// cppcheck-suppress-begin syntaxError
TEST_F(OwnedBytesTest, TakesOwnershipOfVector) {
  std::vector<uint8_t> source{1, 2, 3};
  const uint8_t* data = source.data();

  OwnedBytes bytes = OwnedBytes::from_vector(std::move(source));
  const ByteSpan span = bytes.span();
  OwnedBytes moved = std::move(bytes);

  EXPECT_FALSE(moved.is_mapped());
  EXPECT_EQ(span.data(), data);
  EXPECT_EQ(moved.span().data(), data);
  EXPECT_EQ(
      std::vector<uint8_t>(span.begin(), span.end()),
      (std::vector<uint8_t>{1, 2, 3}));
}

TEST_F(OwnedBytesTest, ReadsFileIntoHeap) {
  const std::string path = temp_path("_heap.bin");
  ASSERT_NO_FATAL_FAILURE(write_file(path, "abc"));

  OwnedBytes bytes = OwnedBytes::from_file(path, false);
  const ByteSpan span = bytes.span();
  const OwnedBytes moved = std::move(bytes);

  EXPECT_FALSE(moved.is_mapped());
  EXPECT_TRUE(bytes.span().empty());
  EXPECT_EQ(
      std::vector<uint8_t>(span.begin(), span.end()),
      (std::vector<uint8_t>{'a', 'b', 'c'}));
}

TEST_F(OwnedBytesTest, MapsFile) {
  const std::string path = temp_path("_mapped.bin");
  ASSERT_NO_FATAL_FAILURE(write_file(path, "abc"));

#if defined(_WIN32)
  EXPECT_THROW(OwnedBytes::from_file(path), std::runtime_error);
#else
  OwnedBytes bytes = OwnedBytes::from_file(path);
  const ByteSpan span = bytes.span();
  const OwnedBytes moved = std::move(bytes);
  EXPECT_TRUE(moved.is_mapped());
  EXPECT_TRUE(bytes.span().empty());
  EXPECT_EQ(
      std::vector<uint8_t>(span.begin(), span.end()),
      (std::vector<uint8_t>{'a', 'b', 'c'}));
#endif
}

TEST_F(OwnedBytesTest, EmptyFileUsesHeapStorage) {
  const std::string path = temp_path("_empty.bin");
  ASSERT_NO_FATAL_FAILURE(write_file(path, ""));

  const OwnedBytes bytes = OwnedBytes::from_file(path);

  EXPECT_FALSE(bytes.is_mapped());
  EXPECT_TRUE(bytes.span().empty());
}

TEST_F(OwnedBytesTest, RejectsInvalidPaths) {
  EXPECT_THROW(
      OwnedBytes::from_file(temp_path("_missing.bin"), false),
      std::runtime_error);
  EXPECT_THROW(
      OwnedBytes::from_file(
          std::filesystem::temp_directory_path().string(), false),
      std::runtime_error);
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
