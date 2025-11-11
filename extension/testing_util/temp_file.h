/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace executorch {
namespace extension {
namespace testing { // Test-only helpers belong in a "testing" sub-namespace.

/**
 * Creates and manages a named temporary file in the file system. Deletes the
 * file when this instance is destroyed.
 *
 * Only for use in gtest tests.
 */
class TempFile {
 public:
  /**
   * Creates a temporary file whose contents are the same as the provided
   * string.
   */
  explicit TempFile(const std::string& contents)
      : TempFile(contents.data(), contents.size()) {}

  /**
   * Creates a temporary file with the provided contents.
   */
  TempFile(const void* data, size_t size) {
    CreateFile(data, size, &path_);
  }

  /**
   * Creates a sparse temporary file with a string at a specific offset.
   * The file will have the specified total size, but only the data at the
   * given offset will be written, creating a sparse file that doesn't
   * allocate all the disk space.
   *
   * Example:
   *   // Create a 3GB file with "DATA_AT_3GB" at 3GB offset
   *   size_t offset = 3ULL * 1024 * 1024 * 1024;
   *   std::string data = "DATA_AT_3GB";
   *   TempFile tf(offset, data, offset + data.size());
   *
   * @param offset Byte offset where the string should be written
   * @param data String to write at the specified offset
   * @param file_size Total size of the sparse file (must be >= offset +
   * data.size())
   */
  TempFile(size_t offset, const std::string& data, size_t file_size) {
    CreateSparseFile(offset, data, file_size, &path_);
  }

  ~TempFile() {
    if (!path_.empty()) {
      std::remove(path_.c_str());
    }
  }

  /**
   * The absolute path to the temporary file.
   */
  const std::string& path() const {
    return path_;
  }

 private:
  // ASSERT_x() macros can only be called from a function returning void, so
  // this logic can't be directly in the ctor.
  void CreateFile(const void* data, size_t size, std::string* out_path) {
    // Find a unique temporary file name.
    std::string path;
    {
      std::array<char, L_tmpnam> buf;
      const char* ret = std::tmpnam(buf.data());
      ASSERT_NE(ret, nullptr) << "Coult not generate temp file";
      buf[L_tmpnam - 1] = '\0';
      path = std::string(buf.data()) + "-executorch-testing";
    }

    // Write the contents to the file.
    std::ofstream file(path, std::ios::out | std::ios::binary);
    ASSERT_TRUE(file.is_open())
        << "open(" << path << ") failed: " << strerror(errno);

    file.write((const char*)data, size);
    ASSERT_TRUE(file.good())
        << "Failed to write " << size << " bytes: " << strerror(errno);

    *out_path = path;
  }

  void CreateSparseFile(
      size_t offset,
      const std::string& data,
      size_t file_size,
      std::string* out_path) {
    ASSERT_GE(file_size, offset + data.size())
        << "file_size must be >= offset + data.size()";

    // Find a unique temporary file name.
    std::string path;
    {
      std::array<char, L_tmpnam> buf;
      const char* ret = std::tmpnam(buf.data());
      ASSERT_NE(ret, nullptr) << "Could not generate temp file";
      buf[L_tmpnam - 1] = '\0';
      path = std::string(buf.data()) + "-executorch-testing";
    }

    // Open file in binary mode for writing.
    std::ofstream file(path, std::ios::out | std::ios::binary);
    ASSERT_TRUE(file.is_open())
        << "open(" << path << ") failed: " << strerror(errno);

    // Seek to the offset.
    file.seekp(offset, std::ios::beg);
    ASSERT_TRUE(file.good()) << "Failed to seek to offset " << offset;

    // Write the data.
    file.write(data.data(), data.size());
    ASSERT_TRUE(file.good())
        << "Failed to write " << data.size() << " bytes at offset " << offset;

    // Ensure file is the specified size by seeking to the end and writing a
    // byte, but only if the file needs to be extended beyond the data we just
    // wrote.
    if (file_size > offset + data.size()) {
      file.seekp(file_size - 1, std::ios::beg);
      ASSERT_TRUE(file.good())
          << "Failed to seek to file_size - 1: " << file_size - 1;

      // Write a single byte to ensure file has the correct size.
      file.write("\0", 1);
      ASSERT_TRUE(file.good())
          << "Failed to write final byte at position " << file_size - 1;
    }

    file.close();
    ASSERT_TRUE(file.good() || file.eof()) << "Error closing file: " << path;

    *out_path = path;
  }

  // Not safely copyable/movable.
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  TempFile(TempFile&&) = delete;
  TempFile& operator=(TempFile&&) = delete;

  std::string path_;
};

} // namespace testing
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace testing {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::testing::TempFile;
} // namespace testing
} // namespace executor
} // namespace torch
