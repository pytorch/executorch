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

#include <fcntl.h> // open()
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
