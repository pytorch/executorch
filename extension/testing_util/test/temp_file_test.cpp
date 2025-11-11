/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/testing_util/temp_file.h>

#include <memory>
#include <string>

#include <errno.h>
#include <fcntl.h>
#include <string.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::extension::testing::TempFile;

TEST(TempFileTest, Smoke) {
  std::string path;
  {
    // Create a TempFile with known contents.
    const char contents[] = "12345678";
    TempFile tf(contents, sizeof(contents));
    EXPECT_NE(tf.path(), "");

    // Hold onto the path so we can check again later.
    path = tf.path();

    // Open the file by name.
    int fd = open(tf.path().c_str(), O_RDONLY);
    ASSERT_GE(fd, 0) << "Failed to open " << tf.path() << ": "
                     << strerror(errno);

    // Read back the contents.
    char buffer[128] = {};
    ssize_t nread = read(fd, buffer, sizeof(buffer));
    EXPECT_EQ(nread, sizeof(contents))
        << "read(" << fd << ", ...) of " << sizeof(buffer)
        << " bytes failed: " << strerror(errno);
    close(fd);

    // Make sure they're the same as what was provided.
    EXPECT_EQ(0, memcmp(buffer, contents, sizeof(contents)));
  }

  // Destroying the TempFile should have deleted the file.
  int fd = open(path.c_str(), O_RDONLY);
  EXPECT_LT(fd, 0) << "File " << path << " should not exist";
}

TEST(TempFileTest, StringCtor) {
  // Create a TempFile using the std::string ctor.
  std::string contents = "abcdefgh";
  TempFile tf(contents);

  // Open the file by name.
  int fd = open(tf.path().c_str(), O_RDONLY);
  ASSERT_GE(fd, 0) << "Failed to open " << tf.path() << ": " << strerror(errno);

  // Read back the contents.
  char buffer[128] = {};
  ssize_t nread = read(fd, buffer, sizeof(buffer));
  EXPECT_EQ(nread, contents.size())
      << "read(" << fd << ", ...) of " << sizeof(buffer)
      << " bytes failed: " << strerror(errno);
  close(fd);

  // Make sure they're the same as what was provided.
  std::string actual(buffer, nread);
  EXPECT_EQ(contents, actual);
}
