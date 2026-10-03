/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gtest/gtest.h>
#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include "coreai_bookmarks.h"

namespace executorch::backends::coreai::testing {

runtime::Result<Manifest> bookmark_manifest(bool aot = false);
runtime::Result<NSString*> test_bookmark_key(const Manifest& manifest);
NSURL* test_bookmark_url(NSURL* root, NSString* key);
runtime::Result<NSData*> saved_bookmark(NSURL* root, NSString* key);
runtime::Result<NSData*> sized_bookmark(size_t size, uint32_t tag = 1);

void set_bookmark_test_executable(const char* path);
// Returns -1 for normal test invocations, otherwise a child-mode exit code.
int bookmark_child_mode(int argc, char** argv);

class BookmarkChild {
 public:
  BookmarkChild() = default;
  ~BookmarkChild();
  BookmarkChild(const BookmarkChild&) = delete;
  BookmarkChild& operator=(const BookmarkChild&) = delete;

  ::testing::AssertionResult spawn(NSString* mode, NSString* root, NSString* argument);
  ::testing::AssertionResult receive(char expected);
  ::testing::AssertionResult expect_blocked();
  ::testing::AssertionResult expect_exit(int expected);
  ::testing::AssertionResult kill_and_reap();

 private:
  pid_t pid_ = -1;
  int signal_fd_ = -1;
  ::testing::AssertionResult reap(int& status);
};

}  // namespace executorch::backends::coreai::testing
