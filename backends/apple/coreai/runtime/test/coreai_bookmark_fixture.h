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
#include "coreai_cache_entry.h"
#include "coreai_cache_key.h"
#include "coreai_manifest.h"

namespace executorch::backends::coreai::testing {

runtime::Result<SelectedBundle> bookmark_bundle(bool aot = false);
runtime::Result<CacheDirectory> test_cache_directory(
    NSString* path,
    StorageCreationMode mode = StorageCreationMode::Create);
runtime::Result<std::unique_ptr<LockedCacheEntry>> acquire_test_cache_entry(
    NSString* path,
    NSString* key);
runtime::Result<std::unique_ptr<LockedCacheEntry>>
acquire_existing_test_cache_entry(NSString* path, NSString* key);
runtime::Result<NSString*> test_cache_key(const SelectedBundle& bundle);
runtime::Result<NSString*> test_cache_key(
    const SelectedBundle& bundle,
    NSString* specialization_identity);
runtime::Result<NSString*> different_cache_key(
    const SelectedBundle& bundle,
    NSString* key,
    bool same_stripe);
NSURL* test_bookmark_url(NSURL* root, NSString* key);
NSURL* test_cache_lock_url(NSURL* root, NSString* key);
runtime::Result<NSData*> saved_bookmark(NSURL* root, NSString* key);
runtime::Result<NSData*> sized_bookmark(size_t size, uint32_t tag = 1);

bool send_child_byte(int fd, char byte);
void set_bookmark_test_executable(const char* path);
// Returns -1 for normal test invocations, otherwise a child-mode exit code.
int bookmark_child_mode(int argc, char** argv);

class BookmarkChild {
 public:
  BookmarkChild() = default;
  ~BookmarkChild();
  BookmarkChild(const BookmarkChild&) = delete;
  BookmarkChild& operator=(const BookmarkChild&) = delete;

  ::testing::AssertionResult
  spawn(NSString* mode, NSString* root, NSString* argument);
  ::testing::AssertionResult receive(char expected);
  ::testing::AssertionResult expect_blocked();
  ::testing::AssertionResult expect_exit(int expected);
  ::testing::AssertionResult kill_and_reap();

 private:
  pid_t pid_ = -1;
  int signal_fd_ = -1;
  ::testing::AssertionResult reap(int& status);
};

} // namespace executorch::backends::coreai::testing
