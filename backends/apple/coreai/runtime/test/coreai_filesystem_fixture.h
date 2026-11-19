/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <Foundation/Foundation.h>
#include <gtest/gtest.h>
#include <optional>
#include <utility>
#include "coreai_cache_root.h"

namespace executorch::backends::coreai::testing {

struct TestDirectory {
  NSURL* url = nil;
  TestDirectory();
  ~TestDirectory();
  TestDirectory(const TestDirectory&) = delete;
  TestDirectory& operator=(const TestDirectory&) = delete;
};

::testing::AssertionResult backup_excluded(NSURL* url);
::testing::AssertionResult set_backup_excluded(NSURL* url, bool excluded);
::testing::AssertionResult asset_tree_snapshot(
    NSURL* root,
    NSDictionary* __strong& result);
::testing::AssertionResult snapshot_matches(
    NSURL* root,
    NSDictionary* expected);
NSArray<NSURL*>* pending_directories(NSURL* root);

// Lets one test body cover both cache directory entry points.
inline runtime::Result<std::optional<CacheDirectory>>
resolve_test_cache_directory(NSString* path, StorageCreationMode mode) {
  if (mode == StorageCreationMode::OpenExisting) {
    return open_cache_directory(path);
  }
  auto created = create_cache_directory(path);
  if (!created.ok()) {
    return created.error();
  }
  return std::optional<CacheDirectory>(std::move(created.get()));
}

} // namespace executorch::backends::coreai::testing
