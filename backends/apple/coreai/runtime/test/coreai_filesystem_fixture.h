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

namespace executorch::backends::coreai::testing {

struct TestDirectory {
  NSURL* url = nil;
  TestDirectory();
  ~TestDirectory();
  TestDirectory(const TestDirectory&) = delete;
  TestDirectory& operator=(const TestDirectory&) = delete;
};

struct BookmarkDirectory {
  NSURL* url = nil;
  explicit BookmarkDirectory(NSString* parent = @"/private/tmp");
  ~BookmarkDirectory();
  BookmarkDirectory(const BookmarkDirectory&) = delete;
  BookmarkDirectory& operator=(const BookmarkDirectory&) = delete;
};

::testing::AssertionResult backup_excluded(NSURL* url);
::testing::AssertionResult set_backup_excluded(NSURL* url, bool excluded);

} // namespace executorch::backends::coreai::testing
