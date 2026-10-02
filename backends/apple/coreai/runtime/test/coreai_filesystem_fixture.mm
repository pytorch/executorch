/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_filesystem_fixture.h"
#include <unistd.h>
#include <string>

namespace executorch::backends::coreai::testing {

TestDirectory::TestDirectory() {
  char pattern[] = "/private/tmp/coreai-host-test-XXXXXX";
  if (mkdtemp(pattern) == nullptr) {
    ADD_FAILURE() << "mkdtemp: " << strerror(errno);
    return;
  }
  url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:pattern] isDirectory:YES]
            .URLByResolvingSymlinksInPath;
}

TestDirectory::~TestDirectory() {
  if (url == nil) return;
  EXPECT_TRUE([url.lastPathComponent hasPrefix:@"coreai-host-test-"]);
  if (![url.lastPathComponent hasPrefix:@"coreai-host-test-"]) return;
  EXPECT_TRUE([NSFileManager.defaultManager removeItemAtURL:url error:nil]);
}

::testing::AssertionResult backup_excluded(NSURL* url) {
  NSURL* fresh = [NSURL fileURLWithPath:url.path];
  [fresh removeAllCachedResourceValues];
  NSNumber* value = nil;
  NSError* error = nil;
  if (![fresh getResourceValue:&value forKey:NSURLIsExcludedFromBackupKey error:&error] ||
      error != nil || value == nil) {
    ADD_FAILURE() << "Cannot read backup flag: " << url.path.UTF8String;
    return ::testing::AssertionFailure() << "Backup flag unavailable";
  }
  return value.boolValue ? ::testing::AssertionSuccess()
                         : ::testing::AssertionFailure() << "Not excluded: " << url.path.UTF8String;
}

::testing::AssertionResult set_backup_excluded(NSURL* url, bool excluded) {
  // Foundation can defer metadata writes; settle deliberate fixture changes.
  for (int attempt = 0; attempt < 50; ++attempt) {
    NSURL* fresh = [NSURL fileURLWithPath:url.path];
    [fresh removeAllCachedResourceValues];
    if (![fresh setResourceValue:@(excluded) forKey:NSURLIsExcludedFromBackupKey error:nil]) {
      return ::testing::AssertionFailure() << "Cannot set backup flag: " << url.path.UTF8String;
    }
    usleep(20000);
    if (static_cast<bool>(backup_excluded(url)) == excluded) {
      return ::testing::AssertionSuccess();
    }
  }
  return ::testing::AssertionFailure() << "Backup flag did not settle: " << url.path.UTF8String;
}

BookmarkDirectory::BookmarkDirectory(NSString* parent) {
  if (parent == nil) {
    ADD_FAILURE() << "Bookmark directory requires a parent";
    return;
  }
  std::string pattern =
      std::string(parent.fileSystemRepresentation) + "/.coreai-bookmark-test-XXXXXX";
  if (mkdtemp(pattern.data()) == nullptr) {
    ADD_FAILURE() << "mkdtemp: " << strerror(errno);
    return;
  }
  url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:pattern.c_str()] isDirectory:YES];
}

BookmarkDirectory::~BookmarkDirectory() {
  if (url == nil) return;
  EXPECT_TRUE([url.lastPathComponent hasPrefix:@".coreai-bookmark-test-"]);
  if (![url.lastPathComponent hasPrefix:@".coreai-bookmark-test-"]) return;
  EXPECT_TRUE([NSFileManager.defaultManager removeItemAtURL:url error:nil]);
}

}  // namespace executorch::backends::coreai::testing
