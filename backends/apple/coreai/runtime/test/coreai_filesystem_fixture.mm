/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_filesystem_fixture.h"
#include <sys/stat.h>
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

::testing::AssertionResult asset_tree_snapshot(NSURL* root, NSDictionary* __strong& result) {
  result = nil;
  if (root == nil) return ::testing::AssertionFailure() << "Missing snapshot root";
  NSFileManager* manager = NSFileManager.defaultManager;
  NSError* error = nil;
  NSArray<NSString*>* children = [manager subpathsOfDirectoryAtPath:root.path error:&error];
  if (children == nil || error != nil) {
    return ::testing::AssertionFailure() << "Cannot list " << root.path.UTF8String;
  }
  NSMutableDictionary* snapshot = [NSMutableDictionary dictionary];
  for (NSString* path in [@[ @"" ] arrayByAddingObjectsFromArray:children]) {
    NSURL* url = path.length == 0 ? root : [root URLByAppendingPathComponent:path];
    struct stat info;
    if (lstat(url.fileSystemRepresentation, &info) != 0) {
      return ::testing::AssertionFailure() << "Cannot stat " << url.path.UTF8String;
    }
    NSMutableDictionary* item = [@{
      @"inode" : @(info.st_ino),
      @"device" : @(info.st_dev),
      @"mode" : @(info.st_mode),
      @"size" : @(info.st_size),
      @"mtime_seconds" : @(info.st_mtimespec.tv_sec),
      @"mtime_nanos" : @(info.st_mtimespec.tv_nsec)
    } mutableCopy];
    error = nil;
    if (S_ISREG(info.st_mode)) {
      NSData* bytes = [NSData dataWithContentsOfURL:url options:0 error:&error];
      if (bytes == nil || error != nil) {
        return ::testing::AssertionFailure() << "Cannot read " << url.path.UTF8String;
      }
      item[@"bytes"] = bytes;
    } else if (S_ISLNK(info.st_mode)) {
      NSString* destination = [manager destinationOfSymbolicLinkAtPath:url.path error:&error];
      if (destination == nil || error != nil) {
        return ::testing::AssertionFailure() << "Cannot read link " << url.path.UTF8String;
      }
      item[@"link"] = destination;
    }
    snapshot[path] = item;
  }
  result = snapshot;
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult snapshot_matches(NSURL* root, NSDictionary* expected) {
  if (expected == nil) return ::testing::AssertionFailure() << "Missing expected snapshot";
  NSDictionary* actual = nil;
  auto read = asset_tree_snapshot(root, actual);
  if (!read) return read;
  return [actual isEqual:expected] ? ::testing::AssertionSuccess()
                                   : ::testing::AssertionFailure()
                                         << "Filesystem snapshot changed: " << root.path.UTF8String;
}

NSArray<NSURL*>* staging_directories(NSURL* root) {
  if (root == nil) {
    ADD_FAILURE() << "Missing staging root";
    return nil;
  }
  NSError* error = nil;
  NSArray<NSURL*>* children = [NSFileManager.defaultManager contentsOfDirectoryAtURL:root
                                                          includingPropertiesForKeys:nil
                                                                             options:0
                                                                               error:&error];
  if (children == nil || error != nil) {
    ADD_FAILURE() << "Cannot list staging directories: " << root.path.UTF8String;
    return nil;
  }
  NSMutableArray<NSURL*>* staging = [NSMutableArray array];
  for (NSURL* child in children) {
    if ([child.lastPathComponent hasPrefix:@".staging-"]) [staging addObject:child];
  }
  return staging;
}

}  // namespace executorch::backends::coreai::testing
