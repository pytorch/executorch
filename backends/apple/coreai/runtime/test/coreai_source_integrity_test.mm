/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dispatch/dispatch.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#include "coreai_cache_entry.h"
#include "coreai_cache_key.h"
#include "coreai_fault_scope.h"
#include "coreai_source_fixture.h"
namespace executorch::backends::coreai::testing {
using runtime::Error;
using runtime::Result;
TEST_F(CoreAISourceTest, SourcesRemainAnchoredWhenConfiguredSymlinkChanges) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  NSURL* target = [storage.url URLByAppendingPathComponent:@"target"];
  NSURL* replacement = [storage.url URLByAppendingPathComponent:@"replacement"];
  NSURL* configured = [storage.url URLByAppendingPathComponent:@"configured"];
  ASSERT_EQ(mkdir(target.fileSystemRepresentation, 0777), 0);
  ASSERT_EQ(mkdir(replacement.fileSystemRepresentation, 0777), 0);
  ASSERT_EQ(
      symlink(
          target.fileSystemRepresentation, configured.fileSystemRepresentation),
      0);
  auto bundle = selected_bundle(manifest_dict());
  ASSERT_TRUE(bundle.ok());
  auto key = test_cache_key(bundle.get());
  ASSERT_TRUE(key.ok());
  auto directory = test_cache_directory(configured.path);
  ASSERT_TRUE(directory.ok());
  NSString* canonical_target = [directory->path copy];
  auto locked = LockedCacheEntry::acquire(directory.get(), key.get());
  ASSERT_TRUE(locked.ok());

  ASSERT_EQ(unlink(configured.fileSystemRepresentation), 0);
  ASSERT_EQ(
      symlink(
          replacement.fileSystemRepresentation,
          configured.fileSystemRepresentation),
      0);
  TestData data;
  auto prepared = executorch::backends::coreai::prepare_source_bundle(
      bundle.get(), &data, *locked.get());
  ASSERT_TRUE(prepared.ok());
  EXPECT_TRUE([prepared.get().path hasPrefix:canonical_target]);
  EXPECT_FALSE([prepared.get().path
      hasPrefix:replacement.URLByResolvingSymlinksInPath.path]);
  EXPECT_TRUE(
      [NSFileManager.defaultManager fileExistsAtPath:prepared.get().path]);
}

TEST_F(CoreAISourceTest, RejectsMalformedStoredBundlesWithoutReplacement) {
  for (bool aot : {false, true}) {
    SCOPED_TRACE(aot ? "aot" : "source");
    for (NSString* damage in @[
           @"missing-file",
           @"extra-file",
           @"wrong-size",
           @"symlink-file",
           @"file-at-key",
           @"extra-directory"
         ]) {
      SCOPED_TRACE(damage.UTF8String);
      TestDirectory storage;
      TestDirectory target;
      ASSERT_NE(storage.url, nil);
      ASSERT_NE(target.url, nil);
      TestData data;
      if (aot)
        aot_data(data, @"arch_b");
      auto manifest =
          selected_bundle(aot ? aot_manifest_dict() : manifest_dict());
      ASSERT_TRUE(manifest.ok());
      auto source =
          prepare_source_bundle(manifest.get(), &data, storage.url.path);
      ASSERT_TRUE(source.ok());
      NSURL* entry = source.get().URLByDeletingLastPathComponent;
      NSURL* graph = [source.get() URLByAppendingPathComponent:@"graph.bin"];
      NSData* bytes = [NSData dataWithContentsOfURL:graph];
      ASSERT_NE(bytes, nil);
      NSFileManager* manager = NSFileManager.defaultManager;
      if ([damage isEqual:@"missing-file"]) {
        ASSERT_TRUE([manager removeItemAtURL:graph error:nil]);
      } else if ([damage isEqual:@"extra-file"]) {
        ASSERT_TRUE([[NSData data]
            writeToURL:[source.get() URLByAppendingPathComponent:@"extra"]
            atomically:NO]);
      } else if ([damage isEqual:@"wrong-size"]) {
        ASSERT_TRUE([[NSData data] writeToURL:graph atomically:NO]);
      } else if ([damage isEqual:@"symlink-file"]) {
        NSURL* destination =
            [target.url URLByAppendingPathComponent:@"graph.bin"];
        ASSERT_TRUE([bytes writeToURL:destination atomically:NO]);
        ASSERT_TRUE([manager removeItemAtURL:graph error:nil]);
        ASSERT_TRUE([manager createSymbolicLinkAtPath:graph.path
                                  withDestinationPath:destination.path
                                                error:nil]);
      } else if ([damage isEqual:@"file-at-key"]) {
        ASSERT_TRUE([manager removeItemAtURL:entry error:nil]);
        ASSERT_TRUE([bytes writeToURL:entry atomically:NO]);
      } else {
        ASSERT_TRUE([manager
                   createDirectoryAtURL:
                       [source.get() URLByAppendingPathComponent:@"extra"]
            withIntermediateDirectories:NO
                             attributes:nil
                                  error:nil]);
      }
      NSDictionary* snapshot = nil;
      NSDictionary* target_snapshot = nil;
      ASSERT_TRUE(asset_tree_snapshot(storage.url, snapshot));
      ASSERT_TRUE(asset_tree_snapshot(target.url, target_snapshot));
      const int attempts = data.attempts;
      EXPECT_EQ(
          prepare_source_bundle(manifest.get(), &data, storage.url.path)
              .error(),
          Error::InvalidExternalData);
      EXPECT_EQ(data.attempts, attempts);
      EXPECT_TRUE(snapshot_matches(storage.url, snapshot));
      EXPECT_TRUE(snapshot_matches(target.url, target_snapshot));
      EXPECT_EQ(pending_directories(storage.url).count, 0u);
    }
  }
}

TEST_F(CoreAISourceTest, ExplicitReplacementRepairsMalformedStoredBundles) {
  for (NSString* damage in @[
         @"missing-file",
         @"extra-file",
         @"wrong-size",
         @"symlink-file",
         @"fifo-file",
         @"file-at-key",
         @"symlink-at-key",
         @"extra-directory"
       ]) {
    SCOPED_TRACE(damage.UTF8String);
    TestDirectory storage;
    TestDirectory target;
    ASSERT_NE(storage.url, nil);
    ASSERT_NE(target.url, nil);
    TestData data;
    auto bundle = selected_bundle(manifest_dict());
    ASSERT_TRUE(bundle.ok());
    auto source = prepare_source_bundle(bundle.get(), &data, storage.url.path);
    ASSERT_TRUE(source.ok());
    NSURL* entry = source.get().URLByDeletingLastPathComponent;
    NSURL* graph = [source.get() URLByAppendingPathComponent:@"graph.bin"];
    NSFileManager* manager = NSFileManager.defaultManager;
    if ([damage isEqual:@"file-at-key"] || [damage isEqual:@"symlink-at-key"]) {
      ASSERT_TRUE([manager removeItemAtURL:entry error:nil]);
      if ([damage isEqual:@"file-at-key"]) {
        ASSERT_TRUE([[NSData data] writeToURL:entry atomically:NO]);
      } else {
        ASSERT_EQ(
            symlink(
                target.url.fileSystemRepresentation,
                entry.fileSystemRepresentation),
            0);
      }
    } else if ([damage isEqual:@"extra-file"]) {
      ASSERT_TRUE([[NSData data]
          writeToURL:[source.get() URLByAppendingPathComponent:@"extra"]
          atomically:NO]);
    } else if ([damage isEqual:@"extra-directory"]) {
      ASSERT_TRUE([manager
                 createDirectoryAtURL:[source.get()
                                          URLByAppendingPathComponent:@"extra"]
          withIntermediateDirectories:NO
                           attributes:nil
                                error:nil]);
    } else {
      ASSERT_TRUE([manager removeItemAtURL:graph error:nil]);
      if ([damage isEqual:@"wrong-size"]) {
        ASSERT_TRUE([[NSData data] writeToURL:graph atomically:NO]);
      } else if ([damage isEqual:@"symlink-file"]) {
        NSURL* sentinel = [target.url URLByAppendingPathComponent:@"sentinel"];
        ASSERT_TRUE([@"external" writeToURL:sentinel
                                 atomically:NO
                                   encoding:NSUTF8StringEncoding
                                      error:nil]);
        ASSERT_EQ(
            symlink(
                sentinel.fileSystemRepresentation,
                graph.fileSystemRepresentation),
            0);
      } else if ([damage isEqual:@"fifo-file"]) {
        ASSERT_EQ(mkfifo(graph.fileSystemRepresentation, 0600), 0);
      }
    }
    NSDictionary* target_snapshot = nil;
    ASSERT_TRUE(asset_tree_snapshot(target.url, target_snapshot));
    auto repaired = prepare_source_bundle(
        bundle.get(), &data, storage.url.path, CorruptBundlePolicy::Replace);
    ASSERT_TRUE(repaired.ok());
    EXPECT_TRUE([repaired.get() isEqual:source.get()]);
    EXPECT_TRUE(([[NSData
        dataWithContentsOfURL:[repaired.get()
                                  URLByAppendingPathComponent:@"graph.bin"]]
        isEqual:[NSData dataWithBytes:"model data" length:11]]));
    EXPECT_EQ(data.attempts, 2);
    EXPECT_EQ(data.requests, data.releases);
    EXPECT_EQ(pending_directories(storage.url).count, 0u);
    EXPECT_TRUE(snapshot_matches(target.url, target_snapshot));
  }
}

TEST_F(CoreAISourceTest, PublishedDirectoriesUseDefaultPermissions) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  NSURL* reference = [storage.url URLByAppendingPathComponent:@"reference"];
  ASSERT_EQ(mkdir(reference.fileSystemRepresentation, 0777), 0);
  struct stat expected;
  ASSERT_EQ(stat(reference.fileSystemRepresentation, &expected), 0);
  TestData data;
  auto manifest = selected_bundle(manifest_dict());
  ASSERT_TRUE(manifest.ok());
  auto source = prepare_source_bundle(manifest.get(), &data, storage.url.path);
  ASSERT_TRUE(source.ok());
  for (NSURL* directory in
       @[ source.get().URLByDeletingLastPathComponent, source.get() ]) {
    SCOPED_TRACE(directory.path.UTF8String);
    struct stat actual;
    ASSERT_EQ(stat(directory.fileSystemRepresentation, &actual), 0);
    EXPECT_EQ(actual.st_mode & 0777, expected.st_mode & 0777);
  }
}

TEST_F(CoreAISourceTest, PermissionErrorsRemainDistinctFromCorruptEntries) {
  if (geteuid() == 0) {
    GTEST_SKIP() << "Root can bypass directory read permissions";
  }
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  TestData data;
  auto manifest = selected_bundle(manifest_dict());
  ASSERT_TRUE(manifest.ok());
  auto source = prepare_source_bundle(manifest.get(), &data, storage.url.path);
  ASSERT_TRUE(source.ok());
  NSDictionary* snapshot = nil;
  ASSERT_TRUE(asset_tree_snapshot(storage.url, snapshot));
  for (NSURL* directory in
       @[ source.get().URLByDeletingLastPathComponent, source.get() ]) {
    SCOPED_TRACE(directory.path.UTF8String);
    struct stat original;
    ASSERT_EQ(stat(directory.fileSystemRepresentation, &original), 0);
    ASSERT_EQ(chmod(directory.fileSystemRepresentation, 0000), 0);
    auto denied =
        prepare_source_bundle(manifest.get(), &data, storage.url.path);
    EXPECT_EQ(
        chmod(directory.fileSystemRepresentation, original.st_mode & 07777), 0);
    EXPECT_EQ(denied.error(), Error::AccessFailed);
    EXPECT_TRUE(snapshot_matches(storage.url, snapshot));
    EXPECT_EQ(data.attempts, 1);
    EXPECT_TRUE(
        prepare_source_bundle(manifest.get(), nullptr, storage.url.path).ok());
  }
}

TEST_F(CoreAISourceTest, PreservesNamedDataErrorsWithoutRetrying) {
  for (Error error : {Error::AccessFailed, Error::MemoryAllocationFailed}) {
    SCOPED_TRACE(static_cast<int>(error));
    TestDirectory storage;
    ASSERT_NE(storage.url, nil);
    TestData data;
    data.fail_at = 1;
    data.failure_error = error;
    auto manifest = selected_bundle(manifest_dict());
    ASSERT_TRUE(manifest.ok());
    EXPECT_EQ(
        prepare_source_bundle(manifest.get(), &data, storage.url.path).error(),
        error);
    EXPECT_EQ(data.attempts, 1);
    EXPECT_EQ(data.requests, 0);
    EXPECT_EQ(data.releases, 0);
    EXPECT_EQ(pending_directories(storage.url).count, 0u);
    EXPECT_FALSE([NSFileManager.defaultManager
        fileExistsAtPath:source_entry(storage.url, manifest.get()).path]);
  }
}

TEST_F(CoreAISourceTest, MissingNamedDataIsInvalidExternalData) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  TestData data;
  data.files.clear();
  auto manifest = selected_bundle(manifest_dict());
  ASSERT_TRUE(manifest.ok());
  EXPECT_EQ(
      prepare_source_bundle(manifest.get(), &data, storage.url.path).error(),
      Error::InvalidExternalData);
  EXPECT_EQ(data.attempts, 1);
  EXPECT_EQ(data.requests, 0);
  EXPECT_EQ(pending_directories(storage.url).count, 0u);
}

} // namespace executorch::backends::coreai::testing
