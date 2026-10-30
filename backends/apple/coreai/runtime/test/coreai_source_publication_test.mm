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
TEST_F(CoreAISourceTest, RejectsMismatchedBufferSizesAndReleasesPayload) {
  for (size_t size : {0u, 10u, 12u}) {
    SCOPED_TRACE(size);
    TestDirectory storage;
    ASSERT_NE(storage.url, nil);
    TestData data;
    data.files["coreai/ab/model.aimodel/graph.bin"] = std::string(size, 'x');
    auto manifest = selected_bundle(manifest_dict());
    ASSERT_TRUE(manifest.ok());
    EXPECT_EQ(
        prepare_source_bundle(manifest.get(), &data, storage.url.path).error(),
        Error::InvalidExternalData);
    EXPECT_EQ(data.attempts, 1);
    EXPECT_EQ(data.requests, 1);
    EXPECT_EQ(data.releases, 1);
    EXPECT_EQ(pending_directories(storage.url).count, 0u);
    EXPECT_FALSE([NSFileManager.defaultManager
        fileExistsAtPath:source_entry(storage.url, manifest.get()).path]);
  }
}

TEST_F(CoreAISourceTest, FullyFlushesSourceBeforeAndAfterPublication) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  TestData data;
  aot_data(data, @"arch_b");
  auto manifest = selected_bundle(aot_manifest_dict());
  ASSERT_TRUE(manifest.ok());
  NSURL* final = source_entry(storage.url, manifest.get());
  NSURL* root = sources_root(storage.url);
  const NSUInteger file_count = manifest->files.count;
  __block int synced_files = 0;
  __block int full_syncs = 0;
  __block int publications = 0;
  __block bool ordered = true;
  StorageFaultScope fault(FaultPoint::FullSync);
  fault.armed = false;
  fault.observe = ^(FaultPoint operation) {
    if (operation == FaultPoint::SyncFile)
      ++synced_files;
    if (operation == FaultPoint::FullSync) {
      ++full_syncs;
      ordered &= synced_files == file_count;
      const bool published =
          [NSFileManager.defaultManager fileExistsAtPath:final.path];
      ordered &= full_syncs == 1 ? !published : published;
      ordered &= pending_directories(root).count == (full_syncs == 1 ? 1u : 0u);
    }
    if (operation == FaultPoint::Rename) {
      ++publications;
      ordered &= full_syncs == 1;
    }
  };
  auto source = prepare_source_bundle(manifest.get(), &data, storage.url.path);
  ASSERT_TRUE(source.ok());
  EXPECT_TRUE(ordered);
  EXPECT_EQ(full_syncs, 2);
  EXPECT_EQ(publications, 1);
  EXPECT_EQ(data.attempts, file_count);
  EXPECT_EQ(data.requests, data.releases);
}

TEST_F(CoreAISourceTest, PublicationFailuresCleanPendingWithoutAutomaticRetry) {
  for (auto operation :
       {FaultPoint::CreatePendingDirectory,
        FaultPoint::OpenPendingDirectory,
        FaultPoint::Write,
        FaultPoint::SyncFile,
        FaultPoint::SyncDirectory,
        FaultPoint::FullSync,
        FaultPoint::Rename}) {
    SCOPED_TRACE(static_cast<int>(operation));
    TestDirectory storage;
    ASSERT_NE(storage.url, nil);
    TestData data;
    auto manifest = selected_bundle(manifest_dict());
    ASSERT_TRUE(manifest.ok());
    NSString* key = source_key(manifest.get());
    auto lock = acquire_test_cache_entry(storage.url.path, key);
    ASSERT_TRUE(lock.ok());
    NSURL* final = source_entry(storage.url, manifest.get());
    {
      StorageFaultScope fault(operation);
      EXPECT_EQ(
          executorch::backends::coreai::prepare_source_bundle(
              manifest.get(), &data, *lock.get())
              .error(),
          Error::AccessFailed);
      EXPECT_EQ(fault.matches, 1);
      EXPECT_EQ(fault.hits, 1);
    }
    const int attempts = operation == FaultPoint::CreatePendingDirectory ||
            operation == FaultPoint::OpenPendingDirectory ||
            operation == FaultPoint::SyncDirectory
        ? 0
        : 1;
    EXPECT_EQ(data.attempts, attempts);
    EXPECT_EQ(data.requests, data.releases);
    EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:final.path]);
    EXPECT_EQ(pending_directories(storage.url).count, 0u);
    auto retried = executorch::backends::coreai::prepare_source_bundle(
        manifest.get(), &data, *lock.get());
    ASSERT_TRUE(retried.ok());
    EXPECT_EQ(data.attempts, attempts + 1);
  }
}

TEST_F(CoreAISourceTest, RetriesInterruptedPendingAndPublicationCalls) {
  for (auto operation :
       {FaultPoint::CreatePendingDirectory,
        FaultPoint::OpenPendingDirectory,
        FaultPoint::SyncDirectory,
        FaultPoint::Rename}) {
    SCOPED_TRACE(static_cast<int>(operation));
    TestDirectory storage;
    ASSERT_NE(storage.url, nil);
    TestData data;
    auto manifest = selected_bundle(manifest_dict());
    ASSERT_TRUE(manifest.ok());
    NSString* key = source_key(manifest.get());
    auto lock = acquire_test_cache_entry(storage.url.path, key);
    ASSERT_TRUE(lock.ok());
    StorageFaultScope fault(operation);
    fault.error = EINTR;
    auto source = executorch::backends::coreai::prepare_source_bundle(
        manifest.get(), &data, *lock.get());
    ASSERT_TRUE(source.ok());
    EXPECT_EQ(fault.hits, 1);
    EXPECT_GE(fault.matches, 2);
    EXPECT_EQ(data.attempts, 1);
    EXPECT_EQ(data.requests, data.releases);
    EXPECT_EQ(pending_directories(storage.url).count, 0u);
  }
}

TEST_F(CoreAISourceTest, PostRenameFlushFailureLeavesCompleteSourceVisible) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  TestData data;
  auto manifest = selected_bundle(manifest_dict());
  ASSERT_TRUE(manifest.ok());
  NSURL* final = source_entry(storage.url, manifest.get());
  {
    StorageFaultScope fault(FaultPoint::FullSync);
    fault.match_index = 2;
    EXPECT_EQ(
        prepare_source_bundle(manifest.get(), &data, storage.url.path).error(),
        Error::AccessFailed);
    EXPECT_EQ(fault.matches, 2);
    EXPECT_EQ(fault.hits, 1);
  }
  EXPECT_EQ(data.attempts, 1);
  EXPECT_EQ(data.requests, data.releases);
  EXPECT_EQ(pending_directories(storage.url).count, 0u);
  NSURL* graph = [final URLByAppendingPathComponent:@"model.aimodel/graph.bin"];
  EXPECT_TRUE(([[NSData dataWithContentsOfURL:graph]
      isEqual:[NSData dataWithBytes:"model data" length:11]]));
  NSDictionary* snapshot = nil;
  ASSERT_TRUE(asset_tree_snapshot(storage.url, snapshot));
  {
    StorageFaultScope fault(FaultPoint::FullSync);
    EXPECT_EQ(
        prepare_source_bundle(manifest.get(), nullptr, storage.url.path)
            .error(),
        Error::AccessFailed);
    EXPECT_EQ(fault.hits, 1);
    EXPECT_TRUE(snapshot_matches(storage.url, snapshot));
  }
  StorageFaultScope flush(FaultPoint::FullSync);
  flush.error = EINTR;
  auto reused =
      prepare_source_bundle(manifest.get(), nullptr, storage.url.path);
  ASSERT_TRUE(reused.ok());
  EXPECT_EQ(flush.hits, 1);
  EXPECT_EQ(flush.matches, 2);
  EXPECT_TRUE(snapshot_matches(storage.url, snapshot));
  EXPECT_EQ(data.attempts, 1);
}

TEST_F(CoreAISourceTest, IncompleteConcurrentWinnerFailsWithoutReplacement) {
  TestDirectory storage;
  TestDirectory winner_storage;
  ASSERT_NE(storage.url, nil);
  ASSERT_NE(winner_storage.url, nil);
  TestData data;
  TestData winner_data;
  auto manifest = selected_bundle(manifest_dict());
  ASSERT_TRUE(manifest.ok());
  auto winner = prepare_source_bundle(
      manifest.get(), &winner_data, winner_storage.url.path);
  ASSERT_TRUE(winner.ok());
  ASSERT_TRUE([NSFileManager.defaultManager
      removeItemAtURL:[winner.get() URLByAppendingPathComponent:@"graph.bin"]
                error:nil]);
  NSURL* winner_entry = winner.get().URLByDeletingLastPathComponent;
  NSDictionary* snapshot = nil;
  ASSERT_TRUE(asset_tree_snapshot(winner_entry, snapshot));
  NSURL* root = sources_root(storage.url);
  NSURL* final = source_entry(storage.url, manifest.get());
  __block int publications = 0;
  __block bool callback_ok = true;
  {
    StorageFaultScope fault(FaultPoint::Rename);
    fault.armed = false;
    fault.observe = ^(FaultPoint operation) {
      if (operation != FaultPoint::Rename)
        return;
      ++publications;
      callback_ok &= pending_directories(root).count == 1;
      callback_ok &= [NSFileManager.defaultManager moveItemAtURL:winner_entry
                                                           toURL:final
                                                           error:nil];
    };
    EXPECT_EQ(
        prepare_source_bundle(manifest.get(), &data, storage.url.path).error(),
        Error::InvalidExternalData);
    EXPECT_TRUE(callback_ok);
    EXPECT_EQ(publications, 1);
  }
  EXPECT_EQ(data.attempts, 1);
  EXPECT_EQ(data.requests, data.releases);
  EXPECT_EQ(pending_directories(storage.url).count, 0u);
  EXPECT_TRUE(snapshot_matches(final, snapshot));
}

TEST_F(
    CoreAISourceTest,
    ExclusivePublicationValidatesWinnerAndRemovesLosingPending) {
  TestDirectory storage;
  TestDirectory winner_storage;
  ASSERT_NE(storage.url, nil);
  ASSERT_NE(winner_storage.url, nil);
  TestData data;
  TestData winner_data;
  auto manifest = selected_bundle(manifest_dict());
  ASSERT_TRUE(manifest.ok());
  auto winner = prepare_source_bundle(
      manifest.get(), &winner_data, winner_storage.url.path);
  ASSERT_TRUE(winner.ok());
  NSURL* winner_entry = winner.get().URLByDeletingLastPathComponent;
  NSDictionary* snapshot = nil;
  ASSERT_TRUE(asset_tree_snapshot(winner_entry, snapshot));
  NSURL* root = sources_root(storage.url);
  NSURL* final = source_entry(storage.url, manifest.get());
  __block int publications = 0;
  __block bool callback_ok = true;
  {
    StorageFaultScope fault(FaultPoint::FullSync);
    fault.match_index = 2;
    fault.error = EINTR;
    fault.observe = ^(FaultPoint operation) {
      if (operation != FaultPoint::Rename)
        return;
      ++publications;
      callback_ok &= pending_directories(root).count == 1;
      callback_ok &= [NSFileManager.defaultManager moveItemAtURL:winner_entry
                                                           toURL:final
                                                           error:nil];
    };
    auto result =
        prepare_source_bundle(manifest.get(), &data, storage.url.path);
    EXPECT_TRUE(callback_ok);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE([result.get()
        isEqual:[final URLByAppendingPathComponent:manifest->bundle_name]]);
    EXPECT_EQ(publications, 1);
    EXPECT_EQ(fault.hits, 1);
    EXPECT_EQ(fault.matches, 3);
  }
  EXPECT_EQ(data.attempts, manifest->files.count);
  EXPECT_EQ(data.requests, data.releases);
  EXPECT_EQ(pending_directories(storage.url).count, 0u);
  EXPECT_TRUE(snapshot_matches(final, snapshot));
}

} // namespace executorch::backends::coreai::testing
