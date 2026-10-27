/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_acquisition_fixture.h"

#include <sys/stat.h>
#include <cerrno>
#include <vector>
#include "coreai_fault_scope.h"
#include "coreai_filesystem_fixture.h"
#include "coreai_load_coordinator.h"

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

@interface InvalidAcquisitionLoader : BookmarkFakeLoader
@property(nonatomic) ETCoreAIAcquisitionStatus restoreStatus;
@property(nonatomic, strong) id<ETCoreAIPreparedModel> restoreModel;
@end

@implementation InvalidAcquisitionLoader
- (void)restoreModelFromBookmark:(NSData*)bookmark
                      completion:(ETCoreAIAcquisitionCompletion)completion {
  ++restores;
  completion(self.restoreStatus, self.restoreModel, nil);
}
- (void)specializeModelAtURL:(NSURL*)url
                  completion:(ETCoreAIAcquisitionCompletion)completion {
  ++specializations;
  completion(ETCoreAIAcquisitionStatusMiss, nil, nil);
}
@end

namespace {

class CoreAIAcquisitionTest : public ::testing::Test {
 protected:
  // Declared first so each case's loader, models and data die before it.
  ScopedFakeBridgeState bridge;
  TestDirectory root;
  TestData data;
  Manifest manifest;
  NSString* key = nil;

  void SetUp() override {
    ASSERT_NE(root.url, nil);
    auto selected = bookmark_manifest();
    ASSERT_TRUE(selected.ok());
    manifest = selected.get();
    auto identity = test_bookmark_key(manifest);
    ASSERT_TRUE(identity.ok());
    key = identity.get();
  }

  Error acquire(BookmarkFakeLoader* loader, const NamedDataMap* named_data) {
    @autoreleasepool {
      auto model = acquire_bookmark_model(
          manifest, named_data, root.url.path, @"macOS", @"arch_b", loader);
      if (!model.ok())
        return model.error();
      return model.get() != nil ? Error::Ok : Error::Internal;
    }
  }

  NSData* saved() {
    auto bookmark = saved_bookmark(root.url, key);
    EXPECT_TRUE(bookmark.ok());
    return bookmark.ok() ? bookmark.get() : nil;
  }

  NSURL* staging_key_url() {
    return [[[root.url URLByAppendingPathComponent:@"staging"]
        URLByAppendingPathComponent:key] copy];
  }

  BOOL staging_key_exists() {
    return
        [NSFileManager.defaultManager fileExistsAtPath:staging_key_url().path];
  }

  NSURL* lock_url() {
    return [[root.url URLByAppendingPathComponent:@"locks"]
        URLByAppendingPathComponent:[key stringByAppendingString:@".lock"]];
  }

  ino_t lock_inode() {
    struct stat info;
    if (lstat(lock_url().fileSystemRepresentation, &info) != 0)
      return 0;
    return info.st_ino;
  }
};

TEST_F(CoreAIAcquisitionTest, ColdSpecializesStagedSourceAndPersistsBookmark) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), 1);
  EXPECT_EQ(fake.loader->restores.load(), 0);
  EXPECT_EQ(data.attempts.load(), 1);
  auto staged = acquisition_staged_url(root.url, manifest);
  ASSERT_TRUE(staged.ok());
  EXPECT_TRUE([fake.loader.lastSpecializeURL isEqual:staged.get().path]);
  EXPECT_TRUE([saved() isEqual:fake.loader.lastAcquired.bookmark]);
  EXPECT_FALSE(staging_key_exists());
  EXPECT_FALSE(
      [NSFileManager.defaultManager fileExistsAtPath:staged.get().path]);
}

TEST_F(CoreAIAcquisitionTest, WarmHitRestoresWithoutNamedData) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  NSData* original = saved();
  EXPECT_FALSE(staging_key_exists());
  const int attempts = data.attempts.load();
  ASSERT_EQ(acquire(fake.loader, nullptr), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), 1);
  EXPECT_EQ(fake.loader->restores.load(), 1);
  EXPECT_EQ(data.attempts.load(), attempts);
  EXPECT_TRUE([saved() isEqual:original]);
  EXPECT_FALSE(staging_key_exists());
}

TEST_F(CoreAIAcquisitionTest, RestoreErrorFailsClosedWithoutSourceFallback) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  NSData* original = saved();
  const int attempts = data.attempts;
  fake.loader.restoreError = [NSError errorWithDomain:NSPOSIXErrorDomain
                                                 code:EIO
                                             userInfo:nil];
  EXPECT_EQ(acquire(fake.loader, &data), Error::Internal);
  EXPECT_EQ(fake.loader->specializations.load(), 1);
  EXPECT_EQ(fake.loader->evictions.load(), 0);
  EXPECT_EQ(data.attempts.load(), attempts);
  EXPECT_TRUE([saved() isEqual:original]);
}

TEST_F(
    CoreAIAcquisitionTest,
    RestoreMissRespecializesStagedSourceAndReplacesBookmark) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  NSData* original = saved();
  const int attempts = data.attempts;
  fake.loader.restoreMiss = YES;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), 2);
  EXPECT_EQ(data.attempts.load(), attempts + 1);
  NSData* replaced = saved();
  EXPECT_NE(replaced, nil);
  EXPECT_FALSE([replaced isEqual:original]);
  EXPECT_FALSE(staging_key_exists());
}

TEST_F(
    CoreAIAcquisitionTest,
    SpecializationErrorFailsClosedPreservingBookmark) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  NSData* original = saved();
  ASSERT_NE(original, nil);
  const int attempts = data.attempts.load();
  const int specializations = fake.loader->specializations.load();
  fake.loader.restoreMiss = YES;
  fake.loader.specializeError = [NSError errorWithDomain:NSPOSIXErrorDomain
                                                    code:EIO
                                                userInfo:nil];
  EXPECT_EQ(acquire(fake.loader, &data), Error::Internal);
  EXPECT_EQ(fake.loader->specializations.load(), specializations + 1);
  EXPECT_EQ(fake.loader->evictions.load(), 0);
  EXPECT_EQ(data.attempts.load(), attempts + 1);
  EXPECT_TRUE([saved() isEqual:original]);
  EXPECT_TRUE([NSFileManager.defaultManager
      fileExistsAtPath:fake.loader.lastSpecializeURL]);
}

TEST_F(CoreAIAcquisitionTest, EmptyBookmarkFailsClosedPreservingPrior) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  NSData* original = saved();
  ASSERT_NE(original, nil);
  const int specializations = fake.loader->specializations.load();
  fake.loader.restoreMiss = YES;
  fake.loader.emptyBookmark = YES;
  EXPECT_EQ(acquire(fake.loader, &data), Error::InvalidExternalData);
  EXPECT_EQ(fake.loader->specializations.load(), specializations + 1);
  EXPECT_EQ(fake.loader->evictions.load(), 0);
  EXPECT_TRUE([saved() isEqual:original]);
  EXPECT_TRUE([NSFileManager.defaultManager
      fileExistsAtPath:fake.loader.lastSpecializeURL]);
}

TEST_F(CoreAIAcquisitionTest, NilLoaderRejectsWithoutFilesystemEffects) {
  NSDictionary* before = nil;
  ASSERT_TRUE(asset_tree_snapshot(root.url, before));
  Error error = Error::Ok;
  @autoreleasepool {
    auto result = acquire_bookmark_model(
        manifest, &data, root.url.path, @"macOS", @"arch_b", nil);
    ASSERT_FALSE(result.ok());
    error = result.error();
  }
  EXPECT_EQ(error, Error::InvalidArgument);
  EXPECT_TRUE(snapshot_matches(root.url, before));
  EXPECT_EQ(saved(), nil);
  EXPECT_FALSE(staging_key_exists());
}

TEST_F(CoreAIAcquisitionTest, RestoreHitWithoutModelFailsClosed) {
  NSData* original = nil;
  {
    BookmarkLoaderScope fake;
    ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
    original = saved();
    ASSERT_NE(original, nil);
  }
  InvalidAcquisitionLoader* loader = [[InvalidAcquisitionLoader alloc] init];
  ASSERT_NE(loader, nil);
  loader.restoreStatus = ETCoreAIAcquisitionStatusHit;
  Error error = Error::Ok;
  @autoreleasepool {
    auto result = acquire_bookmark_model(
        manifest, &data, root.url.path, @"macOS", @"arch_b", loader);
    ASSERT_FALSE(result.ok());
    error = result.error();
  }
  EXPECT_EQ(error, Error::Internal);
  loader = nil;
  EXPECT_TRUE([saved() isEqual:original]);
}

TEST_F(CoreAIAcquisitionTest, RestoreMissWithModelFailsClosed) {
  NSData* original = nil;
  {
    BookmarkLoaderScope fake;
    ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
    original = saved();
    ASSERT_NE(original, nil);
  }
  InvalidAcquisitionLoader* loader = [[InvalidAcquisitionLoader alloc] init];
  ASSERT_NE(loader, nil);
  BookmarkFakeModel* model = [[BookmarkFakeModel alloc] init];
  ASSERT_NE(model, nil);
  loader.restoreStatus = ETCoreAIAcquisitionStatusMiss;
  loader.restoreModel = model;
  Error error = Error::Ok;
  @autoreleasepool {
    auto result = acquire_bookmark_model(
        manifest, &data, root.url.path, @"macOS", @"arch_b", loader);
    ASSERT_FALSE(result.ok());
    error = result.error();
  }
  EXPECT_EQ(error, Error::Internal);
  loader.restoreModel = nil;
  loader = nil;
  EXPECT_TRUE([saved() isEqual:original]);
}

TEST_F(CoreAIAcquisitionTest, RestoreErrorWithoutErrorFailsClosed) {
  NSData* original = nil;
  {
    BookmarkLoaderScope fake;
    ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
    original = saved();
    ASSERT_NE(original, nil);
  }
  InvalidAcquisitionLoader* loader = [[InvalidAcquisitionLoader alloc] init];
  ASSERT_NE(loader, nil);
  loader.restoreStatus = ETCoreAIAcquisitionStatusError;
  Error error = Error::Ok;
  @autoreleasepool {
    auto result = acquire_bookmark_model(
        manifest, &data, root.url.path, @"macOS", @"arch_b", loader);
    ASSERT_FALSE(result.ok());
    error = result.error();
  }
  EXPECT_EQ(error, Error::Internal);
  loader = nil;
  EXPECT_TRUE([saved() isEqual:original]);
}

TEST_F(CoreAIAcquisitionTest, SpecializeMissFailsClosed) {
  InvalidAcquisitionLoader* loader = [[InvalidAcquisitionLoader alloc] init];
  ASSERT_NE(loader, nil);
  Error error = Error::Ok;
  @autoreleasepool {
    auto result = acquire_bookmark_model(
        manifest, &data, root.url.path, @"macOS", @"arch_b", loader);
    ASSERT_FALSE(result.ok());
    error = result.error();
  }
  EXPECT_EQ(error, Error::Internal);
  EXPECT_EQ(loader->specializations.load(), 1);
  EXPECT_EQ(loader->restores.load(), 0);
  loader = nil;
  EXPECT_EQ(saved(), nil);
}

TEST_F(
    CoreAIAcquisitionTest,
    ColdSuccessRemovesWholeStagingAfterBookmarkFlush) {
  BookmarkLoaderScope fake;
  __block std::vector<StorageOperation> ops;
  Error error = Error::Internal;
  {
    StorageFaultScope fault(StorageOperation::Remove);
    fault.armed = false;
    fault.observe = ^(StorageOperation operation) {
      ops.push_back(operation);
    };
    error = acquire(fake.loader, &data);
  }
  ASSERT_EQ(error, Error::Ok);
  EXPECT_TRUE([saved() isEqual:fake.loader.lastAcquired.bookmark]);
  EXPECT_FALSE(staging_key_exists());
  auto staged = acquisition_staged_url(root.url, manifest);
  ASSERT_TRUE(staged.ok());
  EXPECT_FALSE(
      [NSFileManager.defaultManager fileExistsAtPath:staged.get().path]);
  size_t last_flush = 0;
  size_t removes = 0;
  size_t first_remove = ops.size();
  for (size_t i = 0; i < ops.size(); ++i) {
    if (ops[i] == StorageOperation::FullSync)
      last_flush = i;
    if (ops[i] == StorageOperation::Remove) {
      ++removes;
      if (first_remove == ops.size())
        first_remove = i;
    }
  }
  EXPECT_EQ(removes, 3u);
  EXPECT_GT(first_remove, last_flush);
}

TEST_F(CoreAIAcquisitionTest, SpecializeErrorPerformsNoCleanup) {
  BookmarkLoaderScope fake;
  __block std::vector<StorageOperation> ops;
  StorageFaultScope fault(StorageOperation::Remove);
  fault.armed = false;
  fault.observe = ^(StorageOperation operation) {
    ops.push_back(operation);
  };
  fake.loader.specializeError = [NSError errorWithDomain:NSPOSIXErrorDomain
                                                    code:EIO
                                                userInfo:nil];
  EXPECT_EQ(acquire(fake.loader, &data), Error::Internal);
  size_t removes = 0;
  for (auto operation : ops) {
    if (operation == StorageOperation::Remove)
      ++removes;
  }
  EXPECT_EQ(removes, 0u);
  EXPECT_EQ(saved(), nil);
  EXPECT_EQ(fake.loader->evictions.load(), 0);
}

TEST_F(CoreAIAcquisitionTest, BookmarkFinalFlushFailurePerformsNoCleanup) {
  BookmarkLoaderScope fake;
  __block std::vector<StorageOperation> ops;
  Error error = Error::Internal;
  {
    // Cold source contributes two FullSyncs and the bookmark two more, so
    // index four is the post-rename bookmark flush.
    StorageFaultScope fault(StorageOperation::FullSync);
    fault.match_index = 4;
    fault.observe = ^(StorageOperation operation) {
      ops.push_back(operation);
    };
    error = acquire(fake.loader, &data);
  }
  EXPECT_EQ(error, Error::AccessFailed);
  EXPECT_TRUE([saved() isEqual:fake.loader.lastAcquired.bookmark]);
  size_t removes = 0;
  for (auto operation : ops) {
    if (operation == StorageOperation::Remove)
      ++removes;
  }
  EXPECT_EQ(removes, 0u);
  auto staged = acquisition_staged_url(root.url, manifest);
  ASSERT_TRUE(staged.ok());
  EXPECT_TRUE(
      [NSFileManager.defaultManager fileExistsAtPath:staged.get().path]);
}

TEST_F(
    CoreAIAcquisitionTest,
    CleanupFirstRemoveFailurePreservesDurableBookmark) {
  BookmarkLoaderScope fake;
  NSData* bookmark = nil;
  int specializations = 0;
  {
    StorageFaultScope fault(StorageOperation::Remove);
    fault.match_index = 1;
    EXPECT_EQ(acquire(fake.loader, &data), Error::Ok);
    EXPECT_EQ(fault.hits, 1);
    bookmark = saved();
    ASSERT_NE(bookmark, nil);
    EXPECT_TRUE([bookmark isEqual:fake.loader.lastAcquired.bookmark]);
    EXPECT_EQ(fake.loader->specializations.load(), 1);
    EXPECT_EQ(fake.loader->restores.load(), 0);
    EXPECT_EQ(fake.loader->evictions.load(), 0);
    specializations = fake.loader->specializations.load();
  }
  EXPECT_EQ(fake.loader->specializations.load(), specializations);
  const int attempts = data.attempts.load();
  const int restores = fake.loader->restores.load();
  EXPECT_TRUE(staging_key_exists());
  ASSERT_EQ(acquire(fake.loader, nullptr), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), specializations);
  EXPECT_EQ(fake.loader->restores.load(), restores + 1);
  EXPECT_EQ(data.attempts.load(), attempts);
  EXPECT_TRUE([saved() isEqual:bookmark]);
  EXPECT_FALSE(staging_key_exists());
}

TEST_F(
    CoreAIAcquisitionTest,
    CleanupPartialRemoveFailurePreservesDurableBookmark) {
  BookmarkLoaderScope fake;
  NSData* bookmark = nil;
  int specializations = 0;
  {
    StorageFaultScope fault(StorageOperation::Remove);
    fault.match_index = 2;
    EXPECT_EQ(acquire(fake.loader, &data), Error::Ok);
    EXPECT_EQ(fault.hits, 1);
    bookmark = saved();
    ASSERT_NE(bookmark, nil);
    EXPECT_TRUE([bookmark isEqual:fake.loader.lastAcquired.bookmark]);
    EXPECT_EQ(fake.loader->specializations.load(), 1);
    EXPECT_EQ(fake.loader->evictions.load(), 0);
    specializations = fake.loader->specializations.load();
  }
  EXPECT_EQ(fake.loader->specializations.load(), specializations);
  const int attempts = data.attempts.load();
  const int restores = fake.loader->restores.load();
  EXPECT_TRUE(staging_key_exists());
  ASSERT_EQ(acquire(fake.loader, nullptr), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), specializations);
  EXPECT_EQ(fake.loader->restores.load(), restores + 1);
  EXPECT_EQ(data.attempts.load(), attempts);
  EXPECT_TRUE([saved() isEqual:bookmark]);
  EXPECT_FALSE(staging_key_exists());
}

TEST_F(
    CoreAIAcquisitionTest,
    CleanupDirectorySyncFailurePreservesDurableBookmark) {
  BookmarkLoaderScope fake;
  NSData* bookmark = nil;
  int specializations = 0;
  {
    StorageFaultScope scope(StorageOperation::SyncDirectory);
    scope.armed = false;
    StorageFaultScope* arming = &scope;
    scope.observe = ^(StorageOperation operation) {
      if (operation == StorageOperation::Remove)
        arming->armed = true;
    };
    EXPECT_EQ(acquire(fake.loader, &data), Error::Ok);
    EXPECT_EQ(scope.hits, 1);
    bookmark = saved();
    ASSERT_NE(bookmark, nil);
    EXPECT_TRUE([bookmark isEqual:fake.loader.lastAcquired.bookmark]);
    EXPECT_EQ(fake.loader->specializations.load(), 1);
    EXPECT_EQ(fake.loader->evictions.load(), 0);
    specializations = fake.loader->specializations.load();
  }
  const int attempts = data.attempts.load();
  const int restores = fake.loader->restores.load();
  EXPECT_TRUE(staging_key_exists());
  ASSERT_EQ(acquire(fake.loader, nullptr), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), specializations);
  EXPECT_EQ(fake.loader->restores.load(), restores + 1);
  EXPECT_EQ(data.attempts.load(), attempts);
  EXPECT_TRUE([saved() isEqual:bookmark]);
  EXPECT_FALSE(staging_key_exists());
}

TEST_F(CoreAIAcquisitionTest, WarmHitCleanupFailureStillReturnsModel) {
  BookmarkLoaderScope fake;
  {
    StorageFaultScope fault(StorageOperation::Remove);
    ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  }
  NSData* original = saved();
  StorageFaultScope fault(StorageOperation::Remove);
  EXPECT_EQ(acquire(fake.loader, nullptr), Error::Ok);
  EXPECT_EQ(fault.hits, 1);
  EXPECT_EQ(fake.loader->restores.load(), 1);
  EXPECT_EQ(fake.loader->specializations.load(), 1);
  EXPECT_EQ(fake.loader->evictions.load(), 0);
  EXPECT_EQ(data.attempts.load(), 1);
  EXPECT_TRUE([saved() isEqual:original]);
}

TEST_F(CoreAIAcquisitionTest, PartialCleanupThenMissRebuildsSource) {
  BookmarkLoaderScope fake;
  {
    StorageFaultScope fault(StorageOperation::Remove);
    fault.match_index = 2;
    ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
    EXPECT_EQ(fault.hits, 1);
  }
  ASSERT_TRUE(staging_key_exists());
  NSData* original = saved();
  fake.loader.restoreMiss = YES;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  EXPECT_EQ(fake.loader->restores.load(), 1);
  EXPECT_EQ(fake.loader->specializations.load(), 2);
  EXPECT_EQ(fake.loader->evictions.load(), 0);
  EXPECT_EQ(data.attempts.load(), 2);
  EXPECT_FALSE([saved() isEqual:original]);
  EXPECT_FALSE(staging_key_exists());
}

TEST_F(
    CoreAIAcquisitionTest,
    DamagedSourceCleanupFailureDoesNotReadOrSpecialize) {
  BookmarkLoaderScope fake;
  {
    StorageFaultScope fault(StorageOperation::Remove);
    fault.match_index = 2;
    ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  }
  NSData* original = saved();
  fake.loader.restoreMiss = YES;
  {
    StorageFaultScope fault(StorageOperation::Remove);
    EXPECT_EQ(acquire(fake.loader, &data), Error::AccessFailed);
    EXPECT_EQ(fault.hits, 1);
  }
  EXPECT_EQ(data.attempts.load(), 1);
  EXPECT_EQ(fake.loader->specializations.load(), 1);
  EXPECT_EQ(fake.loader->evictions.load(), 0);
  EXPECT_TRUE([saved() isEqual:original]);
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  EXPECT_FALSE(staging_key_exists());
}

TEST_F(CoreAIAcquisitionTest, MissingNamedDataIsNotRetried) {
  BookmarkLoaderScope fake;
  data.files.clear();
  EXPECT_EQ(acquire(fake.loader, &data), Error::InvalidExternalData);
  EXPECT_EQ(data.attempts.load(), 1);
  EXPECT_EQ(fake.loader->specializations.load(), 0);
  EXPECT_EQ(saved(), nil);
}

TEST_F(CoreAIAcquisitionTest, BeforeSdkFaultNeverInvokesLoader) {
  BookmarkLoaderScope fake;
  StorageFaultScope fault(StorageOperation::BeforeSDK);
  EXPECT_EQ(acquire(fake.loader, &data), Error::AccessFailed);
  EXPECT_EQ(fake.loader->specializations.load(), 0);
  EXPECT_EQ(fake.loader->restores.load(), 0);
  EXPECT_EQ(fault.hits, 1);
  EXPECT_EQ(saved(), nil);
}

TEST_F(CoreAIAcquisitionTest, KeyLockIdentitySurvivesSuccessfulAcquisition) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  ino_t first = lock_inode();
  ASSERT_NE(first, 0u);
  ASSERT_EQ(acquire(fake.loader, nullptr), Error::Ok);
  EXPECT_EQ(lock_inode(), first);
}

class CoreAIAcquisitionSaveFailureTest
    : public CoreAIAcquisitionTest,
      public ::testing::WithParamInterface<bool> {};
TEST_P(CoreAIAcquisitionSaveFailureTest, PreservesBookmarkAndRetries) {
  const bool prior = GetParam();
  BookmarkLoaderScope fake;
  NSData* original = nil;
  if (prior) {
    ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
    original = saved();
    ASSERT_NE(original, nil);
    EXPECT_FALSE(staging_key_exists());
    fake.loader.restoreMiss = YES;
  }
  {
    // Cold and rematerialized sources publish before the bookmark, so the
    // bookmark rename is the second rename once the first success has
    // cleaned the source.
    StorageFaultScope fault(StorageOperation::Rename);
    fault.match_index = 2;
    EXPECT_EQ(acquire(fake.loader, &data), Error::AccessFailed);
    EXPECT_EQ(fault.hits, 1);
  }
  EXPECT_TRUE(original == nil ? saved() == nil : [saved() isEqual:original]);
  EXPECT_TRUE([NSFileManager.defaultManager
      fileExistsAtPath:fake.loader.lastSpecializeURL]);
  const int calls = fake.loader->specializations;
  fake.loader.restoreMiss = NO;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), calls + (prior ? 0 : 1));
  EXPECT_FALSE(staging_key_exists());
}
INSTANTIATE_TEST_SUITE_P(
    Boundaries,
    CoreAIAcquisitionSaveFailureTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "PriorBookmark" : "Cold";
    });

class CoreAIAcquisitionProcessTest : public CoreAIAcquisitionTest {
 protected:
  void expect_parallel(bool same_key) {
    NSMutableDictionary* first = manifest_dict();
    NSMutableDictionary* second = [first mutableCopy];
    if (!same_key)
      second[@"bundle_digests"] = @{@"model.aimodel" : fixture_identity('2')};
    BookmarkChild child;
    ASSERT_TRUE(
        spawn_parallel_acquisition(child, root.url, first, second, same_key));
    ASSERT_TRUE(child.receive('C'));
    ASSERT_TRUE(child.receive(1)) << "harness completed without timeouts";
    ASSERT_TRUE(child.receive(1))
        << "second acquisition blocked or entered as expected";
    ASSERT_TRUE(child.receive(1)) << "both acquisitions succeeded";
    ASSERT_TRUE(child.receive(same_key ? 1 : 2)) << "specializations";
    ASSERT_TRUE(child.receive(same_key ? 1 : 2)) << "named-data attempts";
    ASSERT_TRUE(child.receive(same_key ? 1 : 0)) << "restores";
    ASSERT_TRUE(child.receive(1)) << "no live fake SDK objects";
    ASSERT_TRUE(child.expect_exit(0));
  }
};

TEST_F(CoreAIAcquisitionProcessTest, ContendedSameKeySerializesSpecialization) {
  expect_parallel(true);
}

TEST_F(CoreAIAcquisitionProcessTest, IndependentKeysSpecializeInParallel) {
  expect_parallel(false);
}

TEST_F(CoreAIAcquisitionProcessTest, CrashBeforeSdkRetainsStagingAndRetries) {
  BookmarkChild child;
  ASSERT_TRUE(spawn_acquisition_crash(
      child, root.url, StorageOperation::BeforeSDK, manifest_dict()));
  ASSERT_TRUE(child.receive('C'));
  ASSERT_TRUE(child.expect_exit(73));
  EXPECT_EQ(saved(), nil);
  EXPECT_TRUE(staging_key_exists());
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), 1);
  EXPECT_EQ(data.attempts.load(), 0);
  EXPECT_FALSE(staging_key_exists());
}

TEST_F(CoreAIAcquisitionProcessTest, CrashAfterSdkRetainsStagingAndRetries) {
  BookmarkChild child;
  ASSERT_TRUE(spawn_acquisition_crash(
      child, root.url, StorageOperation::AfterSDK, manifest_dict()));
  ASSERT_TRUE(child.receive('C'));
  ASSERT_TRUE(child.expect_exit(73));
  EXPECT_EQ(saved(), nil);
  auto staged = acquisition_staged_url(root.url, manifest);
  ASSERT_TRUE(staged.ok());
  EXPECT_TRUE(
      [NSFileManager.defaultManager fileExistsAtPath:staged.get().path]);
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), 1);
  EXPECT_EQ(fake.loader->restores.load(), 0);
  EXPECT_EQ(data.attempts.load(), 0);
  EXPECT_NE(saved(), nil);
  EXPECT_FALSE(staging_key_exists());
}

} // namespace
