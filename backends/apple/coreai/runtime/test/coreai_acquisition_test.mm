/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_acquisition_fixture.h"

#include <executorch/runtime/platform/log.h>
#include <sys/stat.h>
#include <cerrno>
#include <initializer_list>
#include <optional>
#include <string>
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

TEST(CoreAIRuntimeAssetsTest, ProbesHostAndSelectsImmutableBundle) {
  ScopedFakeBridgeState bridge;
  auto parsed = parse_manifest(encode(aot_manifest_dict()));
  ASSERT_TRUE(parsed.ok());
  auto assets = select_runtime_assets(parsed.get());
  ASSERT_TRUE(assets.ok());
  EXPECT_TRUE([assets->platform isEqual:@"macOS"]);
  EXPECT_TRUE([assets->architecture isEqual:@"arch_b"]);
  EXPECT_TRUE([assets->bundle.bundle_name isEqual:@"model.arch_b.aimodelc"]);
  EXPECT_TRUE([assets->bundle.model_hash isEqual:parsed->model_hash]);
}

class CoreAIAcquisitionTest : public ::testing::Test {
 protected:
  // Declared first so each case's loader, models and data die before it.
  ScopedFakeBridgeState bridge;
  TestDirectory root;
  TestData data;
  std::optional<RuntimeAssets> assets;
  NSString* key = nil;

  void SetUp() override {
    ASSERT_NE(root.url, nil);
    auto selected = bookmark_bundle();
    ASSERT_TRUE(selected.ok());
    assets.emplace(RuntimeAssets{selected.get(), @"macOS", @"arch_b"});
    auto identity = test_cache_key(assets->bundle);
    ASSERT_TRUE(identity.ok());
    key = identity.get();
  }

  Error acquire(BookmarkFakeLoader* loader, const NamedDataMap* named_data) {
    @autoreleasepool {
      auto model =
          acquire_bookmark_model(*assets, named_data, root.url.path, loader);
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
    return test_cache_lock_url(root.url, key);
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
  auto staged = acquisition_staged_url(root.url, assets->bundle);
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
    auto result = acquire_bookmark_model(*assets, &data, root.url.path, nil);
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
    auto result = acquire_bookmark_model(*assets, &data, root.url.path, loader);
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
    auto result = acquire_bookmark_model(*assets, &data, root.url.path, loader);
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
    auto result = acquire_bookmark_model(*assets, &data, root.url.path, loader);
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
    auto result = acquire_bookmark_model(*assets, &data, root.url.path, loader);
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
  __block std::vector<FaultPoint> ops;
  Error error = Error::Internal;
  {
    StorageFaultScope fault(FaultPoint::Remove);
    fault.armed = false;
    fault.observe = ^(FaultPoint operation) {
      ops.push_back(operation);
    };
    error = acquire(fake.loader, &data);
  }
  ASSERT_EQ(error, Error::Ok);
  EXPECT_TRUE([saved() isEqual:fake.loader.lastAcquired.bookmark]);
  EXPECT_FALSE(staging_key_exists());
  auto staged = acquisition_staged_url(root.url, assets->bundle);
  ASSERT_TRUE(staged.ok());
  EXPECT_FALSE(
      [NSFileManager.defaultManager fileExistsAtPath:staged.get().path]);
  size_t last_flush = 0;
  size_t removes = 0;
  size_t first_remove = ops.size();
  for (size_t i = 0; i < ops.size(); ++i) {
    if (ops[i] == FaultPoint::FullSync)
      last_flush = i;
    if (ops[i] == FaultPoint::Remove) {
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
  __block std::vector<FaultPoint> ops;
  StorageFaultScope fault(FaultPoint::Remove);
  fault.armed = false;
  fault.observe = ^(FaultPoint operation) {
    ops.push_back(operation);
  };
  fake.loader.specializeError = [NSError errorWithDomain:NSPOSIXErrorDomain
                                                    code:EIO
                                                userInfo:nil];
  EXPECT_EQ(acquire(fake.loader, &data), Error::Internal);
  size_t removes = 0;
  for (auto operation : ops) {
    if (operation == FaultPoint::Remove)
      ++removes;
  }
  EXPECT_EQ(removes, 0u);
  EXPECT_EQ(saved(), nil);
  EXPECT_EQ(fake.loader->evictions.load(), 0);
}

TEST_F(CoreAIAcquisitionTest, BookmarkFinalFlushFailurePerformsNoCleanup) {
  BookmarkLoaderScope fake;
  __block std::vector<FaultPoint> ops;
  Error error = Error::Internal;
  {
    // Cold source contributes two FullSyncs and the bookmark two more, so
    // index four is the post-rename bookmark flush.
    StorageFaultScope fault(FaultPoint::FullSync);
    fault.match_index = 4;
    fault.observe = ^(FaultPoint operation) {
      ops.push_back(operation);
    };
    error = acquire(fake.loader, &data);
  }
  EXPECT_EQ(error, Error::AccessFailed);
  EXPECT_TRUE([saved() isEqual:fake.loader.lastAcquired.bookmark]);
  size_t removes = 0;
  for (auto operation : ops) {
    if (operation == FaultPoint::Remove)
      ++removes;
  }
  EXPECT_EQ(removes, 0u);
  auto staged = acquisition_staged_url(root.url, assets->bundle);
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
    StorageFaultScope fault(FaultPoint::Remove);
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
    StorageFaultScope fault(FaultPoint::Remove);
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
    StorageFaultScope scope(FaultPoint::SyncDirectory);
    scope.armed = false;
    StorageFaultScope* arming = &scope;
    scope.observe = ^(FaultPoint operation) {
      if (operation == FaultPoint::Remove)
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
    StorageFaultScope fault(FaultPoint::Remove);
    ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  }
  NSData* original = saved();
  StorageFaultScope fault(FaultPoint::Remove);
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
    StorageFaultScope fault(FaultPoint::Remove);
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
    StorageFaultScope fault(FaultPoint::Remove);
    fault.match_index = 2;
    ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  }
  NSData* original = saved();
  fake.loader.restoreMiss = YES;
  {
    StorageFaultScope fault(FaultPoint::Remove);
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
  StorageFaultScope fault(FaultPoint::BeforeSDK);
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
    StorageFaultScope fault(FaultPoint::Rename);
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
      child, root.url, FaultPoint::BeforeSDK, manifest_dict()));
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
      child, root.url, FaultPoint::AfterSDK, manifest_dict()));
  ASSERT_TRUE(child.receive('C'));
  ASSERT_TRUE(child.expect_exit(73));
  EXPECT_EQ(saved(), nil);
  auto staged = acquisition_staged_url(root.url, assets->bundle);
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

class CoreAILifecycleLogTest : public CoreAIAcquisitionTest {
 protected:
  void SetUp() override {
    CoreAIAcquisitionTest::SetUp();
    if (!ET_LOG_ENABLED || LogLevel::Info < LogLevel::ET_MIN_LOG_LEVEL) {
      GTEST_SKIP() << "Info logging is compiled out";
    }
  }

  template <typename Operation>
  std::string capture(Operation operation, Error expected = Error::Ok) {
    ::testing::internal::CaptureStderr();
    const auto error = operation();
    auto logs = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(error, expected);
    return logs;
  }

  void expect_order(
      const std::string& logs,
      std::initializer_list<const char*> events) {
    size_t position = 0;
    for (const char* event : events) {
      const auto found = logs.find(event, position);
      ASSERT_NE(found, std::string::npos) << event << "\n" << logs;
      position = found + 1;
    }
    EXPECT_NE(logs.find(key.UTF8String), std::string::npos);
  }
};

TEST_F(CoreAILifecycleLogTest, ColdAcquisitionLogsValidatedMilestonesInOrder) {
  BookmarkLoaderScope fake;
  const auto logs = capture([&] { return acquire(fake.loader, &data); });
  expect_order(
      logs,
      {
          "Core AI cache lookup:",
          "Core AI cache miss: no bookmark",
          "Core AI specialization started:",
          "Core AI specialization succeeded:",
          "Core AI cache bookmark saved durably:",
      });
  EXPECT_EQ(logs.find("Core AI cache hit:"), std::string::npos);
  EXPECT_TRUE([saved() isEqual:fake.loader.lastAcquired.bookmark]);
}

TEST_F(CoreAILifecycleLogTest, WarmHitDoesNotLogSpecializationOrSave) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  const auto logs = capture([&] { return acquire(fake.loader, nullptr); });
  expect_order(
      logs,
      {
          "Core AI cache lookup:",
          "Core AI cache hit: restored model from bookmark",
      });
  EXPECT_EQ(logs.find("Core AI cache miss:"), std::string::npos);
  EXPECT_EQ(logs.find("Core AI specialization"), std::string::npos);
  EXPECT_EQ(logs.find("Core AI cache bookmark saved"), std::string::npos);
}

TEST_F(CoreAILifecycleLogTest, ConfirmedMissLogsRespecialization) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  fake.loader.restoreMiss = YES;
  const auto logs = capture([&] { return acquire(fake.loader, &data); });
  expect_order(
      logs,
      {
          "Core AI cache miss: SDK could not restore bookmark",
          "Core AI specialization started:",
          "Core AI specialization succeeded:",
          "Core AI cache bookmark saved durably:",
      });
  EXPECT_EQ(logs.find("Core AI cache hit:"), std::string::npos);
}

TEST_F(CoreAILifecycleLogTest, RestoreErrorIsNotReportedAsHitOrMiss) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  fake.loader.restoreError = [NSError errorWithDomain:NSPOSIXErrorDomain
                                                 code:EIO
                                             userInfo:nil];
  const auto logs =
      capture([&] { return acquire(fake.loader, &data); }, Error::Internal);
  EXPECT_EQ(logs.find("Core AI cache hit:"), std::string::npos);
  EXPECT_EQ(logs.find("Core AI cache miss:"), std::string::npos);
  EXPECT_EQ(logs.find("Core AI specialization"), std::string::npos);
  EXPECT_EQ(logs.find("Core AI cache bookmark saved"), std::string::npos);
}

TEST_F(CoreAILifecycleLogTest, SpecializationErrorDoesNotLogSuccessOrSave) {
  BookmarkLoaderScope fake;
  fake.loader.specializeError = [NSError errorWithDomain:NSPOSIXErrorDomain
                                                    code:EIO
                                                userInfo:nil];
  const auto logs =
      capture([&] { return acquire(fake.loader, &data); }, Error::Internal);
  EXPECT_NE(logs.find("Core AI specialization started:"), std::string::npos);
  EXPECT_EQ(logs.find("Core AI specialization succeeded:"), std::string::npos);
  EXPECT_EQ(logs.find("Core AI cache bookmark saved"), std::string::npos);
}

TEST_F(CoreAILifecycleLogTest, BeforeSdkFaultDoesNotLogSpecializationStart) {
  BookmarkLoaderScope fake;
  StorageFaultScope fault(FaultPoint::BeforeSDK);
  const auto logs =
      capture([&] { return acquire(fake.loader, &data); }, Error::AccessFailed);
  EXPECT_EQ(fake.loader->specializations.load(), 0);
  EXPECT_EQ(logs.find("Core AI specialization"), std::string::npos);
  EXPECT_EQ(logs.find("Core AI cache bookmark saved"), std::string::npos);
}

TEST_F(CoreAILifecycleLogTest, FinalBookmarkFlushFailureDoesNotLogDurableSave) {
  BookmarkLoaderScope fake;
  {
    auto locked = lock_bookmark(root.url.path, key);
    ASSERT_TRUE(locked.ok());
    ASSERT_EQ(
        ensure_bookmark_identity(*locked.get(), manifest, @"macOS", @"arch_b"),
        Error::Ok);
  }
  StorageFaultScope fault(StorageOperation::FullSync);
  fault.match_index = 5;
  const auto logs =
      capture([&] { return acquire(fake.loader, &data); }, Error::AccessFailed);
  EXPECT_EQ(fault.hits, 1);
  EXPECT_NE(logs.find("Core AI specialization succeeded:"), std::string::npos);
  EXPECT_EQ(logs.find("Core AI cache bookmark saved"), std::string::npos);
  EXPECT_TRUE([saved() isEqual:fake.loader.lastAcquired.bookmark]);
}

TEST_F(CoreAILifecycleLogTest, ClearLogsConfirmedAbsenceBeforeLocalCompletion) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  const auto logs = capture(
      [&] { return clear_bookmark_assets(root.url.path, key, fake.loader); });
  expect_order(
      logs,
      {
          "Core AI cache clear started:",
          "Core AI SDK eviction requested:",
          "Core AI SDK cache entry absence confirmed:",
          "Core AI local cache entry cleared:",
      });
  EXPECT_EQ(saved(), nil);
}

TEST_F(CoreAILifecycleLogTest, FailedEvictionWithConfirmedMissLogsAbsence) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  fake.loader.evictError = [NSError errorWithDomain:NSPOSIXErrorDomain
                                               code:ENOENT
                                           userInfo:nil];
  fake.loader.restoreMiss = YES;
  const auto logs = capture(
      [&] { return clear_bookmark_assets(root.url.path, key, fake.loader); });
  expect_order(
      logs,
      {
          "Core AI SDK eviction requested:",
          "Core AI SDK cache entry absence confirmed:",
          "Core AI local cache entry cleared:",
      });
}

TEST_F(CoreAILifecycleLogTest, EvictionErrorDoesNotLogAbsenceOrCompletion) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  fake.loader.evictError = [NSError errorWithDomain:NSPOSIXErrorDomain
                                               code:EBUSY
                                           userInfo:nil];
  const auto logs = capture(
      [&] { return clear_bookmark_assets(root.url.path, key, fake.loader); },
      Error::Internal);
  EXPECT_NE(logs.find("Core AI SDK eviction requested:"), std::string::npos);
  EXPECT_EQ(
      logs.find("Core AI SDK cache entry absence confirmed:"),
      std::string::npos);
  EXPECT_EQ(logs.find("Core AI local cache entry cleared:"), std::string::npos);
  EXPECT_NE(saved(), nil);
}

TEST_F(
    CoreAILifecycleLogTest,
    IdentityRemovalFailureDoesNotLogLocalCompletion) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  StorageFaultScope fault(StorageOperation::Remove);
  fault.match_index = 2;
  const auto logs = capture(
      [&] { return clear_bookmark_assets(root.url.path, key, fake.loader); },
      Error::AccessFailed);
  EXPECT_EQ(fault.hits, 1);
  EXPECT_NE(
      logs.find("Core AI SDK cache entry absence confirmed:"),
      std::string::npos);
  EXPECT_EQ(logs.find("Core AI local cache entry cleared:"), std::string::npos);
  auto locked = lock_bookmark(root.url.path, key, false);
  ASSERT_TRUE(locked.ok());
  ASSERT_NE(locked.get(), nullptr);
  auto identity = read_bookmark_identity(*locked.get());
  ASSERT_TRUE(identity.ok());
  EXPECT_NE(identity.get(), nil);
}

TEST_F(CoreAILifecycleLogTest, AbsentEntryClearRemainsQuiet) {
  const auto logs =
      capture([&] { return clear_bookmark_assets(root.url.path, key, nil); });
  EXPECT_TRUE(logs.empty()) << logs;
}

} // namespace
