/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_acquisition_fixture.h"

#include <cerrno>
#include "coreai_fault_scope.h"
#include "coreai_load_coordinator.h"

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

namespace {

class CoreAIAcquisitionTest : public ::testing::Test {
 protected:
  // Declared first so each case's loader, models and data die before it.
  ScopedFakeBridgeState bridge;
  BookmarkDirectory root;
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
      auto model =
          acquire_bookmark_model(manifest, named_data, root.url.path, @"macOS", @"arch_b", loader);
      if (!model.ok()) return model.error();
      return model.get() != nil ? Error::Ok : Error::Internal;
    }
  }

  NSData* saved() {
    auto bookmark = saved_bookmark(root.url, key);
    EXPECT_TRUE(bookmark.ok());
    return bookmark.ok() ? bookmark.get() : nil;
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
}

TEST_F(CoreAIAcquisitionTest, WarmHitRestoresWithoutNamedData) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  NSData* original = saved();
  ASSERT_EQ(acquire(fake.loader, nullptr), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), 1);
  EXPECT_EQ(fake.loader->restores.load(), 1);
  EXPECT_TRUE([saved() isEqual:original]);
}

TEST_F(CoreAIAcquisitionTest, RestoreErrorFailsClosedWithoutSourceFallback) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  NSData* original = saved();
  const int attempts = data.attempts;
  fake.loader.restoreError = [NSError errorWithDomain:NSPOSIXErrorDomain code:EIO userInfo:nil];
  EXPECT_EQ(acquire(fake.loader, &data), Error::Internal);
  EXPECT_EQ(fake.loader->specializations.load(), 1);
  EXPECT_EQ(data.attempts.load(), attempts);
  EXPECT_TRUE([saved() isEqual:original]);
}

TEST_F(CoreAIAcquisitionTest, RestoreMissRespecializesStagedSourceAndReplacesBookmark) {
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  NSData* original = saved();
  const int attempts = data.attempts;
  fake.loader.restoreMiss = YES;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), 2);
  EXPECT_EQ(data.attempts.load(), attempts);
  NSData* replaced = saved();
  EXPECT_NE(replaced, nil);
  EXPECT_FALSE([replaced isEqual:original]);
}

class CoreAIAcquisitionSaveFailureTest : public CoreAIAcquisitionTest,
                                         public ::testing::WithParamInterface<bool> {};
TEST_P(CoreAIAcquisitionSaveFailureTest, PreservesBookmarkAndRetries) {
  const bool prior = GetParam();
  BookmarkLoaderScope fake;
  NSData* original = nil;
  if (prior) {
    ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
    original = saved();
    ASSERT_NE(original, nil);
    fake.loader.restoreMiss = YES;
  }
  {
    // A cold acquisition publishes its staged source before the bookmark.
    StorageFaultScope fault(StorageOperation::Rename);
    fault.match_index = prior ? 1 : 2;
    EXPECT_EQ(acquire(fake.loader, &data), Error::AccessFailed);
    EXPECT_EQ(fault.hits, 1);
  }
  EXPECT_TRUE(original == nil ? saved() == nil : [saved() isEqual:original]);
  EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:fake.loader.lastSpecializeURL]);
  const int calls = fake.loader->specializations;
  fake.loader.restoreMiss = NO;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), calls + (prior ? 0 : 1));
}
INSTANTIATE_TEST_SUITE_P(Boundaries, CoreAIAcquisitionSaveFailureTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "PriorBookmark" : "Cold";
                         });

class CoreAIAcquisitionProcessTest : public CoreAIAcquisitionTest {
 protected:
  void expect_parallel(bool same_key) {
    NSMutableDictionary* first = manifest_dict();
    NSMutableDictionary* second = [first mutableCopy];
    if (!same_key) second[@"bundle_digests"] = @{@"model.aimodel" : fixture_identity('2')};
    BookmarkChild child;
    ASSERT_TRUE(spawn_parallel_acquisition(child, root.url, first, second, same_key));
    ASSERT_TRUE(child.receive('C'));
    ASSERT_TRUE(child.receive(1)) << "harness completed without timeouts";
    ASSERT_TRUE(child.receive(1)) << "second acquisition blocked or entered as expected";
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

TEST_F(CoreAIAcquisitionProcessTest, CrashAfterSdkRetainsStagingAndRetries) {
  BookmarkChild child;
  ASSERT_TRUE(spawn_acquisition_crash(child, root.url, StorageOperation::AfterSDK, manifest_dict()));
  ASSERT_TRUE(child.receive('C'));
  ASSERT_TRUE(child.expect_exit(73));
  EXPECT_EQ(saved(), nil);
  auto staged = acquisition_staged_url(root.url, manifest);
  ASSERT_TRUE(staged.ok());
  EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:staged.get().path]);
  BookmarkLoaderScope fake;
  ASSERT_EQ(acquire(fake.loader, &data), Error::Ok);
  EXPECT_EQ(fake.loader->specializations.load(), 1);
  EXPECT_EQ(fake.loader->restores.load(), 0);
  EXPECT_EQ(data.attempts.load(), 0);
  EXPECT_NE(saved(), nil);
}

}  // namespace
