/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <future>
#include <thread>
#include "coreai_bookmark_fixture.h"
#include "coreai_fault_scope.h"
#include "coreai_filesystem_fixture.h"

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

namespace {
constexpr auto kWaitTimeout = std::chrono::seconds(5);

struct CacheEntryWaiter {
  std::future<void> attempting;
  std::future<Result<std::unique_ptr<CacheEntryLock>>> result;
};

CacheEntryWaiter start_cache_entry_waiter(
    NSString* path,
    NSString* key,
    bool existing_only = false) {
  std::promise<void> attempting;
  std::promise<Result<std::unique_ptr<CacheEntryLock>>> result;
  CacheEntryWaiter waiter{attempting.get_future(), result.get_future()};
  // Detached promise-backed work cannot block test teardown if acquisition
  // stalls after a failed assertion.
  std::thread([path,
               key,
               existing_only,
               attempting = std::move(attempting),
               result = std::move(result)]() mutable {
    @autoreleasepool {
      StorageFaultScope scope(FaultPoint::BeforeCacheEntryLock);
      scope.armed = false;
      auto* signal = &attempting;
      scope.observe = ^(FaultPoint point) {
        if (point == FaultPoint::BeforeCacheEntryLock) {
          signal->set_value();
        }
      };
      result.set_value(
          existing_only ? lock_existing_cache_entry(path, key)
                        : lock_cache_entry(path, key));
    }
  }).detach();
  return waiter;
}

class CoreAIBookmarkContentionTest : public ::testing::Test {
 protected:
  TestDirectory root;
  std::unique_ptr<SelectedBundle> bundle;
  NSString* key = nil;

  void SetUp() override {
    ASSERT_NE(root.url, nil);
    auto selected = bookmark_bundle();
    ASSERT_TRUE(selected.ok());
    bundle = std::make_unique<SelectedBundle>(selected.get());
    auto result = test_cache_key(*bundle);
    ASSERT_TRUE(result.ok());
    key = result.get();
  }

  NSURL* lock_url() {
    return test_cache_lock_url(root.url, key);
  }
};

TEST_F(
    CoreAIBookmarkContentionTest,
    ThreadContentionSerializesSameKeyButNotDifferentStripes) {
  auto holder = lock_cache_entry(root.url.path, key);
  ASSERT_TRUE(holder.ok());
  ASSERT_NE(holder.get(), nullptr);
  auto waiter = start_cache_entry_waiter(root.url.path, key);
  ASSERT_EQ(
      waiter.attempting.wait_for(kWaitTimeout), std::future_status::ready);
  EXPECT_EQ(
      waiter.result.wait_for(std::chrono::milliseconds(100)),
      std::future_status::timeout);

  auto independent_key = different_cache_key(*bundle, key, false);
  ASSERT_TRUE(independent_key.ok());
  auto independent =
      start_cache_entry_waiter(root.url.path, independent_key.get());
  ASSERT_EQ(
      independent.result.wait_for(kWaitTimeout), std::future_status::ready);
  auto other = independent.result.get();
  ASSERT_TRUE(other.ok());
  ASSERT_NE(other.get(), nullptr);

  holder.get().reset();
  ASSERT_EQ(waiter.result.wait_for(kWaitTimeout), std::future_status::ready);
  auto acquired = waiter.result.get();
  ASSERT_TRUE(acquired.ok());
  EXPECT_NE(acquired.get(), nullptr);
}

TEST_F(CoreAIBookmarkContentionTest, CollidingKeysSerializeAcrossThreads) {
  auto collision = different_cache_key(*bundle, key, true);
  ASSERT_TRUE(collision.ok());
  auto holder = lock_cache_entry(root.url.path, key);
  ASSERT_TRUE(holder.ok());
  auto waiter = start_cache_entry_waiter(root.url.path, collision.get());
  ASSERT_EQ(
      waiter.attempting.wait_for(kWaitTimeout), std::future_status::ready);
  EXPECT_EQ(
      waiter.result.wait_for(std::chrono::milliseconds(100)),
      std::future_status::timeout);
  holder.get().reset();
  ASSERT_EQ(waiter.result.wait_for(kWaitTimeout), std::future_status::ready);
  auto acquired = waiter.result.get();
  ASSERT_TRUE(acquired.ok());
  EXPECT_NE(acquired.get(), nullptr);
}

TEST_F(
    CoreAIBookmarkContentionTest,
    MissingCollidingLookupDoesNotWaitOrCreateAnotherLock) {
  auto collision = different_cache_key(*bundle, key, true);
  ASSERT_TRUE(collision.ok());
  auto holder = lock_cache_entry(root.url.path, key);
  ASSERT_TRUE(holder.ok());
  auto waiter = start_cache_entry_waiter(root.url.path, collision.get(), true);
  EXPECT_EQ(waiter.result.wait_for(kWaitTimeout), std::future_status::ready);
  holder.get().reset();
  auto absent = waiter.result.get();
  ASSERT_TRUE(absent.ok());
  EXPECT_EQ(absent.get(), nullptr);
  NSArray* names = [NSFileManager.defaultManager
      contentsOfDirectoryAtPath:[root.url URLByAppendingPathComponent:@"locks"]
                                    .path
                          error:nil];
  ASSERT_NE(names, nil);
  EXPECT_EQ(names.count, 1u);
}

TEST_F(CoreAIBookmarkContentionTest, ExistingWaiterRechecksEvictedEntry) {
  auto holder = lock_cache_entry(root.url.path, key);
  ASSERT_TRUE(holder.ok());
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(holder.get()->write_bookmark(bytes), Error::Ok);
  auto waiter = start_cache_entry_waiter(root.url.path, key, true);
  ASSERT_EQ(
      waiter.attempting.wait_for(kWaitTimeout), std::future_status::ready);
  ASSERT_EQ(
      waiter.result.wait_for(std::chrono::milliseconds(100)),
      std::future_status::timeout);
  NSURL* bookmark = test_bookmark_url(root.url, key);
  ASSERT_EQ(unlink(bookmark.fileSystemRepresentation), 0);
  holder.get().reset();
  ASSERT_EQ(waiter.result.wait_for(kWaitTimeout), std::future_status::ready);
  auto absent = waiter.result.get();
  ASSERT_TRUE(absent.ok());
  EXPECT_EQ(absent.get(), nullptr);
}

TEST_F(
    CoreAIBookmarkContentionTest,
    WaiterOpensBookmarkDirectoryAfterAcquiringLock) {
  auto holder = lock_cache_entry(root.url.path, key);
  ASSERT_TRUE(holder.ok());
  auto waiter = start_cache_entry_waiter(root.url.path, key);
  ASSERT_EQ(
      waiter.attempting.wait_for(kWaitTimeout), std::future_status::ready);
  ASSERT_EQ(
      waiter.result.wait_for(std::chrono::milliseconds(100)),
      std::future_status::timeout);
  NSURL* bookmarks = [root.url URLByAppendingPathComponent:@"bookmarks"];
  ASSERT_EQ(rmdir(bookmarks.fileSystemRepresentation), 0);
  ASSERT_EQ(mkdir(bookmarks.fileSystemRepresentation, 0700), 0);
  holder.get().reset();
  ASSERT_EQ(waiter.result.wait_for(kWaitTimeout), std::future_status::ready);
  auto acquired = waiter.result.get();
  ASSERT_TRUE(acquired.ok());
  ASSERT_NE(acquired.get(), nullptr);
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(acquired.get()->write_bookmark(bytes), Error::Ok);
  EXPECT_TRUE([[NSData dataWithContentsOfURL:test_bookmark_url(root.url, key)]
      isEqual:bytes]);
}

TEST_F(
    CoreAIBookmarkContentionTest,
    ProcessContentionAllowsIndependentKeyAndRecoversAfterHolderDeath) {
  BookmarkChild holder;
  ASSERT_TRUE(holder.spawn(@"--bookmark-hold", root.url.path, @"same"));
  ASSERT_TRUE(holder.receive('L'));
  NSURL* file = lock_url();
  struct stat before {};
  ASSERT_EQ(stat(file.fileSystemRepresentation, &before), 0);
  BookmarkChild waiter;
  ASSERT_TRUE(waiter.spawn(@"--bookmark-lock", root.url.path, @"same"));
  ASSERT_TRUE(waiter.receive('B'));
  ASSERT_TRUE(waiter.expect_blocked());
  BookmarkChild independent;
  ASSERT_TRUE(independent.spawn(@"--bookmark-lock", root.url.path, @"other"));
  ASSERT_TRUE(independent.receive('L'));
  ASSERT_TRUE(independent.expect_exit(0));
  ASSERT_TRUE(holder.kill_and_reap());
  ASSERT_TRUE(waiter.receive('L'));
  ASSERT_TRUE(waiter.expect_exit(0));
  struct stat after {};
  ASSERT_EQ(stat(file.fileSystemRepresentation, &after), 0);
  EXPECT_EQ(before.st_dev, after.st_dev);
  EXPECT_EQ(before.st_ino, after.st_ino);
}

TEST_F(
    CoreAIBookmarkContentionTest,
    CollidingProcessWaiterRecoversAfterHolderDeath) {
  BookmarkChild holder;
  ASSERT_TRUE(holder.spawn(@"--bookmark-hold", root.url.path, @"same"));
  ASSERT_TRUE(holder.receive('L'));
  struct stat before {};
  ASSERT_EQ(lstat(lock_url().fileSystemRepresentation, &before), 0);
  BookmarkChild waiter;
  ASSERT_TRUE(waiter.spawn(@"--bookmark-lock", root.url.path, @"collision"));
  ASSERT_TRUE(waiter.receive('B'));
  ASSERT_TRUE(waiter.expect_blocked());
  ASSERT_TRUE(holder.kill_and_reap());
  ASSERT_TRUE(waiter.receive('L'));
  ASSERT_TRUE(waiter.expect_exit(0));
  struct stat after {};
  ASSERT_EQ(lstat(lock_url().fileSystemRepresentation, &after), 0);
  EXPECT_EQ(before.st_dev, after.st_dev);
  EXPECT_EQ(before.st_ino, after.st_ino);
}

TEST_F(CoreAIBookmarkContentionTest, LockInventoryIsBoundedAndStable) {
  NSMutableDictionary<NSString*, NSString*>* representatives =
      [NSMutableDictionary dictionary];
  for (NSUInteger index = 0;
       index < 100000 && representatives.count < kCacheEntryLockStripeCount;
       ++index) {
    auto candidate = test_cache_key(
        *bundle, [NSString stringWithFormat:@"inventory-%lu", index]);
    ASSERT_TRUE(candidate.ok());
    auto name = cache_entry_lock_name(candidate.get());
    ASSERT_TRUE(name.ok());
    representatives[name.get()] = candidate.get();
  }
  ASSERT_EQ(representatives.count, kCacheEntryLockStripeCount);

  NSMutableDictionary<NSString*, NSNumber*>* inodes =
      [NSMutableDictionary dictionary];
  for (NSString* name in representatives) {
    auto locked = lock_cache_entry(root.url.path, representatives[name]);
    ASSERT_TRUE(locked.ok());
    NSURL* file = [[root.url URLByAppendingPathComponent:@"locks"]
        URLByAppendingPathComponent:name];
    struct stat info {};
    ASSERT_EQ(lstat(file.fileSystemRepresentation, &info), 0);
    ASSERT_TRUE(S_ISREG(info.st_mode));
    EXPECT_EQ(info.st_size, 0);
    inodes[name] = @(info.st_ino);
  }
  NSURL* locks = [root.url URLByAppendingPathComponent:@"locks"];
  NSArray<NSString*>* children =
      [NSFileManager.defaultManager contentsOfDirectoryAtPath:locks.path
                                                        error:nil];
  ASSERT_EQ(children.count, kCacheEntryLockStripeCount);

  for (NSString* name in representatives) {
    auto locked = lock_cache_entry(root.url.path, representatives[name]);
    ASSERT_TRUE(locked.ok());
    struct stat info {};
    NSURL* file = [locks URLByAppendingPathComponent:name];
    ASSERT_EQ(lstat(file.fileSystemRepresentation, &info), 0);
    EXPECT_EQ(info.st_ino, inodes[name].longLongValue);
  }
}

} // namespace
