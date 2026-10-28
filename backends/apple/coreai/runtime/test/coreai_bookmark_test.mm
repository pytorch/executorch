/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <array>
#include <chrono>
#include <future>
#include <thread>
#include "coreai_bookmark_fixture.h"
#include "coreai_fault_scope.h"
#include "coreai_filesystem_fixture.h"
#include "coreai_manifest_fixture.h"
#include "coreai_storage.h"

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

namespace {
constexpr size_t kMiB = 1024 * 1024;
constexpr auto kWaitTimeout = std::chrono::seconds(5);

NSString* changed_key(NSString* key, bool same_stripe) {
  NSMutableString* changed = [key mutableCopy];
  const NSUInteger index = same_stripe ? 63 : 0;
  [changed replaceCharactersInRange:NSMakeRange(index, 1)
                         withString:[key characterAtIndex:index] == '0' ? @"1"
                                                                        : @"0"];
  return changed;
}

struct BookmarkWaiter {
  std::future<void> attempting;
  std::future<Result<std::unique_ptr<BookmarkLock>>> result;
};

BookmarkWaiter
start_bookmark_waiter(NSString* path, NSString* key, bool create = true) {
  std::promise<void> attempting;
  std::promise<Result<std::unique_ptr<BookmarkLock>>> result;
  BookmarkWaiter waiter{attempting.get_future(), result.get_future()};
  // Promise-backed futures cannot block teardown if acquisition stalls.
  std::thread([path,
               key,
               create,
               attempting = std::move(attempting),
               result = std::move(result)]() mutable {
    @autoreleasepool {
      StorageFaultScope scope(StorageOperation::BeforeBookmarkLock);
      scope.armed = false;
      auto* signal = &attempting;
      scope.observe = ^(StorageOperation operation) {
        if (operation == StorageOperation::BeforeBookmarkLock) {
          signal->set_value();
        }
      };
      result.set_value(lock_bookmark(path, key, create));
    }
  }).detach();
  return waiter;
}

class CoreAIBookmarkIdentityTest : public ::testing::Test {
 protected:
  Manifest manifest;
  NSString* key = nil;

  void SetUp() override {
    auto selected = bookmark_manifest();
    ASSERT_TRUE(selected.ok());
    manifest = selected.get();
    auto result = test_bookmark_key(manifest);
    ASSERT_TRUE(result.ok());
    key = result.get();
  }
};

class CoreAIBookmarkPathTest : public CoreAIBookmarkIdentityTest {
 protected:
  TestDirectory root;
  void SetUp() override {
    CoreAIBookmarkIdentityTest::SetUp();
    if (HasFatalFailure())
      return;
    ASSERT_NE(root.url, nil);
  }
  NSURL* lock_url() {
    return test_bookmark_lock_url(root.url, key);
  }
};

class CoreAIBookmarkRawTest : public CoreAIBookmarkPathTest {
 protected:
  std::unique_ptr<BookmarkLock> locked;
  NSURL* url = nil;
  void SetUp() override {
    CoreAIBookmarkPathTest::SetUp();
    if (HasFatalFailure())
      return;
    auto result = lock_bookmark(root.url.path, key);
    ASSERT_TRUE(result.ok());
    locked = std::move(result.get());
    ASSERT_NE(locked, nullptr);
    url = test_bookmark_url(root.url, key);
  }
};

::testing::AssertionResult bookmark_bytes_equal(
    const BookmarkLock& lock,
    NSData* expected) {
  auto result = read_bookmark(lock);
  if (!result.ok())
    return ::testing::AssertionFailure()
        << "Bookmark read error " << int(result.error());
  if (expected == nil ? result.get() != nil
                      : ![result.get() isEqual:expected]) {
    return ::testing::AssertionFailure() << "Bookmark bytes differ";
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult no_temporary_bookmarks(NSURL* root) {
  NSURL* directory = [root URLByAppendingPathComponent:@"bookmarks"];
  FileDescriptor fd(open(
      directory.fileSystemRepresentation, O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (fd.get() < 0)
    return ::testing::AssertionFailure()
        << "open bookmark directory: " << errno;
  auto children = storage_children(fd.get());
  if (!children.ok())
    return ::testing::AssertionFailure()
        << "Cannot enumerate bookmark directory";
  for (NSString* name in children.get()) {
    if ([name hasPrefix:@".tmp-"])
      return ::testing::AssertionFailure() << "Leaked " << name.UTF8String;
  }
  return ::testing::AssertionSuccess();
}

TEST_F(CoreAIBookmarkIdentityTest, IgnoresFunctionBindingsAndDeliveryLocation) {
  EXPECT_EQ(key.length, 64u);
  manifest.function = @"another_function";
  manifest.inputs = @[ @"different_input" ];
  manifest.outputs = @[ @"different_output" ];
  manifest.hash = @"another_delivery_location";
  auto changed = test_bookmark_key(manifest);
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE([changed.get() isEqual:key]);
}

TEST_F(
    CoreAIBookmarkIdentityTest,
    KeySeparatesPlatformArchitectureWeightsAndAotContext) {
  auto platform = bookmark_key(manifest, @"iOS", @"arch_b");
  auto architecture = bookmark_key(manifest, @"macOS", @"arch_a");
  Manifest reweighted = manifest;
  reweighted.bundle_digests = @{@"model.aimodel" : fixture_identity('2')};
  auto weights = test_bookmark_key(reweighted);
  auto aot_manifest = bookmark_manifest(true);
  ASSERT_TRUE(aot_manifest.ok());
  auto aot = test_bookmark_key(aot_manifest.get());
  for (auto* changed : {&platform, &architecture, &weights, &aot}) {
    ASSERT_TRUE(changed->ok());
    EXPECT_FALSE([changed->get() isEqual:key]);
  }
}

TEST_F(CoreAIBookmarkIdentityTest, RejectsMissingArchitecture) {
  EXPECT_FALSE(bookmark_key(manifest, @"macOS", nil).ok());
  EXPECT_FALSE(bookmark_key(manifest, @"macOS", @"").ok());
}

TEST_F(CoreAIBookmarkIdentityTest, RejectsUnselectedBundle) {
  manifest.path = @"../escape";
  EXPECT_FALSE(test_bookmark_key(manifest).ok());
}

TEST_F(CoreAIBookmarkRawTest, MissingBookmarkIsAnEmptySuccessfulRead) {
  EXPECT_TRUE(bookmark_bytes_equal(*locked, nil));
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:url.path]);
}

TEST_F(CoreAIBookmarkRawTest, RoundTripsMaximumBookmarkAndReopens) {
  auto bytes = sized_bookmark(8 * kMiB);
  ASSERT_TRUE(bytes.ok());
  ASSERT_EQ(write_bookmark(*locked, bytes.get()), Error::Ok);
  EXPECT_TRUE(bookmark_bytes_equal(*locked, bytes.get()));
  EXPECT_TRUE([[NSData dataWithContentsOfURL:url] isEqual:bytes.get()]);
  locked.reset();
  auto saved = saved_bookmark(root.url, key);
  ASSERT_TRUE(saved.ok());
  EXPECT_TRUE([saved.get() isEqual:bytes.get()]);
}

TEST_F(CoreAIBookmarkRawTest, InvalidWritesPreserveMaximumBookmark) {
  auto maximum = sized_bookmark(8 * kMiB);
  ASSERT_TRUE(maximum.ok());
  ASSERT_EQ(write_bookmark(*locked, maximum.get()), Error::Ok);
  auto oversized = sized_bookmark(8 * kMiB + 1);
  ASSERT_TRUE(oversized.ok());
  for (NSData* invalid : {[NSData data], oversized.get()}) {
    SCOPED_TRACE(invalid.length);
    EXPECT_EQ(write_bookmark(*locked, invalid), Error::InvalidExternalData);
    EXPECT_TRUE([[NSData dataWithContentsOfURL:url] isEqual:maximum.get()]);
    EXPECT_TRUE(bookmark_bytes_equal(*locked, maximum.get()));
    EXPECT_TRUE(no_temporary_bookmarks(root.url));
  }
}

TEST_F(CoreAIBookmarkRawTest, RejectsEmptyStoredBookmark) {
  FileDescriptor file(open(
      url.fileSystemRepresentation,
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
      0600));
  ASSERT_GE(file.get(), 0);
  EXPECT_EQ(read_bookmark(*locked).error(), Error::InvalidExternalData);
}

TEST_F(CoreAIBookmarkRawTest, RejectsSymlinkedBookmarkWithoutReadingTarget) {
  NSURL* target = [root.url URLByAppendingPathComponent:@"target"];
  NSData* bytes = [@"target bytes" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_TRUE([bytes writeToURL:target atomically:NO]);
  ASSERT_EQ(
      symlink(target.fileSystemRepresentation, url.fileSystemRepresentation),
      0);
  EXPECT_EQ(read_bookmark(*locked).error(), Error::AccessFailed);
  EXPECT_TRUE(([[NSData dataWithContentsOfURL:target] isEqual:bytes]));
}

TEST_F(CoreAIBookmarkRawTest, RejectsDirectoryAtBookmarkPath) {
  ASSERT_EQ(mkdir(url.fileSystemRepresentation, 0700), 0);
  EXPECT_EQ(read_bookmark(*locked).error(), Error::InvalidExternalData);
}

TEST_F(CoreAIBookmarkRawTest, RejectsFifoWithoutWaitingForWriter) {
  ASSERT_EQ(mkfifo(url.fileSystemRepresentation, 0600), 0);
  EXPECT_EQ(read_bookmark(*locked).error(), Error::InvalidExternalData);
}

TEST_F(CoreAIBookmarkRawTest, RejectsInaccessibleBookmark) {
  if (geteuid() == 0) {
    GTEST_SKIP() << "Root can bypass file read permissions";
  }
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(write_bookmark(*locked, bytes), Error::Ok);
  ASSERT_EQ(chmod(url.fileSystemRepresentation, 0000), 0);
  EXPECT_EQ(read_bookmark(*locked).error(), Error::AccessFailed);
  ASSERT_EQ(chmod(url.fileSystemRepresentation, 0600), 0);
  EXPECT_TRUE(bookmark_bytes_equal(*locked, bytes));
}

TEST_F(CoreAIBookmarkRawTest, NoncreatingLookupReadsExistingBookmark) {
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(write_bookmark(*locked, bytes), Error::Ok);
  locked.reset();
  auto existing = lock_bookmark(root.url.path, key, false);
  ASSERT_TRUE(existing.ok());
  ASSERT_NE(existing.get(), nullptr);
  EXPECT_TRUE(bookmark_bytes_equal(*existing.get(), bytes));
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[root.url URLByAppendingPathComponent:@"staging"].path]);
}

TEST_F(CoreAIBookmarkRawTest, PreparesSharedStagingParentOnDemand) {
  NSString* expected = [root.url URLByAppendingPathComponent:@"staging"].path;
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:expected]);
  auto staging = prepare_bookmark_staging(*locked);
  ASSERT_TRUE(staging.ok());
  EXPECT_TRUE([staging.get() isEqual:expected]);
  struct stat info {};
  ASSERT_EQ(lstat(staging.get().fileSystemRepresentation, &info), 0);
  EXPECT_TRUE(S_ISDIR(info.st_mode));
  auto repeated = prepare_bookmark_staging(*locked);
  ASSERT_TRUE(repeated.ok());
  EXPECT_TRUE([repeated.get() isEqual:expected]);
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[expected stringByAppendingPathComponent:key]]);
}

TEST_F(CoreAIBookmarkRawTest, CreatesDirectoriesWithDefaultPermissions) {
  NSURL* reference = [root.url URLByAppendingPathComponent:@"reference"];
  ASSERT_EQ(mkdir(reference.fileSystemRepresentation, 0777), 0);
  struct stat expected;
  ASSERT_EQ(stat(reference.fileSystemRepresentation, &expected), 0);
  ASSERT_TRUE(prepare_bookmark_staging(*locked).ok());
  for (NSString* name in @[ @"locks", @"bookmarks", @"staging" ]) {
    NSURL* directory = [root.url URLByAppendingPathComponent:name];
    struct stat actual;
    ASSERT_EQ(stat(directory.fileSystemRepresentation, &actual), 0);
    EXPECT_EQ(actual.st_mode & 0777, expected.st_mode & 0777);
  }
}

TEST_F(CoreAIBookmarkRawTest, ReusesDirectoriesWithoutCreation) {
  ASSERT_TRUE(prepare_bookmark_staging(*locked).ok());
  ASSERT_EQ(
      write_bookmark(
          *locked, [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding]),
      Error::Ok);
  locked.reset();
  StorageFaultScope fault(StorageOperation::BeforeCreateDirectory);
  for (bool create : {true, false}) {
    auto reused = lock_bookmark(root.url.path, key, create);
    ASSERT_TRUE(reused.ok());
    ASSERT_NE(reused.get(), nullptr);
    EXPECT_TRUE(prepare_bookmark_staging(*reused.get()).ok());
  }
  EXPECT_EQ(fault.matches, 0);
}

TEST_F(CoreAIBookmarkRawTest, RecreatesMissingStagingParent) {
  auto staging = prepare_bookmark_staging(*locked);
  ASSERT_TRUE(staging.ok());
  ASSERT_EQ(rmdir(staging.get().fileSystemRepresentation), 0);
  StorageFaultScope fault(StorageOperation::BeforeCreateDirectory);
  EXPECT_EQ(prepare_bookmark_staging(*locked).error(), Error::AccessFailed);
  EXPECT_EQ(fault.hits, 1);
  auto recreated = prepare_bookmark_staging(*locked);
  ASSERT_TRUE(recreated.ok());
  EXPECT_TRUE([recreated.get() isEqual:staging.get()]);
}

TEST_F(CoreAIBookmarkRawTest, AcceptsConcurrentStagingCreation) {
  NSURL* staging = [root.url URLByAppendingPathComponent:@"staging"];
  StorageFaultScope fault(StorageOperation::BeforeCreateDirectory);
  fault.armed = false;
  __block int creations = 0;
  fault.observe = ^(StorageOperation operation) {
    if (operation == StorageOperation::BeforeCreateDirectory) {
      ++creations;
      EXPECT_EQ(mkdir(staging.fileSystemRepresentation, 0700), 0);
    }
  };
  EXPECT_TRUE(prepare_bookmark_staging(*locked).ok());
  EXPECT_EQ(creations, 1);
}

TEST_F(CoreAIBookmarkRawTest, RejectsSymlinkedStagingParent) {
  TestDirectory target;
  ASSERT_NE(target.url, nil);
  NSURL* staging = [root.url URLByAppendingPathComponent:@"staging"];
  ASSERT_EQ(
      symlink(
          target.url.fileSystemRepresentation,
          staging.fileSystemRepresentation),
      0);
  EXPECT_EQ(prepare_bookmark_staging(*locked).error(), Error::AccessFailed);
  NSArray* contents =
      [NSFileManager.defaultManager contentsOfDirectoryAtPath:target.url.path
                                                        error:nil];
  ASSERT_NE(contents, nil);
  EXPECT_EQ(contents.count, 0u);
}

TEST_F(CoreAIBookmarkRawTest, RejectsFileAtStagingParent) {
  NSURL* staging = [root.url URLByAppendingPathComponent:@"staging"];
  NSData* bytes = [@"not a directory" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_TRUE([bytes writeToURL:staging atomically:NO]);
  EXPECT_EQ(prepare_bookmark_staging(*locked).error(), Error::AccessFailed);
  EXPECT_TRUE(([[NSData dataWithContentsOfURL:staging] isEqual:bytes]));
}

TEST_F(CoreAIBookmarkRawTest, RejectsOversizedStoredBookmark) {
  FileDescriptor file(open(
      url.fileSystemRepresentation,
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
      0600));
  ASSERT_GE(file.get(), 0);
  ASSERT_EQ(ftruncate(file.get(), 8 * kMiB + 1), 0);
  ASSERT_EQ(close(file.release()), 0);
  EXPECT_EQ(read_bookmark(*locked).error(), Error::InvalidExternalData);
  struct stat info {};
  ASSERT_EQ(lstat(url.fileSystemRepresentation, &info), 0);
  EXPECT_EQ(info.st_size, static_cast<off_t>(8 * kMiB + 1));
}

TEST_F(CoreAIBookmarkPathTest, AllowsTemporaryRootsAndRejectsInvalidKeys) {
  auto allowed = lock_bookmark(root.url.path, key);
  ASSERT_TRUE(allowed.ok());
  ASSERT_NE(allowed.get(), nullptr);
  EXPECT_FALSE(lock_bookmark(root.url.path, @"../escape").ok());
}

TEST_F(CoreAIBookmarkPathTest, ExcludesRootWithoutCreatingStaging) {
  auto locked = lock_bookmark(root.url.path, key);
  ASSERT_TRUE(locked.ok());
  EXPECT_TRUE(backup_excluded(root.url));
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[root.url URLByAppendingPathComponent:@"staging"].path]);
}

TEST_F(
    CoreAIBookmarkPathTest,
    NoncreatingLookupLeavesMissingRootAndEntryUntouched) {
  ASSERT_TRUE(set_backup_excluded(root.url, false));
  NSString* missing = [root.url URLByAppendingPathComponent:@"missing"].path;
  for (NSString* path in @[ root.url.path, missing ]) {
    SCOPED_TRACE(path.UTF8String);
    auto resolved = resolve_bookmark_root(path, false);
    ASSERT_TRUE(resolved.ok());
    EXPECT_TRUE([resolved.get() isEqual:path]);
    auto absent = lock_bookmark(path, key, false);
    ASSERT_TRUE(absent.ok());
    EXPECT_EQ(absent.get(), nullptr);
  }
  NSArray* contents =
      [NSFileManager.defaultManager contentsOfDirectoryAtPath:root.url.path
                                                        error:nil];
  ASSERT_NE(contents, nil);
  EXPECT_EQ(contents.count, 0u);
  EXPECT_FALSE(backup_excluded(root.url));
}

TEST_F(CoreAIBookmarkPathTest, NoncreatingLookupIgnoresOtherEntries) {
  NSString* other_key = changed_key(key, false);
  auto other = lock_bookmark(root.url.path, other_key);
  ASSERT_TRUE(other.ok());
  ASSERT_NE(other.get(), nullptr);
  NSData* bytes = [@"other" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(write_bookmark(*other.get(), bytes), Error::Ok);
  auto absent = lock_bookmark(root.url.path, key, false);
  ASSERT_TRUE(absent.ok());
  EXPECT_EQ(absent.get(), nullptr);
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:lock_url().path]);
  EXPECT_TRUE(bookmark_bytes_equal(*other.get(), bytes));
}

TEST_F(CoreAIBookmarkPathTest, NoncreatingLookupLocksStagingOnlyEntry) {
  NSURL* staged = [[root.url URLByAppendingPathComponent:@"staging"]
      URLByAppendingPathComponent:key];
  ASSERT_TRUE([NSFileManager.defaultManager createDirectoryAtURL:staged
                                     withIntermediateDirectories:YES
                                                      attributes:nil
                                                           error:nil]);
  auto existing = lock_bookmark(root.url.path, key, false);
  ASSERT_TRUE(existing.ok());
  ASSERT_NE(existing.get(), nullptr);
  EXPECT_TRUE(bookmark_bytes_equal(*existing.get(), nil));
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  EXPECT_EQ(write_bookmark(*existing.get(), bytes), Error::InvalidArgument);
  EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:lock_url().path]);
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[root.url URLByAppendingPathComponent:@"bookmarks"]
                           .path]);
}

TEST_F(CoreAIBookmarkPathTest, ResolvesAndCreatesRootWithBackupExclusion) {
  ASSERT_TRUE(set_backup_excluded(root.url, false));
  NSString* path = [root.url URLByAppendingPathComponent:@"nested/assets"].path;
  auto resolved = resolve_bookmark_root(path);
  ASSERT_TRUE(resolved.ok());
  EXPECT_TRUE([resolved.get() isEqual:path]);
  EXPECT_TRUE(backup_excluded([NSURL fileURLWithPath:resolved.get()]));
  EXPECT_FALSE(backup_excluded(root.url));
  auto existing = resolve_bookmark_root(path, false);
  ASSERT_TRUE(existing.ok());
  EXPECT_TRUE([existing.get() isEqual:path]);
}

TEST_F(CoreAIBookmarkPathTest, RejectsInvalidResolvedRoots) {
  for (bool create : {false, true}) {
    for (NSString* path in
         @[ @"relative", @"~", @"~/cache", @"~nobody/cache" ]) {
      SCOPED_TRACE(path.UTF8String);
      EXPECT_EQ(
          resolve_bookmark_root(path, create).error(), Error::InvalidArgument);
    }
    EXPECT_EQ(
        resolve_bookmark_root(nil, create).error(), Error::InvalidArgument);
  }
  NSURL* file = [root.url URLByAppendingPathComponent:@"file"];
  ASSERT_TRUE([[NSData data] writeToURL:file atomically:NO]);
  EXPECT_EQ(
      resolve_bookmark_root(file.path, false).error(), Error::AccessFailed);
  EXPECT_EQ(
      resolve_bookmark_root(
          [file.path stringByAppendingPathComponent:@"child"], false)
          .error(),
      Error::AccessFailed);
}

TEST_F(
    CoreAIBookmarkPathTest,
    ThreadContentionSerializesSameKeyButNotDifferentStripes) {
  auto holder = lock_bookmark(root.url.path, key);
  ASSERT_TRUE(holder.ok());
  ASSERT_NE(holder.get(), nullptr);
  auto waiter = start_bookmark_waiter(root.url.path, key);
  ASSERT_EQ(
      waiter.attempting.wait_for(kWaitTimeout), std::future_status::ready);
  EXPECT_EQ(
      waiter.result.wait_for(std::chrono::milliseconds(100)),
      std::future_status::timeout);
  auto independent =
      start_bookmark_waiter(root.url.path, changed_key(key, false));
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

TEST_F(CoreAIBookmarkRawTest, NoncreatingWaiterRechecksEvictedEntry) {
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(write_bookmark(*locked, bytes), Error::Ok);
  auto waiter = start_bookmark_waiter(root.url.path, key, false);
  ASSERT_EQ(
      waiter.attempting.wait_for(kWaitTimeout), std::future_status::ready);
  ASSERT_EQ(
      waiter.result.wait_for(std::chrono::milliseconds(100)),
      std::future_status::timeout);
  ASSERT_EQ(unlink(url.fileSystemRepresentation), 0);
  locked.reset();
  ASSERT_EQ(waiter.result.wait_for(kWaitTimeout), std::future_status::ready);
  auto absent = waiter.result.get();
  ASSERT_TRUE(absent.ok());
  EXPECT_EQ(absent.get(), nullptr);
}

TEST_F(CoreAIBookmarkRawTest, WaiterOpensBookmarkDirectoryAfterAcquiringLock) {
  auto waiter = start_bookmark_waiter(root.url.path, key);
  ASSERT_EQ(
      waiter.attempting.wait_for(kWaitTimeout), std::future_status::ready);
  ASSERT_EQ(
      waiter.result.wait_for(std::chrono::milliseconds(100)),
      std::future_status::timeout);
  NSURL* bookmarks = [root.url URLByAppendingPathComponent:@"bookmarks"];
  ASSERT_EQ(rmdir(bookmarks.fileSystemRepresentation), 0);
  ASSERT_EQ(mkdir(bookmarks.fileSystemRepresentation, 0700), 0);
  locked.reset();
  ASSERT_EQ(waiter.result.wait_for(kWaitTimeout), std::future_status::ready);
  auto acquired = waiter.result.get();
  ASSERT_TRUE(acquired.ok());
  ASSERT_NE(acquired.get(), nullptr);
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(write_bookmark(*acquired.get(), bytes), Error::Ok);
  EXPECT_TRUE([[NSData dataWithContentsOfURL:url] isEqual:bytes]);
}

TEST_F(
    CoreAIBookmarkPathTest,
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

TEST_F(CoreAIBookmarkPathTest, LockNamesUseOnlyValidatedFirstByte) {
  EXPECT_EQ(bookmark_lock_name(nil).error(), Error::InvalidArgument);
  for (NSString* invalid in
       @[ @"", @"../escape", @"00", [fixture_identity('a') uppercaseString] ]) {
    EXPECT_EQ(bookmark_lock_name(invalid).error(), Error::InvalidArgument);
  }
  auto first = bookmark_lock_name(fixture_identity('0'));
  auto last = bookmark_lock_name(fixture_identity('f'));
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(last.ok());
  EXPECT_TRUE([first.get() isEqual:@"stripe-v1-00.lock"]);
  EXPECT_TRUE([last.get() isEqual:@"stripe-v1-ff.lock"]);
  EXPECT_TRUE([test_bookmark_lock_url(root.url, key)
      isEqual:test_bookmark_lock_url(root.url, changed_key(key, true))]);
}

TEST_F(CoreAIBookmarkPathTest, CollidingKeysSerializeAcrossThreads) {
  auto holder = lock_bookmark(root.url.path, key);
  ASSERT_TRUE(holder.ok());
  NSString* collision = changed_key(key, true);
  ASSERT_FALSE([collision isEqual:key]);
  auto waiter = start_bookmark_waiter(root.url.path, collision);
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
    CoreAIBookmarkPathTest,
    NoncreatingCollisionMissDoesNotWaitOrCreateFiles) {
  auto holder = lock_bookmark(root.url.path, key);
  ASSERT_TRUE(holder.ok());
  auto waiter =
      start_bookmark_waiter(root.url.path, changed_key(key, true), false);
  EXPECT_EQ(waiter.result.wait_for(kWaitTimeout), std::future_status::ready);
  holder.get().reset();
  ASSERT_EQ(waiter.result.wait_for(kWaitTimeout), std::future_status::ready);
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

TEST_F(CoreAIBookmarkPathTest, RejectsSymlinkDirectoryAndFifoStripeFiles) {
  NSData* bytes = [@"untouched target" dataUsingEncoding:NSUTF8StringEncoding];
  NSURL* target = [root.url URLByAppendingPathComponent:@"target"];
  ASSERT_TRUE([bytes writeToURL:target atomically:NO]);
  for (int kind = 0; kind < 3; ++kind) {
    TestDirectory isolated;
    ASSERT_NE(isolated.url, nil);
    NSURL* locks = [isolated.url URLByAppendingPathComponent:@"locks"];
    ASSERT_EQ(mkdir(locks.fileSystemRepresentation, 0700), 0);
    NSURL* stripe = test_bookmark_lock_url(isolated.url, key);
    if (kind == 0)
      ASSERT_EQ(
          symlink(
              target.fileSystemRepresentation, stripe.fileSystemRepresentation),
          0);
    else if (kind == 1)
      ASSERT_EQ(mkdir(stripe.fileSystemRepresentation, 0700), 0);
    else
      ASSERT_EQ(mkfifo(stripe.fileSystemRepresentation, 0600), 0);
    EXPECT_EQ(
        lock_bookmark(isolated.url.path, key).error(), Error::AccessFailed);
    EXPECT_TRUE([[NSData dataWithContentsOfURL:target] isEqual:bytes]);
  }
}

TEST_F(CoreAIBookmarkPathTest, LegacyLockFilesAreLeftUntouched) {
  NSURL* locks = [root.url URLByAppendingPathComponent:@"locks"];
  ASSERT_EQ(mkdir(locks.fileSystemRepresentation, 0700), 0);
  NSURL* legacy = [locks
      URLByAppendingPathComponent:[key stringByAppendingString:@".lock"]];
  NSData* bytes = [@"legacy" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_TRUE([bytes writeToURL:legacy atomically:NO]);
  struct stat before {};
  ASSERT_EQ(lstat(legacy.fileSystemRepresentation, &before), 0);
  auto locked = lock_bookmark(root.url.path, key);
  ASSERT_TRUE(locked.ok());
  locked.get().reset();
  struct stat after {};
  ASSERT_EQ(lstat(legacy.fileSystemRepresentation, &after), 0);
  EXPECT_EQ(before.st_ino, after.st_ino);
  EXPECT_TRUE([[NSData dataWithContentsOfURL:legacy] isEqual:bytes]);
  EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:lock_url().path]);
}

TEST_F(CoreAIBookmarkPathTest, CollidingProcessWaiterRecoversAfterHolderDeath) {
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

TEST_F(CoreAIBookmarkPathTest, LockPoolIsBoundedAcrossManyKeys) {
  std::array<ino_t, kBookmarkLockStripeCount> inodes{};
  NSURL* locks = [root.url URLByAppendingPathComponent:@"locks"];
  for (size_t generation = 0; generation < 4; ++generation) {
    for (size_t stripe = 0; stripe < kBookmarkLockStripeCount; ++stripe) {
      @autoreleasepool {
        NSString* entry =
            [NSString stringWithFormat:@"%02zx%062zx", stripe, generation];
        auto locked = lock_bookmark(root.url.path, entry);
        ASSERT_TRUE(locked.ok());
        struct stat info {};
        NSURL* file = test_bookmark_lock_url(root.url, entry);
        ASSERT_EQ(lstat(file.fileSystemRepresentation, &info), 0);
        ASSERT_TRUE(S_ISREG(info.st_mode));
        EXPECT_EQ(info.st_size, 0);
        if (generation == 0)
          inodes[stripe] = info.st_ino;
        else
          EXPECT_EQ(info.st_ino, inodes[stripe]);
        locked.get().reset();
        auto reused = lock_bookmark(root.url.path, entry);
        ASSERT_TRUE(reused.ok());
        ASSERT_EQ(lstat(file.fileSystemRepresentation, &info), 0);
        EXPECT_EQ(info.st_ino, inodes[stripe]);
      }
    }
    NSArray* children =
        [NSFileManager.defaultManager contentsOfDirectoryAtPath:locks.path
                                                          error:nil];
    ASSERT_NE(children, nil);
    ASSERT_EQ(children.count, kBookmarkLockStripeCount);
    for (NSString* name in children) {
      EXPECT_TRUE([name hasPrefix:@"stripe-v1-"]);
      EXPECT_EQ(name.length, 17u);
    }
  }
}

} // namespace
