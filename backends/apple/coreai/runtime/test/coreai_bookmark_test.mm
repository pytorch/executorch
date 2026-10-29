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
#include "coreai_bookmark_fixture.h"
#include "coreai_fault_scope.h"
#include "coreai_filesystem_fixture.h"
#include "coreai_storage.h"

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

namespace {
constexpr size_t kMiB = 1024 * 1024;

class CoreAIBookmarkPathTest : public ::testing::Test {
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

class CoreAIBookmarkIOTest : public CoreAIBookmarkPathTest {
 protected:
  std::unique_ptr<CacheEntryLock> locked;
  NSURL* url = nil;

  void SetUp() override {
    CoreAIBookmarkPathTest::SetUp();
    if (HasFatalFailure()) {
      return;
    }
    auto result = lock_cache_entry(root.url.path, key);
    ASSERT_TRUE(result.ok());
    locked = std::move(result.get());
    ASSERT_NE(locked, nullptr);
    ASSERT_EQ(locked->persistence(), CacheEntryPersistence::BookmarkCapable);
    url = test_bookmark_url(root.url, key);
  }
};

::testing::AssertionResult bookmark_bytes_equal(
    const CacheEntryLock& lock,
    NSData* expected) {
  auto result = lock.read_bookmark();
  if (!result.ok()) {
    return ::testing::AssertionFailure()
        << "Bookmark read error " << int(result.error());
  }
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
  if (fd.get() < 0) {
    return ::testing::AssertionFailure()
        << "open bookmark directory: " << errno;
  }
  auto children = storage_children(fd.get());
  if (!children.ok()) {
    return ::testing::AssertionFailure()
        << "Cannot enumerate bookmark directory";
  }
  for (NSString* name in children.get()) {
    if ([name hasPrefix:@".tmp-"]) {
      return ::testing::AssertionFailure() << "Leaked " << name.UTF8String;
    }
  }
  return ::testing::AssertionSuccess();
}

TEST_F(CoreAIBookmarkIOTest, MissingBookmarkIsAnEmptySuccessfulRead) {
  EXPECT_TRUE(bookmark_bytes_equal(*locked, nil));
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:url.path]);
}

TEST_F(CoreAIBookmarkIOTest, RoundTripsMaximumBookmarkAndReopens) {
  auto bytes = sized_bookmark(8 * kMiB);
  ASSERT_TRUE(bytes.ok());
  ASSERT_EQ(locked->write_bookmark(bytes.get()), Error::Ok);
  EXPECT_TRUE(bookmark_bytes_equal(*locked, bytes.get()));
  EXPECT_TRUE([[NSData dataWithContentsOfURL:url] isEqual:bytes.get()]);
  locked.reset();
  auto saved = saved_bookmark(root.url, key);
  ASSERT_TRUE(saved.ok());
  EXPECT_TRUE([saved.get() isEqual:bytes.get()]);
}

TEST_F(CoreAIBookmarkIOTest, InvalidWritesPreserveExistingBookmark) {
  auto maximum = sized_bookmark(8 * kMiB);
  ASSERT_TRUE(maximum.ok());
  ASSERT_EQ(locked->write_bookmark(maximum.get()), Error::Ok);
  auto oversized = sized_bookmark(8 * kMiB + 1);
  ASSERT_TRUE(oversized.ok());
  for (NSData* invalid : {[NSData data], oversized.get()}) {
    SCOPED_TRACE(invalid.length);
    EXPECT_EQ(locked->write_bookmark(invalid), Error::InvalidExternalData);
    EXPECT_TRUE([[NSData dataWithContentsOfURL:url] isEqual:maximum.get()]);
    EXPECT_TRUE(bookmark_bytes_equal(*locked, maximum.get()));
    EXPECT_TRUE(no_temporary_bookmarks(root.url));
  }
}

TEST_F(CoreAIBookmarkIOTest, RejectsEmptyStoredBookmark) {
  FileDescriptor file(open(
      url.fileSystemRepresentation,
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
      0600));
  ASSERT_GE(file.get(), 0);
  EXPECT_EQ(locked->read_bookmark().error(), Error::InvalidExternalData);
}

TEST_F(CoreAIBookmarkIOTest, RejectsSymlinkWithoutReadingTarget) {
  NSURL* target = [root.url URLByAppendingPathComponent:@"target"];
  NSData* bytes = [@"target bytes" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_TRUE([bytes writeToURL:target atomically:NO]);
  ASSERT_EQ(
      symlink(target.fileSystemRepresentation, url.fileSystemRepresentation),
      0);
  EXPECT_EQ(locked->read_bookmark().error(), Error::AccessFailed);
  EXPECT_TRUE([[NSData dataWithContentsOfURL:target] isEqual:bytes]);
}

TEST_F(CoreAIBookmarkIOTest, RejectsNonregularStoredBookmarks) {
  ASSERT_EQ(mkdir(url.fileSystemRepresentation, 0700), 0);
  EXPECT_EQ(locked->read_bookmark().error(), Error::InvalidExternalData);
  ASSERT_EQ(rmdir(url.fileSystemRepresentation), 0);
  ASSERT_EQ(mkfifo(url.fileSystemRepresentation, 0600), 0);
  EXPECT_EQ(locked->read_bookmark().error(), Error::InvalidExternalData);
}

TEST_F(CoreAIBookmarkIOTest, RejectsInaccessibleBookmark) {
  if (geteuid() == 0) {
    GTEST_SKIP() << "Root can bypass file read permissions";
  }
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(locked->write_bookmark(bytes), Error::Ok);
  ASSERT_EQ(chmod(url.fileSystemRepresentation, 0000), 0);
  EXPECT_EQ(locked->read_bookmark().error(), Error::AccessFailed);
  ASSERT_EQ(chmod(url.fileSystemRepresentation, 0600), 0);
  EXPECT_TRUE(bookmark_bytes_equal(*locked, bytes));
}

TEST_F(CoreAIBookmarkIOTest, RejectsOversizedStoredBookmark) {
  FileDescriptor file(open(
      url.fileSystemRepresentation,
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
      0600));
  ASSERT_GE(file.get(), 0);
  ASSERT_EQ(ftruncate(file.get(), 8 * kMiB + 1), 0);
  ASSERT_EQ(close(file.release()), 0);
  EXPECT_EQ(locked->read_bookmark().error(), Error::InvalidExternalData);
  struct stat info {};
  ASSERT_EQ(lstat(url.fileSystemRepresentation, &info), 0);
  EXPECT_EQ(info.st_size, static_cast<off_t>(8 * kMiB + 1));
}

TEST_F(CoreAIBookmarkIOTest, ExistingLookupReadsWithoutPreparingStaging) {
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(locked->write_bookmark(bytes), Error::Ok);
  locked.reset();
  auto existing = lock_existing_cache_entry(root.url.path, key);
  ASSERT_TRUE(existing.ok());
  ASSERT_NE(existing.get(), nullptr);
  EXPECT_EQ(
      existing.get()->persistence(), CacheEntryPersistence::BookmarkCapable);
  EXPECT_TRUE(bookmark_bytes_equal(*existing.get(), bytes));
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[root.url URLByAppendingPathComponent:@"staging"].path]);
}

TEST_F(CoreAIBookmarkIOTest, PreparesSharedStagingParentOnDemand) {
  NSString* expected = [root.url URLByAppendingPathComponent:@"staging"].path;
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:expected]);
  auto staging = locked->prepare_staging_root();
  ASSERT_TRUE(staging.ok());
  EXPECT_TRUE([staging.get() isEqual:expected]);
  struct stat info {};
  ASSERT_EQ(lstat(staging.get().fileSystemRepresentation, &info), 0);
  EXPECT_TRUE(S_ISDIR(info.st_mode));
  auto repeated = locked->prepare_staging_root();
  ASSERT_TRUE(repeated.ok());
  EXPECT_TRUE([repeated.get() isEqual:expected]);
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[expected stringByAppendingPathComponent:key]]);
}

TEST_F(CoreAIBookmarkIOTest, CreatesDirectoriesWithDefaultPermissions) {
  NSURL* reference = [root.url URLByAppendingPathComponent:@"reference"];
  ASSERT_EQ(mkdir(reference.fileSystemRepresentation, 0777), 0);
  struct stat expected;
  ASSERT_EQ(stat(reference.fileSystemRepresentation, &expected), 0);
  ASSERT_TRUE(locked->prepare_staging_root().ok());
  for (NSString* name in @[ @"locks", @"bookmarks", @"staging" ]) {
    NSURL* directory = [root.url URLByAppendingPathComponent:name];
    struct stat actual;
    ASSERT_EQ(stat(directory.fileSystemRepresentation, &actual), 0);
    EXPECT_EQ(actual.st_mode & 0777, expected.st_mode & 0777);
  }
}

TEST_F(CoreAIBookmarkIOTest, ReusesDirectoriesWithoutCreation) {
  ASSERT_TRUE(locked->prepare_staging_root().ok());
  ASSERT_EQ(
      locked->write_bookmark(
          [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding]),
      Error::Ok);
  locked.reset();
  StorageFaultScope fault(FaultPoint::BeforeCreateDirectory);
  auto created = lock_cache_entry(root.url.path, key);
  ASSERT_TRUE(created.ok());
  ASSERT_NE(created.get(), nullptr);
  EXPECT_TRUE(created.get()->prepare_staging_root().ok());
  created.get().reset();
  auto existing = lock_existing_cache_entry(root.url.path, key);
  ASSERT_TRUE(existing.ok());
  ASSERT_NE(existing.get(), nullptr);
  EXPECT_TRUE(existing.get()->prepare_staging_root().ok());
  EXPECT_EQ(fault.matches, 0);
}

TEST_F(CoreAIBookmarkIOTest, RecreatesMissingStagingParent) {
  auto staging = locked->prepare_staging_root();
  ASSERT_TRUE(staging.ok());
  ASSERT_EQ(rmdir(staging.get().fileSystemRepresentation), 0);
  StorageFaultScope fault(FaultPoint::BeforeCreateDirectory);
  EXPECT_EQ(locked->prepare_staging_root().error(), Error::AccessFailed);
  EXPECT_EQ(fault.hits, 1);
  auto recreated = locked->prepare_staging_root();
  ASSERT_TRUE(recreated.ok());
  EXPECT_TRUE([recreated.get() isEqual:staging.get()]);
}

TEST_F(CoreAIBookmarkIOTest, AcceptsConcurrentStagingCreation) {
  NSURL* staging = [root.url URLByAppendingPathComponent:@"staging"];
  StorageFaultScope fault(FaultPoint::BeforeCreateDirectory);
  fault.armed = false;
  __block int creations = 0;
  fault.observe = ^(FaultPoint point) {
    if (point == FaultPoint::BeforeCreateDirectory) {
      ++creations;
      EXPECT_EQ(mkdir(staging.fileSystemRepresentation, 0700), 0);
    }
  };
  EXPECT_TRUE(locked->prepare_staging_root().ok());
  EXPECT_EQ(creations, 1);
}

TEST_F(CoreAIBookmarkIOTest, RejectsInvalidStagingParent) {
  TestDirectory target;
  ASSERT_NE(target.url, nil);
  NSURL* staging = [root.url URLByAppendingPathComponent:@"staging"];
  ASSERT_EQ(
      symlink(
          target.url.fileSystemRepresentation,
          staging.fileSystemRepresentation),
      0);
  EXPECT_EQ(locked->prepare_staging_root().error(), Error::AccessFailed);
  ASSERT_EQ(unlink(staging.fileSystemRepresentation), 0);
  NSData* bytes = [@"not a directory" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_TRUE([bytes writeToURL:staging atomically:NO]);
  EXPECT_EQ(locked->prepare_staging_root().error(), Error::AccessFailed);
  EXPECT_TRUE([[NSData dataWithContentsOfURL:staging] isEqual:bytes]);
}

TEST_F(CoreAIBookmarkPathTest, AllowsTemporaryRootsAndRejectsInvalidKeys) {
  auto allowed = lock_cache_entry(root.url.path, key);
  ASSERT_TRUE(allowed.ok());
  ASSERT_NE(allowed.get(), nullptr);
  EXPECT_FALSE(lock_cache_entry(root.url.path, @"../escape").ok());
}

TEST_F(CoreAIBookmarkPathTest, SnapshotsMutableRootAtAcquisition) {
  TestDirectory other;
  ASSERT_NE(other.url, nil);
  NSMutableString* path = [root.url.path mutableCopy];
  auto locked = lock_cache_entry(path, key);
  ASSERT_TRUE(locked.ok());
  ASSERT_NE(locked.get(), nullptr);
  [path setString:other.url.path];

  auto staging = locked.get()->prepare_staging_root();
  ASSERT_TRUE(staging.ok());
  NSString* expected = [root.url URLByAppendingPathComponent:@"staging"].path;
  NSString* unrelated = [other.url URLByAppendingPathComponent:@"staging"].path;
  EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:expected]);
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:unrelated]);
  EXPECT_TRUE([staging.get() isEqual:expected])
      << "Returned staging path: " << staging.get().UTF8String
      << "; descriptor-pinned staging path: " << expected.UTF8String;
}

TEST_F(CoreAIBookmarkPathTest, ExcludesRootWithoutCreatingStaging) {
  auto locked = lock_cache_entry(root.url.path, key);
  ASSERT_TRUE(locked.ok());
  EXPECT_TRUE(backup_excluded(root.url));
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[root.url URLByAppendingPathComponent:@"staging"].path]);
}

TEST_F(CoreAIBookmarkPathTest, MissingLookupLeavesRootAndEntryUntouched) {
  ASSERT_TRUE(set_backup_excluded(root.url, false));
  NSString* missing = [root.url URLByAppendingPathComponent:@"missing"].path;
  for (NSString* path in @[ root.url.path, missing ]) {
    SCOPED_TRACE(path.UTF8String);
    auto resolved =
        resolve_bookmark_root(path, StorageCreationMode::OpenExisting);
    ASSERT_TRUE(resolved.ok());
    EXPECT_TRUE([resolved.get() isEqual:path]);
    auto absent = lock_existing_cache_entry(path, key);
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

TEST_F(CoreAIBookmarkPathTest, MissingLookupIgnoresOtherEntries) {
  auto other_key = different_cache_key(*bundle, key, false);
  ASSERT_TRUE(other_key.ok());
  auto other = lock_cache_entry(root.url.path, other_key.get());
  ASSERT_TRUE(other.ok());
  NSData* bytes = [@"other" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(other.get()->write_bookmark(bytes), Error::Ok);
  auto absent = lock_existing_cache_entry(root.url.path, key);
  ASSERT_TRUE(absent.ok());
  EXPECT_EQ(absent.get(), nullptr);
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:lock_url().path]);
  EXPECT_TRUE(bookmark_bytes_equal(*other.get(), bytes));
}

TEST_F(CoreAIBookmarkPathTest, StagingOnlyLookupHasExplicitPersistence) {
  NSURL* staged = [[root.url URLByAppendingPathComponent:@"staging"]
      URLByAppendingPathComponent:key];
  ASSERT_TRUE([NSFileManager.defaultManager createDirectoryAtURL:staged
                                     withIntermediateDirectories:YES
                                                      attributes:nil
                                                           error:nil]);
  auto existing = lock_existing_cache_entry(root.url.path, key);
  ASSERT_TRUE(existing.ok());
  ASSERT_NE(existing.get(), nullptr);
  EXPECT_EQ(existing.get()->persistence(), CacheEntryPersistence::StagingOnly);
  EXPECT_TRUE(bookmark_bytes_equal(*existing.get(), nil));
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  EXPECT_EQ(existing.get()->write_bookmark(bytes), Error::InvalidArgument);
  EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:lock_url().path]);
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[root.url URLByAppendingPathComponent:@"bookmarks"]
                           .path]);
}

TEST_F(CoreAIBookmarkPathTest, ResolvesAndCreatesRootWithBackupExclusion) {
  ASSERT_TRUE(set_backup_excluded(root.url, false));
  NSString* path = [root.url URLByAppendingPathComponent:@"nested/assets"].path;
  auto resolved = resolve_bookmark_root(path, StorageCreationMode::Create);
  ASSERT_TRUE(resolved.ok());
  EXPECT_TRUE([resolved.get() isEqual:path]);
  EXPECT_TRUE(backup_excluded([NSURL fileURLWithPath:resolved.get()]));
  EXPECT_FALSE(backup_excluded(root.url));
  auto existing =
      resolve_bookmark_root(path, StorageCreationMode::OpenExisting);
  ASSERT_TRUE(existing.ok());
  EXPECT_TRUE([existing.get() isEqual:path]);
}

TEST_F(CoreAIBookmarkPathTest, RejectsInvalidResolvedRoots) {
  for (StorageCreationMode mode :
       {StorageCreationMode::OpenExisting, StorageCreationMode::Create}) {
    for (NSString* path in
         @[ @"relative", @"~", @"~/cache", @"~nobody/cache" ]) {
      SCOPED_TRACE(path.UTF8String);
      EXPECT_EQ(
          resolve_bookmark_root(path, mode).error(), Error::InvalidArgument);
    }
    EXPECT_EQ(resolve_bookmark_root(nil, mode).error(), Error::InvalidArgument);
  }
  NSURL* file = [root.url URLByAppendingPathComponent:@"file"];
  ASSERT_TRUE([[NSData data] writeToURL:file atomically:NO]);
  EXPECT_EQ(
      resolve_bookmark_root(file.path, StorageCreationMode::OpenExisting)
          .error(),
      Error::AccessFailed);
  EXPECT_EQ(
      resolve_bookmark_root(
          [file.path stringByAppendingPathComponent:@"child"],
          StorageCreationMode::OpenExisting)
          .error(),
      Error::AccessFailed);
}

TEST_F(CoreAIBookmarkPathTest, RejectsInvalidStripeFiles) {
  NSData* bytes = [@"untouched target" dataUsingEncoding:NSUTF8StringEncoding];
  NSURL* target = [root.url URLByAppendingPathComponent:@"target"];
  ASSERT_TRUE([bytes writeToURL:target atomically:NO]);
  for (int kind = 0; kind < 3; ++kind) {
    TestDirectory isolated;
    ASSERT_NE(isolated.url, nil);
    NSURL* locks = [isolated.url URLByAppendingPathComponent:@"locks"];
    ASSERT_EQ(mkdir(locks.fileSystemRepresentation, 0700), 0);
    NSURL* stripe = test_cache_lock_url(isolated.url, key);
    if (kind == 0) {
      ASSERT_EQ(
          symlink(
              target.fileSystemRepresentation, stripe.fileSystemRepresentation),
          0);
    } else if (kind == 1) {
      ASSERT_EQ(mkdir(stripe.fileSystemRepresentation, 0700), 0);
    } else {
      ASSERT_EQ(mkfifo(stripe.fileSystemRepresentation, 0600), 0);
    }
    EXPECT_EQ(
        lock_cache_entry(isolated.url.path, key).error(), Error::AccessFailed);
    EXPECT_TRUE([[NSData dataWithContentsOfURL:target] isEqual:bytes]);
  }
}

} // namespace
