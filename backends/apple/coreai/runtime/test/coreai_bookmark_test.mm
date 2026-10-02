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
#include "coreai_filesystem_fixture.h"
#include "coreai_manifest_fixture.h"
#include "coreai_storage.h"

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

namespace {
constexpr size_t kMiB = 1024 * 1024;

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
  BookmarkDirectory root;
  void SetUp() override {
    CoreAIBookmarkIdentityTest::SetUp();
    if (HasFatalFailure()) return;
    ASSERT_NE(root.url, nil);
  }
  NSURL* lock_url() {
    return [[root.url URLByAppendingPathComponent:@"locks"]
        URLByAppendingPathComponent:[key stringByAppendingString:@".lock"]];
  }
};

class CoreAIBookmarkRawTest : public CoreAIBookmarkPathTest {
 protected:
  std::unique_ptr<BookmarkLock> locked;
  NSURL* url = nil;
  void SetUp() override {
    CoreAIBookmarkPathTest::SetUp();
    if (HasFatalFailure()) return;
    auto result = lock_bookmark(root.url.path, key);
    ASSERT_TRUE(result.ok());
    locked = std::move(result.get());
    ASSERT_NE(locked, nullptr);
    url = test_bookmark_url(root.url, key);
  }
};

::testing::AssertionResult bookmark_bytes_equal(const BookmarkLock& lock, NSData* expected) {
  auto result = read_bookmark(lock);
  if (!result.ok())
    return ::testing::AssertionFailure() << "Bookmark read error " << int(result.error());
  if (expected == nil ? result.get() != nil : ![result.get() isEqual:expected]) {
    return ::testing::AssertionFailure() << "Bookmark bytes differ";
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult no_temporary_bookmarks(NSURL* root) {
  NSURL* directory = [root URLByAppendingPathComponent:@"bookmarks"];
  FileDescriptor fd(open(directory.fileSystemRepresentation, O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (fd.get() < 0) return ::testing::AssertionFailure() << "open bookmark directory: " << errno;
  auto children = storage_children(fd.get());
  if (!children.ok()) return ::testing::AssertionFailure() << "Cannot enumerate bookmark directory";
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

TEST_F(CoreAIBookmarkIdentityTest, KeySeparatesPlatformArchitectureWeightsAndAotContext) {
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

TEST_F(CoreAIBookmarkRawTest, RejectsOversizedStoredBookmark) {
  FileDescriptor file(
      open(url.fileSystemRepresentation, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600));
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

TEST_F(CoreAIBookmarkPathTest, ProcessContentionAllowsIndependentKeyAndRecoversAfterHolderDeath) {
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

}  // namespace
