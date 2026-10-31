/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fcntl.h>
#include <sys/stat.h>
#include "coreai_cache_root.h"
#include "coreai_fault_scope.h"
#include "coreai_file.h"
#include "coreai_filesystem_fixture.h"
#include "coreai_storage.h"

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

TEST(
    CoreAICacheRootTest,
    ResolvesDefaultAndExplicitPathsWithoutCreatingStorage) {
  auto default_path = cache_directory_path(nullptr);
  ASSERT_TRUE(default_path.ok());
  EXPECT_TRUE(default_path.get().isAbsolutePath);
  EXPECT_TRUE(
      [default_path.get().lastPathComponent isEqual:@"executorch_coreai"]);

  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  NSURL* missing = [storage.url URLByAppendingPathComponent:@"missing"];
  auto explicit_path = cache_directory_path(missing.path.UTF8String);
  ASSERT_TRUE(explicit_path.ok());
  EXPECT_TRUE([explicit_path.get() isEqual:missing.path]);
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:missing.path]);
  for (const char* invalid : {"", "relative", "~", "~/cache", "\xff"}) {
    EXPECT_EQ(cache_directory_path(invalid).error(), Error::InvalidArgument);
  }
}

TEST(CoreAICacheRootTest, PreparesRootWithRootOnlyBackupPolicy) {
  NSFileManager* manager = NSFileManager.defaultManager;
  TestDirectory sandbox;
  ASSERT_NE(sandbox.url, nil);
  NSURL* parent = [sandbox.url URLByAppendingPathComponent:@"parent"];
  ASSERT_TRUE(([manager createDirectoryAtURL:parent
                 withIntermediateDirectories:NO
                                  attributes:nil
                                       error:nil]));
  ASSERT_TRUE(set_backup_excluded(sandbox.url, false));
  ASSERT_TRUE(set_backup_excluded(parent, false));
  NSURL* root = [parent URLByAppendingPathComponent:@"nested/models"];
  auto prepared =
      resolve_test_cache_directory(root.path, StorageCreationMode::Create);
  ASSERT_TRUE(prepared.ok());
  ASSERT_TRUE(prepared->has_value());
  EXPECT_TRUE([prepared->value().path
      isEqualToString:root.URLByResolvingSymlinksInPath.path]);
  ASSERT_TRUE((backup_excluded(root)));
  ASSERT_TRUE((!backup_excluded(parent) && !backup_excluded(sandbox.url)));
  ASSERT_TRUE((!backup_excluded(root.URLByDeletingLastPathComponent)));

  ASSERT_TRUE(set_backup_excluded(root, false));
  auto inspected = resolve_test_cache_directory(
      root.path, StorageCreationMode::OpenExisting);
  ASSERT_TRUE(inspected.ok());
  ASSERT_TRUE(inspected->has_value());
  EXPECT_TRUE([inspected->value().path
      isEqualToString:root.URLByResolvingSymlinksInPath.path]);
  ASSERT_TRUE((!backup_excluded(root)));
  ASSERT_TRUE(
      (resolve_test_cache_directory(root.path, StorageCreationMode::Create)
           .ok() &&
       backup_excluded(root)));

  NSURL* missing = [sandbox.url URLByAppendingPathComponent:@"missing"];
  auto absent = resolve_test_cache_directory(
      missing.path, StorageCreationMode::OpenExisting);
  ASSERT_TRUE(absent.ok());
  EXPECT_FALSE(absent->has_value());
  EXPECT_FALSE([manager fileExistsAtPath:missing.path]);
  EXPECT_EQ(
      resolve_test_cache_directory(@"relative", StorageCreationMode::Create)
          .error(),
      Error::InvalidArgument);
  EXPECT_TRUE((!backup_excluded(parent) && !backup_excluded(sandbox.url)));
}

TEST(CoreAICacheRootTest, SkipsCreationForExistingRoots) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  NSURL* root = [storage.url URLByAppendingPathComponent:@"cache"];
  ASSERT_TRUE(
      resolve_test_cache_directory(root.path, StorageCreationMode::Create)
          .ok());
  StorageFaultScope fault(FaultPoint::BeforeCreateDirectory);
  EXPECT_TRUE(
      resolve_test_cache_directory(root.path, StorageCreationMode::Create)
          .ok());
  EXPECT_TRUE(
      resolve_test_cache_directory(root.path, StorageCreationMode::OpenExisting)
          .ok());
  EXPECT_EQ(fault.matches, 0);
  EXPECT_TRUE(backup_excluded(root));

  NSURL* missing = [storage.url URLByAppendingPathComponent:@"missing"];
  auto absent = resolve_test_cache_directory(
      missing.path, StorageCreationMode::OpenExisting);
  ASSERT_TRUE(absent.ok());
  EXPECT_FALSE(absent->has_value());
  EXPECT_EQ(fault.matches, 0);
  EXPECT_EQ(
      resolve_test_cache_directory(missing.path, StorageCreationMode::Create)
          .error(),
      Error::AccessFailed);
  EXPECT_EQ(fault.hits, 1);
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:missing.path]);
  EXPECT_TRUE(
      resolve_test_cache_directory(missing.path, StorageCreationMode::Create)
          .ok());
}

TEST(CoreAICacheRootTest, CreatesDirectoriesWithDefaultPermissions) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  NSURL* reference = [storage.url URLByAppendingPathComponent:@"reference"];
  ASSERT_EQ(mkdir(reference.fileSystemRepresentation, 0777), 0);
  struct stat expected;
  ASSERT_EQ(stat(reference.fileSystemRepresentation, &expected), 0);
  NSURL* root = [storage.url URLByAppendingPathComponent:@"cache"];
  ASSERT_TRUE(
      resolve_test_cache_directory(root.path, StorageCreationMode::Create)
          .ok());
  FileDescriptor directory(
      open(root.fileSystemRepresentation, O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  ASSERT_GE(directory.get(), 0);
  ASSERT_EQ(
      write_storage_file(directory.get(), @"nested/file", "x", 1), Error::Ok);
  for (NSURL* entry in
       @[ root, [root URLByAppendingPathComponent:@"nested"] ]) {
    struct stat actual;
    ASSERT_EQ(stat(entry.fileSystemRepresentation, &actual), 0);
    EXPECT_EQ(actual.st_mode & 0777, expected.st_mode & 0777);
  }
}

TEST(CoreAICacheRootTest, RecreatesRemovedRoot) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  NSURL* root = [storage.url URLByAppendingPathComponent:@"cache"];
  ASSERT_TRUE(
      resolve_test_cache_directory(root.path, StorageCreationMode::Create)
          .ok());
  ASSERT_EQ(rmdir(root.fileSystemRepresentation), 0);
  ASSERT_TRUE(
      resolve_test_cache_directory(root.path, StorageCreationMode::Create)
          .ok());
  EXPECT_TRUE(backup_excluded(root));
}

TEST(CoreAICacheRootTest, NewRootAndDescendantsUseFirstUnlockProtection) {
  TestDirectory sandbox;
  ASSERT_NE(sandbox.url, nil);
  NSURL* root = [sandbox.url URLByAppendingPathComponent:@"cache"];
  ASSERT_TRUE(
      resolve_test_cache_directory(root.path, StorageCreationMode::Create)
          .ok());
  FileDescriptor directory(
      open_directory_at(AT_FDCWD, root.fileSystemRepresentation));
  ASSERT_GE(directory.get(), 0);
  ASSERT_EQ(
      write_storage_file(directory.get(), @"nested/model.bin", "x", 1),
      Error::Ok);
  for (NSURL* entry in @[
         root,
         [root URLByAppendingPathComponent:@"nested"],
         [root URLByAppendingPathComponent:@"nested/model.bin"]
       ]) {
    SCOPED_TRACE(entry.path.UTF8String);
    NSError* error = nil;
    NSDictionary* attributes =
        [NSFileManager.defaultManager attributesOfItemAtPath:entry.path
                                                       error:&error];
    ASSERT_NE(attributes, nil) << error.description.UTF8String;
    EXPECT_TRUE([attributes[NSFileProtectionKey]
        isEqual:NSFileProtectionCompleteUntilFirstUserAuthentication]);
  }
}

TEST(CoreAICacheRootTest, PreservesExistingRootProtection) {
  TestDirectory sandbox;
  ASSERT_NE(sandbox.url, nil);
  NSURL* root = [sandbox.url URLByAppendingPathComponent:@"existing"];
  NSFileManager* manager = NSFileManager.defaultManager;
  NSError* error = nil;
  // Set the fixture policy at creation, without reclassifying a populated
  // cache.
  ASSERT_TRUE([manager
             createDirectoryAtURL:root
      withIntermediateDirectories:NO
                       attributes:@{
                         NSFileProtectionKey : NSFileProtectionComplete
                       }
                            error:&error])
      << error.description.UTF8String;
  NSDictionary* before = [manager attributesOfItemAtPath:root.path
                                                   error:&error];
  ASSERT_NE(before, nil) << error.description.UTF8String;
  NSString* protection = before[NSFileProtectionKey];
  ASSERT_NE(protection, nil);
  if (![protection isEqual:NSFileProtectionComplete]) {
    GTEST_SKIP() << "Filesystem did not apply NSFileProtectionComplete; got "
                 << protection.UTF8String;
  }
  for (StorageCreationMode mode :
       {StorageCreationMode::OpenExisting, StorageCreationMode::Create}) {
    ASSERT_TRUE(resolve_test_cache_directory(root.path, mode).ok());
    NSDictionary* after = [manager attributesOfItemAtPath:root.path
                                                    error:&error];
    ASSERT_NE(after, nil) << error.description.UTF8String;
    EXPECT_TRUE([after[NSFileProtectionKey] isEqual:NSFileProtectionComplete]);
  }
}

TEST(CoreAICacheRootTest, SupportsSymlinkAndAnchorsResolvedDirectory) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  NSURL* first = [storage.url URLByAppendingPathComponent:@"first"];
  NSURL* second = [storage.url URLByAppendingPathComponent:@"second"];
  NSURL* configured = [storage.url URLByAppendingPathComponent:@"configured"];
  ASSERT_EQ(mkdir(first.fileSystemRepresentation, 0777), 0);
  ASSERT_EQ(mkdir(second.fileSystemRepresentation, 0777), 0);
  ASSERT_EQ(
      symlink(
          first.fileSystemRepresentation, configured.fileSystemRepresentation),
      0);

  auto resolved = resolve_test_cache_directory(
      configured.path, StorageCreationMode::Create);
  ASSERT_TRUE(resolved.ok());
  ASSERT_TRUE(resolved->has_value());
  struct stat target_info;
  struct stat resolved_info;
  ASSERT_EQ(stat(first.fileSystemRepresentation, &target_info), 0);
  ASSERT_EQ(fstat(resolved->value().descriptor.get(), &resolved_info), 0);
  EXPECT_EQ(resolved_info.st_ino, target_info.st_ino);

  ASSERT_EQ(unlink(configured.fileSystemRepresentation), 0);
  ASSERT_EQ(
      symlink(
          second.fileSystemRepresentation, configured.fileSystemRepresentation),
      0);
  ASSERT_EQ(
      write_storage_file(
          resolved->value().descriptor.get(), @"anchored", "x", 1),
      Error::Ok);
  EXPECT_TRUE([NSFileManager.defaultManager
      fileExistsAtPath:[first URLByAppendingPathComponent:@"anchored"].path]);
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[second URLByAppendingPathComponent:@"anchored"].path]);
}

TEST(CoreAICacheRootTest, RejectsInvalidRootsConsistently) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  NSURL* regular = [storage.url URLByAppendingPathComponent:@"regular"];
  NSData* bytes = [@"unchanged" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_TRUE([bytes writeToURL:regular atomically:NO]);
  for (StorageCreationMode mode :
       {StorageCreationMode::OpenExisting, StorageCreationMode::Create}) {
    SCOPED_TRACE(static_cast<int>(mode));
    EXPECT_EQ(
        resolve_test_cache_directory(nil, mode).error(),
        Error::InvalidArgument);
    for (NSString* path in
         @[ @"", @"relative", @"~", @"~/cache", @"~nobody/cache" ]) {
      SCOPED_TRACE(path.UTF8String);
      EXPECT_EQ(
          resolve_test_cache_directory(path, mode).error(),
          Error::InvalidArgument);
    }
    for (NSString* path in @[
           regular.path,
           [regular.path stringByAppendingPathComponent:@"child"]
         ]) {
      EXPECT_EQ(
          resolve_test_cache_directory(path, mode).error(),
          mode == StorageCreationMode::Create ? Error::InvalidArgument
                                              : Error::AccessFailed);
    }
  }
  EXPECT_TRUE([[NSData dataWithContentsOfURL:regular] isEqual:bytes]);
}
