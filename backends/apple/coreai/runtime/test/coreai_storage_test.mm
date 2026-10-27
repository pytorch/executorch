/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fcntl.h>
#include <sys/stat.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include "coreai_fault_scope.h"
#include "coreai_file.h"
#include "coreai_filesystem_fixture.h"

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

TEST(CoreAIStorageTest, PreparesRootWithRootOnlyBackupPolicy) {
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
  auto prepared = prepare_storage_root(root.path, true);
  ASSERT_TRUE((prepared.ok() && [prepared.get() isEqualToString:root.path]));
  ASSERT_TRUE((backup_excluded(root)));
  ASSERT_TRUE((!backup_excluded(parent) && !backup_excluded(sandbox.url)));
  ASSERT_TRUE((!backup_excluded(root.URLByDeletingLastPathComponent)));

  ASSERT_TRUE(set_backup_excluded(root, false));
  auto inspected = prepare_storage_root(root.path, false);
  ASSERT_TRUE((inspected.ok() && [inspected.get() isEqualToString:root.path]));
  ASSERT_TRUE((!backup_excluded(root)));
  ASSERT_TRUE(
      (prepare_storage_root(root.path, true).ok() && backup_excluded(root)));

  NSURL* missing = [sandbox.url URLByAppendingPathComponent:@"missing"];
  ASSERT_EQ(
      prepare_storage_root(missing.path, false).error(),
      Error::InvalidArgument);
  EXPECT_FALSE([manager fileExistsAtPath:missing.path]);
  EXPECT_EQ(
      prepare_storage_root(@"relative", true).error(), Error::InvalidArgument);
  EXPECT_TRUE((!backup_excluded(parent) && !backup_excluded(sandbox.url)));
}

TEST(CoreAIStorageTest, RejectsInvalidRootsConsistently) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  NSURL* regular = [storage.url URLByAppendingPathComponent:@"regular"];
  NSData* bytes = [@"unchanged" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_TRUE([bytes writeToURL:regular atomically:NO]);
  for (bool create : {false, true}) {
    SCOPED_TRACE(create);
    EXPECT_EQ(
        prepare_storage_root(nil, create).error(), Error::InvalidArgument);
    for (NSString* path in @[
           @"",
           @"relative",
           @"~",
           @"~/cache",
           @"~nobody/cache",
           regular.path,
           [regular.path stringByAppendingPathComponent:@"child"]
         ]) {
      SCOPED_TRACE(path.UTF8String);
      EXPECT_EQ(
          prepare_storage_root(path, create).error(), Error::InvalidArgument);
    }
  }
  EXPECT_TRUE([[NSData dataWithContentsOfURL:regular] isEqual:bytes]);
}

TEST(CoreAIStorageTest, ListsChildrenRepeatedlyInSortedOrder) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  FileDescriptor root(open(
      storage.url.fileSystemRepresentation,
      O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  ASSERT_TRUE((root.get() >= 0));
  NSMutableArray<NSString*>* expected = [NSMutableArray array];
  for (int i = 7; i >= 0; --i) {
    NSString* name = [NSString stringWithFormat:@"entry-%d", i];
    FileDescriptor file(openat(
        root.get(),
        name.fileSystemRepresentation,
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
        0600));
    ASSERT_TRUE((file.get() >= 0));
    [expected insertObject:name atIndex:0];
  }
  const off_t position = lseek(root.get(), 0, SEEK_CUR);
  ASSERT_GE(position, 0);
  for (int pass = 0; pass < 3; ++pass) {
    errno = EIO;
    auto names = storage_children(root.get());
    ASSERT_TRUE((names.ok() && [names.get() isEqual:expected]));
    ASSERT_TRUE((fcntl(root.get(), F_GETFD) >= 0));
    EXPECT_EQ(lseek(root.get(), 0, SEEK_CUR), position);
  }
}

TEST(CoreAIStorageTest, ListsComposedAndDecomposedUnicodeNames) {
  for (NSString* name in @[ @"caf\u00e9", @"cafe\u0301" ]) {
    SCOPED_TRACE(name.UTF8String);
    TestDirectory storage;
    ASSERT_NE(storage.url, nil);
    FileDescriptor root(open(
        storage.url.fileSystemRepresentation,
        O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    ASSERT_GE(root.get(), 0);
    FileDescriptor file(openat(
        root.get(),
        name.UTF8String,
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
        0600));
    ASSERT_GE(file.get(), 0);
    for (bool skip_invalid : {false, true}) {
      auto names = storage_children(root.get(), skip_invalid);
      ASSERT_TRUE(names.ok());
      ASSERT_EQ(names.get().count, 1);
      EXPECT_STREQ(names.get().firstObject.UTF8String, name.UTF8String);
    }
  }
}

TEST(CoreAIStorageTest, CreatesNestedFilesWithoutOverwritingExistingContents) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  FileDescriptor root(open(
      storage.url.fileSystemRepresentation,
      O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  ASSERT_GE(root.get(), 0);
  NSData* original = [@"original" dataUsingEncoding:NSUTF8StringEncoding];
  NSData* replacement = [@"replacement" dataUsingEncoding:NSUTF8StringEncoding];
  NSString* path = @"nested/parents/model.bin";
  NSURL* file = [storage.url URLByAppendingPathComponent:path];
  ASSERT_EQ(
      write_storage_file(root.get(), path, original.bytes, original.length),
      Error::Ok);
  EXPECT_TRUE(([[NSData dataWithContentsOfURL:file] isEqual:original]));
  EXPECT_EQ(
      write_storage_file(
          root.get(), path, replacement.bytes, replacement.length),
      Error::AccessFailed);
  EXPECT_TRUE(([[NSData dataWithContentsOfURL:file] isEqual:original]));
  ASSERT_EQ(
      write_storage_file(root.get(), @"nested/parents/empty", nullptr, 0),
      Error::Ok);
  NSData* empty = [NSData
      dataWithContentsOfURL:
          [storage.url URLByAppendingPathComponent:@"nested/parents/empty"]];
  ASSERT_NE(empty, nil);
  EXPECT_EQ(empty.length, 0);
}

TEST(CoreAIStorageTest, RemovesFailedWritesAndAllowsRetry) {
  const struct {
    StorageOperation operation;
    int match_index;
  } rows[] = {
      {StorageOperation::Write, 2},
      {StorageOperation::SyncFile, 1},
      {StorageOperation::SyncDirectory, 1},
      {StorageOperation::SyncDirectory, 2}};
  const std::vector<uint8_t> bytes(1024 * 1024 + 1, 0x5a);
  for (const auto& row : rows) {
    SCOPED_TRACE(static_cast<int>(row.operation));
    SCOPED_TRACE(row.match_index);
    TestDirectory storage;
    ASSERT_NE(storage.url, nil);
    FileDescriptor root(open(
        storage.url.fileSystemRepresentation,
        O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    ASSERT_GE(root.get(), 0);
    NSString* path = @"nested/model.bin";
    NSURL* file = [storage.url URLByAppendingPathComponent:path];
    StorageFaultScope fault(row.operation);
    fault.match_index = row.match_index;
    EXPECT_EQ(
        write_storage_file(root.get(), path, bytes.data(), bytes.size()),
        Error::AccessFailed);
    EXPECT_EQ(fault.hits, 1);
    EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:file.path]);
    ASSERT_EQ(
        write_storage_file(root.get(), path, bytes.data(), bytes.size()),
        Error::Ok);
    NSData* actual = [NSData dataWithContentsOfURL:file];
    ASSERT_NE(actual, nil);
    ASSERT_EQ(actual.length, bytes.size());
    EXPECT_EQ(memcmp(actual.bytes, bytes.data(), bytes.size()), 0);
  }
}

TEST(CoreAIStorageTest, SyncsEachCreatedParentAndTheWrittenFile) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  FileDescriptor root(open(
      storage.url.fileSystemRepresentation,
      O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  ASSERT_GE(root.get(), 0);
  __block std::vector<StorageOperation> operations;
  StorageFaultScope fault(StorageOperation::SyncDirectory);
  fault.armed = false;
  fault.observe = ^(StorageOperation operation) {
    operations.push_back(operation);
  };
  EXPECT_EQ(
      write_storage_file(root.get(), @"nested/parents/model.bin", "x", 1),
      Error::Ok);
  EXPECT_EQ(
      operations,
      (std::vector<StorageOperation>{
          StorageOperation::SyncDirectory,
          StorageOperation::SyncDirectory,
          StorageOperation::Write,
          StorageOperation::SyncFile,
          StorageOperation::SyncDirectory}));
}

TEST(CoreAIStorageTest, RejectsUnsafeWritePathsWithoutCreatingFiles) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  FileDescriptor root(open(
      storage.url.fileSystemRepresentation,
      O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  ASSERT_GE(root.get(), 0);
  for (NSString* path in @[
         @"",
         @"..",
         @"../escape",
         @"a/../b",
         @"a/./b",
         @"a//b",
         @"/leading",
         @"trailing/"
       ]) {
    SCOPED_TRACE(path.UTF8String);
    EXPECT_EQ(
        write_storage_file(root.get(), path, "x", 1), Error::InvalidArgument);
  }
  EXPECT_EQ(
      write_storage_file(root.get(), @"no-data", nullptr, 1),
      Error::InvalidArgument);
  auto names = storage_children(root.get());
  ASSERT_TRUE(names.ok());
  EXPECT_EQ(names.get().count, 0);
}

TEST(CoreAIStorageTest, CreatesAndReopensSharedFileWithoutTruncation) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  FileDescriptor root(open(
      storage.url.fileSystemRepresentation,
      O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  ASSERT_GE(root.get(), 0);
  FileDescriptor created(open_storage_shared_file(root.get(), "shared"));
  ASSERT_GE(created.get(), 0);
  ASSERT_EQ(write(created.get(), "data", 4), 4);
  FileDescriptor reopened(open_storage_shared_file(root.get(), "shared"));
  ASSERT_GE(reopened.get(), 0);
  EXPECT_NE(created.get(), reopened.get());
  EXPECT_NE(fcntl(created.get(), F_GETFD) & FD_CLOEXEC, 0);
  EXPECT_NE(fcntl(reopened.get(), F_GETFD) & FD_CLOEXEC, 0);
  struct stat first, second;
  ASSERT_EQ(fstat(created.get(), &first), 0);
  ASSERT_EQ(fstat(reopened.get(), &second), 0);
  EXPECT_EQ(first.st_ino, second.st_ino);
  EXPECT_EQ(second.st_size, 4);
  char bytes[4];
  ASSERT_EQ(read(reopened.get(), bytes, sizeof(bytes)), sizeof(bytes));
  EXPECT_EQ(memcmp(bytes, "data", sizeof(bytes)), 0);
  EXPECT_EQ(lseek(created.get(), 0, SEEK_CUR), 4);
}

TEST(CoreAIStorageTest, RetriesEintr) {
  int calls = 0;
  const ssize_t count = retry_eintr([&]() -> ssize_t {
    if (++calls < 3) {
      errno = EINTR;
      return -1;
    }
    return 7;
  });
  ASSERT_TRUE((calls == 3 && count == 7));
}

static ::testing::AssertionResult no_temporary_files(int root) {
  auto children = storage_children(root);
  if (!children.ok())
    return ::testing::AssertionFailure() << "Cannot list storage";
  for (NSString* child in children.get()) {
    if ([child hasPrefix:@".tmp-"])
      return ::testing::AssertionFailure() << child.UTF8String;
  }
  return ::testing::AssertionSuccess();
}

TEST(CoreAIStorageTest, AtomicPublicationPreservesOldBytesOnFailure) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  FileDescriptor root(open(
      storage.url.fileSystemRepresentation,
      O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  ASSERT_TRUE((root.get() >= 0));
  NSData* old = [@"old" dataUsingEncoding:NSUTF8StringEncoding];
  NSData* next = [@"next" dataUsingEncoding:NSUTF8StringEncoding];
  NSURL* published = [storage.url URLByAppendingPathComponent:@"published"];
  ASSERT_EQ(publish_storage_data(root.get(), @"published", old), Error::Ok);
  for (StorageOperation operation :
       {StorageOperation::Write,
        StorageOperation::SyncFile,
        StorageOperation::SyncDirectory,
        StorageOperation::FullSync,
        StorageOperation::Rename}) {
    SCOPED_TRACE(static_cast<int>(operation));
    StorageFaultScope fault(operation);
    EXPECT_EQ(
        publish_storage_data(root.get(), @"published", next),
        Error::AccessFailed);
    EXPECT_EQ(fault.hits, 1);
    EXPECT_TRUE(([[NSData dataWithContentsOfURL:published] isEqual:old]));
    EXPECT_TRUE(no_temporary_files(root.get()));
  }

  ASSERT_EQ(publish_storage_data(root.get(), @"published", next), Error::Ok);
  EXPECT_TRUE(([[NSData dataWithContentsOfURL:published] isEqual:next]));
  EXPECT_TRUE(no_temporary_files(root.get()));
}

TEST(CoreAIStorageTest, FullyFlushesPublicationBeforeAndAfterRename) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  FileDescriptor root(open(
      storage.url.fileSystemRepresentation,
      O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  ASSERT_GE(root.get(), 0);
  __block std::vector<StorageOperation> operations;
  StorageFaultScope fault(StorageOperation::FullSync);
  fault.armed = false;
  fault.observe = ^(StorageOperation operation) {
    operations.push_back(operation);
  };
  NSData* data = [@"published" dataUsingEncoding:NSUTF8StringEncoding];
  EXPECT_EQ(publish_storage_data(root.get(), @"published", data), Error::Ok);
  EXPECT_EQ(
      operations,
      (std::vector<StorageOperation>{
          StorageOperation::Write,
          StorageOperation::SyncFile,
          StorageOperation::FullSync,
          StorageOperation::SyncDirectory,
          StorageOperation::Rename,
          StorageOperation::FullSync}));
}

TEST(CoreAIStorageTest, ReportsPostRenameFlushFailureWithNewContentsVisible) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  FileDescriptor root(open(
      storage.url.fileSystemRepresentation,
      O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  ASSERT_GE(root.get(), 0);
  NSData* old = [@"old" dataUsingEncoding:NSUTF8StringEncoding];
  NSData* next = [@"next" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(publish_storage_data(root.get(), @"published", old), Error::Ok);
  StorageFaultScope fault(StorageOperation::FullSync);
  fault.match_index = 2;
  EXPECT_EQ(
      publish_storage_data(root.get(), @"published", next),
      Error::AccessFailed);
  EXPECT_EQ(fault.hits, 1);
  NSURL* published = [storage.url URLByAppendingPathComponent:@"published"];
  EXPECT_TRUE(([[NSData dataWithContentsOfURL:published] isEqual:next]));
  EXPECT_TRUE(no_temporary_files(root.get()));
}

TEST(CoreAIStorageTest, RetriesInterruptedPublicationOperations) {
  TestDirectory storage;
  ASSERT_NE(storage.url, nil);
  FileDescriptor root(open(
      storage.url.fileSystemRepresentation,
      O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  ASSERT_GE(root.get(), 0);
  NSData* data = [@"published" dataUsingEncoding:NSUTF8StringEncoding];
  for (StorageOperation operation :
       {StorageOperation::Write,
        StorageOperation::SyncFile,
        StorageOperation::SyncDirectory,
        StorageOperation::FullSync,
        StorageOperation::Rename}) {
    SCOPED_TRACE(static_cast<int>(operation));
    StorageFaultScope fault(operation);
    fault.error = EINTR;
    EXPECT_EQ(publish_storage_data(root.get(), @"published", data), Error::Ok);
    EXPECT_EQ(fault.hits, 1);
    NSURL* published = [storage.url URLByAppendingPathComponent:@"published"];
    EXPECT_TRUE(([[NSData dataWithContentsOfURL:published] isEqual:data]));
    EXPECT_TRUE(no_temporary_files(root.get()));
  }
}
