/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#import "coreai_bookmarks.h"

#import "coreai_cache_key.h"
#import "coreai_cache_root.h"
#import "coreai_fault_injection.h"
#import "coreai_storage.h"
#import "coreai_validation.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

constexpr size_t kBookmarkByteLimit = 8 * 1024 * 1024;
NSString* const kBookmarksDirectory = @"bookmarks";
NSString* const kStagingDirectory = @"staging";
NSString* const kLocksDirectory = @"locks";
NSString* const kBookmarkExtension = @".bookmark";

NSString* bookmark_name(NSString* key) {
  return [key stringByAppendingString:kBookmarkExtension];
}

// Returns -1 when the root does not exist.
Result<int> open_root(NSString* root) {
  const int fd = retry_eintr([&] {
    return open(
        root.fileSystemRepresentation, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  });
  if (fd < 0 && errno == ENOENT) {
    return -1;
  }
  ET_CHECK_OR_RETURN_ERROR(
      fd >= 0, AccessFailed, "Cannot open Core AI assets root");
  return fd;
}

// Returns -1 when the child does not exist in OpenExisting mode.
Result<int> open_child(int root_fd, NSString* name, StorageCreationMode mode) {
  auto opened = open_storage_directory(root_fd, name, mode);
  if (!opened.ok()) {
    return opened.error();
  }
  FileDescriptor child(opened.get().fd);
  if (opened.get().created) {
    ET_CHECK_OK_OR_RETURN_ERROR(sync_storage_directory(root_fd));
  }
  return child.release();
}

Result<bool> entry_exists(int root_fd, NSString* key) {
  for (NSString* directory_name in
       @[ kBookmarksDirectory, kStagingDirectory ]) {
    auto opened =
        open_child(root_fd, directory_name, StorageCreationMode::OpenExisting);
    if (!opened.ok()) {
      return opened.error();
    }
    FileDescriptor directory(opened.get());
    if (directory.get() < 0) {
      continue;
    }
    NSString* entry =
        [directory_name isEqual:kBookmarksDirectory] ? bookmark_name(key) : key;
    struct stat info;
    if (lstat_at(directory.get(), entry.fileSystemRepresentation, &info) == 0) {
      return true;
    }
    ET_CHECK_OR_RETURN_ERROR(
        errno == ENOENT, AccessFailed, "Cannot inspect Core AI cache entry");
  }
  return false;
}
} // namespace

Result<std::unique_ptr<CacheEntryLock>> acquire_cache_entry_lock(
    NSString* root,
    NSString* key,
    StorageCreationMode mode) {
  auto lock_name = cache_entry_lock_name(key);
  if (!lock_name.ok()) {
    return lock_name.error();
  }
  auto resolved = resolve_bookmark_root(root, mode);
  if (!resolved.ok()) {
    return resolved.error();
  }
  root = [resolved.get() copy];
  auto opened = open_root(root);
  if (!opened.ok()) {
    return opened.error();
  }
  FileDescriptor directory(opened.get());
  const bool create = mode == StorageCreationMode::Create;
  if (directory.get() < 0) {
    ET_CHECK_OR_RETURN_ERROR(
        !create, AccessFailed, "Core AI assets root disappeared");
    return std::unique_ptr<CacheEntryLock>();
  }
  if (!create) {
    auto exists = entry_exists(directory.get(), key);
    if (!exists.ok()) {
      return exists.error();
    }
    if (!exists.get()) {
      return std::unique_ptr<CacheEntryLock>();
    }
  }
  auto locks =
      open_child(directory.get(), kLocksDirectory, StorageCreationMode::Create);
  if (!locks.ok()) {
    return locks.error();
  }
  FileDescriptor lock_directory(locks.get());
  FileDescriptor lock_file(open_storage_shared_file(
      lock_directory.get(), lock_name.get().fileSystemRepresentation));
  struct stat info;
  ET_CHECK_OR_RETURN_ERROR(
      lock_file.get() >= 0 && fstat(lock_file.get(), &info) == 0 &&
          S_ISREG(info.st_mode),
      AccessFailed,
      "Invalid Core AI cache entry lock");
  ET_CHECK_OR_RETURN_ERROR(
      fault_point(FaultPoint::BeforeCacheEntryLock) == 0,
      AccessFailed,
      "Cannot begin Core AI cache entry lock acquisition");
  ET_CHECK_OR_RETURN_ERROR(
      retry_eintr([&] { return flock(lock_file.get(), LOCK_EX); }) == 0,
      AccessFailed,
      "Cannot lock Core AI cache entry");
  if (!create) {
    auto exists = entry_exists(directory.get(), key);
    if (!exists.ok()) {
      return exists.error();
    }
    if (!exists.get()) {
      return std::unique_ptr<CacheEntryLock>();
    }
  }
  auto bookmarks = open_child(directory.get(), kBookmarksDirectory, mode);
  if (!bookmarks.ok()) {
    return bookmarks.error();
  }
  FileDescriptor bookmark_directory(bookmarks.get());
  const CacheEntryPersistence persistence = bookmark_directory.get() >= 0
      ? CacheEntryPersistence::BookmarkCapable
      : CacheEntryPersistence::StagingOnly;
  return std::unique_ptr<CacheEntryLock>(new CacheEntryLock(
      directory.release(),
      bookmark_directory.release(),
      lock_file.release(),
      root,
      [key copy],
      persistence));
}

Result<NSString*> resolve_bookmark_root(
    NSString* path,
    StorageCreationMode mode) {
  ET_CHECK_OR_RETURN_ERROR(
      nonempty_string(path) && [path hasPrefix:@"/"],
      InvalidArgument,
      "Core AI assets require an absolute root path");
  if (mode == StorageCreationMode::Create) {
    return prepare_storage_root(path, mode);
  }
  struct stat info;
  const int result =
      retry_eintr([&] { return stat(path.fileSystemRepresentation, &info); });
  if (result < 0 && errno == ENOENT) {
    return path;
  }
  ET_CHECK_OR_RETURN_ERROR(
      result == 0 && S_ISDIR(info.st_mode),
      AccessFailed,
      "Invalid Core AI assets root");
  return path;
}

CacheEntryLock::CacheEntryLock(
    int root,
    int bookmarks,
    int lock_file,
    NSString* root_path,
    NSString* key,
    CacheEntryPersistence persistence)
    : root_(root),
      bookmarks_(bookmarks),
      lock_file_(lock_file),
      root_path_(root_path),
      key_(key),
      persistence_(persistence) {}

Result<NSData*> CacheEntryLock::read_bookmark() const {
  if (persistence_ == CacheEntryPersistence::StagingOnly) {
    return static_cast<NSData*>(nil);
  }
  FileDescriptor file(retry_eintr([&] {
    return openat(
        bookmarks_.get(),
        bookmark_name(key_).fileSystemRepresentation,
        O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  }));
  if (file.get() < 0 && errno == ENOENT) {
    return static_cast<NSData*>(nil);
  }
  ET_CHECK_OR_RETURN_ERROR(
      file.get() >= 0, AccessFailed, "Cannot open Core AI bookmark");
  struct stat info;
  ET_CHECK_OR_RETURN_ERROR(
      fstat(file.get(), &info) == 0 && S_ISREG(info.st_mode) &&
          info.st_size > 0 &&
          static_cast<uint64_t>(info.st_size) <= kBookmarkByteLimit,
      InvalidExternalData,
      "Invalid Core AI bookmark file");
  NSMutableData* bytes = [NSMutableData dataWithLength:info.st_size];
  ET_CHECK_OR_RETURN_ERROR(
      bytes != nil, MemoryAllocationFailed, "Cannot allocate Core AI bookmark");
  size_t offset = 0;
  while (offset < bytes.length) {
    const ssize_t count = ::read(
        file.get(),
        static_cast<char*>(bytes.mutableBytes) + offset,
        bytes.length - offset);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    ET_CHECK_OR_RETURN_ERROR(
        count > 0, AccessFailed, "Cannot read Core AI bookmark");
    offset += count;
  }
  return bytes;
}

Error CacheEntryLock::write_bookmark(NSData* data) const {
  ET_CHECK_OR_RETURN_ERROR(
      [data isKindOfClass:NSData.class] && data.length > 0 &&
          data.length <= kBookmarkByteLimit,
      InvalidExternalData,
      "Invalid Core AI bookmark data");
  ET_CHECK_OR_RETURN_ERROR(
      persistence_ == CacheEntryPersistence::BookmarkCapable,
      InvalidArgument,
      "Cannot write through a staging-only Core AI cache entry lock");
  return publish_storage_data(bookmarks_.get(), bookmark_name(key_), data);
}

Result<NSString*> CacheEntryLock::prepare_staging_root() const {
  auto staging =
      open_child(root_.get(), kStagingDirectory, StorageCreationMode::Create);
  if (!staging.ok()) {
    return staging.error();
  }
  FileDescriptor directory(staging.get());
  return [root_path_ stringByAppendingPathComponent:kStagingDirectory];
}

Result<std::unique_ptr<CacheEntryLock>> lock_cache_entry(
    NSString* root,
    NSString* key) {
  return acquire_cache_entry_lock(root, key, StorageCreationMode::Create);
}

Result<std::unique_ptr<CacheEntryLock>> lock_existing_cache_entry(
    NSString* root,
    NSString* key) {
  return acquire_cache_entry_lock(root, key, StorageCreationMode::OpenExisting);
}

} // namespace executorch::backends::coreai
