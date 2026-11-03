/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#import "coreai_cache_entry.h"

#import "coreai_cache_key.h"
#import "coreai_fault_injection.h"
#import "coreai_storage.h"
#import "coreai_validation.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <utility>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

constexpr size_t kBookmarkByteLimit = 8 * 1024 * 1024;
NSString* const kBookmarksDirectory = @"bookmarks";
NSString* const kSourcesDirectory = @"sources";
NSString* const kLocksDirectory = @"locks";
NSString* const kBookmarkSuffix = @(kBookmarkExtension);

NSString* bookmark_name(NSString* key) {
  return [key stringByAppendingString:kBookmarkSuffix];
}

Result<std::optional<FileDescriptor>>
open_child(int root_fd, NSString* name, StorageCreationMode mode) {
  auto opened = open_storage_directory(root_fd, name, mode);
  if (!opened.ok()) {
    return opened.error();
  }
  if (opened->created) {
    ET_CHECK_OK_OR_RETURN_ERROR(sync_storage_directory(root_fd));
  }
  return std::move(opened->descriptor);
}

Result<bool> entry_exists(int root_fd, NSString* key) {
  for (NSString* directory_name in
       @[ kBookmarksDirectory, kSourcesDirectory ]) {
    auto opened =
        open_child(root_fd, directory_name, StorageCreationMode::OpenExisting);
    if (!opened.ok()) {
      return opened.error();
    }
    if (!opened->has_value()) {
      continue;
    }
    NSString* entry =
        [directory_name isEqual:kBookmarksDirectory] ? bookmark_name(key) : key;
    struct stat info;
    if (lstat_at(
            opened->value().get(), entry.fileSystemRepresentation, &info) ==
        0) {
      return true;
    }
    ET_CHECK_OR_RETURN_ERROR(
        errno == ENOENT, AccessFailed, "Cannot inspect Core AI cache entry");
  }
  return false;
}

Result<FileDescriptor> duplicate_directory(int fd) {
  FileDescriptor duplicate(
      retry_eintr([&] { return fcntl(fd, F_DUPFD_CLOEXEC, 0); }));
  ET_CHECK_OR_RETURN_ERROR(
      duplicate.get() >= 0,
      AccessFailed,
      "Cannot retain Core AI cache directory");
  return duplicate;
}

} // namespace

Result<NSArray<NSString*>*> inventory_cache_keys(
    const CacheDirectory& directory) {
  NSMutableSet<NSString*>* keys = [NSMutableSet set];
  for (NSString* child_name in @[ kBookmarksDirectory, kSourcesDirectory ]) {
    auto child = open_child(
        directory.descriptor.get(),
        child_name,
        StorageCreationMode::OpenExisting);
    if (!child.ok()) {
      return child.error();
    }
    if (!child->has_value()) {
      continue;
    }
    auto names =
        storage_children(child->value().get(), StorageNamePolicy::SkipInvalid);
    if (!names.ok()) {
      return names.error();
    }
    for (NSString* name in names.get()) {
      NSString* key = name;
      if ([child_name isEqual:kBookmarksDirectory]) {
        if (![name hasSuffix:kBookmarkSuffix] ||
            name.length <= kBookmarkSuffix.length) {
          continue;
        }
        key = [name substringToIndex:name.length - kBookmarkSuffix.length];
      }
      if (is_cache_key(key)) {
        [keys addObject:key];
      }
    }
  }
  return [[keys allObjects] sortedArrayUsingSelector:@selector(compare:)];
}

LockedCacheEntry::LockedCacheEntry(
    FileDescriptor root,
    FileDescriptor bookmarks,
    FileDescriptor lock_file,
    NSString* root_path,
    NSString* key,
    CacheEntryPersistence persistence)
    : root_(std::move(root)),
      bookmarks_(std::move(bookmarks)),
      lock_file_(std::move(lock_file)),
      root_path_(root_path),
      key_(key),
      persistence_(persistence) {}

Result<std::unique_ptr<LockedCacheEntry>> LockedCacheEntry::acquire(
    const CacheDirectory& directory,
    NSString* key) {
  return acquire_impl(directory, key, StorageCreationMode::Create);
}

Result<std::unique_ptr<LockedCacheEntry>> LockedCacheEntry::acquire_existing(
    const CacheDirectory& directory,
    NSString* key) {
  return acquire_impl(directory, key, StorageCreationMode::OpenExisting);
}

Result<std::unique_ptr<LockedCacheEntry>> LockedCacheEntry::acquire_impl(
    const CacheDirectory& cache_directory,
    NSString* key,
    StorageCreationMode mode) {
  auto lock_name = cache_entry_lock_name(key);
  if (!lock_name.ok()) {
    return lock_name.error();
  }
  auto duplicated = duplicate_directory(cache_directory.descriptor.get());
  if (!duplicated.ok()) {
    return duplicated.error();
  }
  FileDescriptor directory = std::move(duplicated.get());
  const bool create = mode == StorageCreationMode::Create;
  // Checking before locking avoids creating locks/ for a missing entry.
  if (!create) {
    auto exists = entry_exists(directory.get(), key);
    if (!exists.ok()) {
      return exists.error();
    }
    if (!exists.get()) {
      return std::unique_ptr<LockedCacheEntry>();
    }
  }
  auto locks =
      open_child(directory.get(), kLocksDirectory, StorageCreationMode::Create);
  if (!locks.ok()) {
    return locks.error();
  }
  FileDescriptor lock_file(open_storage_shared_file(
      locks->value().get(), lock_name.get().fileSystemRepresentation));
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
  // Another process may have cleared the entry while this one waited.
  if (!create) {
    auto exists = entry_exists(directory.get(), key);
    if (!exists.ok()) {
      return exists.error();
    }
    if (!exists.get()) {
      return std::unique_ptr<LockedCacheEntry>();
    }
  }
  auto bookmarks = open_child(directory.get(), kBookmarksDirectory, mode);
  if (!bookmarks.ok()) {
    return bookmarks.error();
  }
  const CacheEntryPersistence persistence = bookmarks->has_value()
      ? CacheEntryPersistence::BookmarkCapable
      : CacheEntryPersistence::SourceOnly;
  FileDescriptor bookmark_directory =
      bookmarks->has_value() ? std::move(bookmarks->value()) : FileDescriptor();
  return std::unique_ptr<LockedCacheEntry>(new LockedCacheEntry(
      std::move(directory),
      std::move(bookmark_directory),
      std::move(lock_file),
      [cache_directory.path copy],
      [key copy],
      persistence));
}

Result<NSData*> LockedCacheEntry::read_bookmark() const {
  if (persistence_ == CacheEntryPersistence::SourceOnly) {
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
    const ssize_t count = retry_eintr([&] {
      return ::read(
          file.get(),
          static_cast<char*>(bytes.mutableBytes) + offset,
          bytes.length - offset);
    });
    ET_CHECK_OR_RETURN_ERROR(
        count > 0, AccessFailed, "Cannot read Core AI bookmark");
    offset += count;
  }
  return bytes;
}

Error LockedCacheEntry::write_bookmark(NSData* data) const {
  ET_CHECK_OR_RETURN_ERROR(
      [data isKindOfClass:NSData.class] && data.length > 0 &&
          data.length <= kBookmarkByteLimit,
      InvalidExternalData,
      "Invalid Core AI bookmark data");
  ET_CHECK_OR_RETURN_ERROR(
      persistence_ == CacheEntryPersistence::BookmarkCapable,
      InvalidArgument,
      "Cannot write through a source-only Core AI cache entry lock");
  return publish_storage_data(bookmarks_.get(), bookmark_name(key_), data);
}

Result<std::optional<FileDescriptor>> LockedCacheEntry::open_source_directory(
    StorageCreationMode mode) const {
  return open_child(root_.get(), kSourcesDirectory, mode);
}

Error LockedCacheEntry::apply_to_source(
    Error (*operation)(int, NSString*, StorageRemovalPolicy),
    StorageRemovalPolicy policy) const {
  auto sources = open_source_directory(StorageCreationMode::OpenExisting);
  if (!sources.ok()) {
    return sources.error();
  }
  if (!sources->has_value()) {
    return Error::Ok;
  }
  return operation(sources->value().get(), key_, policy);
}

Result<CacheSourceDirectory> LockedCacheEntry::prepare_source_directory()
    const {
  auto sources = open_source_directory(StorageCreationMode::Create);
  if (!sources.ok()) {
    return sources.error();
  }
  NSURL* url = [[NSURL fileURLWithPath:root_path_ isDirectory:YES]
      URLByAppendingPathComponent:kSourcesDirectory
                      isDirectory:YES];
  return CacheSourceDirectory{std::move(sources->value()), url};
}

Error LockedCacheEntry::remove_source() const {
  return apply_to_source(
      remove_storage_directory, StorageRemovalPolicy::Strict);
}

CacheEntryMaintenance LockedCacheEntry::maintenance() const {
  return CacheEntryMaintenance(*this);
}

Result<CacheBookmarkInspection> CacheEntryMaintenance::inspect_bookmark()
    const {
  if (entry_.persistence_ == CacheEntryPersistence::SourceOnly) {
    return CacheBookmarkInspection{CacheBookmarkState::Missing, nil};
  }
  struct stat info;
  const int inspected = lstat_at(
      entry_.bookmarks_.get(),
      bookmark_name(entry_.key_).fileSystemRepresentation,
      &info);
  if (inspected < 0 && errno == ENOENT) {
    return CacheBookmarkInspection{CacheBookmarkState::Missing, nil};
  }
  ET_CHECK_OR_RETURN_ERROR(
      inspected == 0, AccessFailed, "Cannot inspect Core AI bookmark");
  if (!S_ISREG(info.st_mode)) {
    return CacheBookmarkInspection{CacheBookmarkState::Corrupt, nil};
  }
  auto bookmark = entry_.read_bookmark();
  if (!bookmark.ok()) {
    if (bookmark.error() == Error::InvalidExternalData) {
      return CacheBookmarkInspection{CacheBookmarkState::Corrupt, nil};
    }
    return bookmark.error();
  }
  return CacheBookmarkInspection{
      bookmark.get() == nil ? CacheBookmarkState::Missing
                            : CacheBookmarkState::Present,
      bookmark.get()};
}

Error CacheEntryMaintenance::preflight() const {
  if (entry_.persistence_ == CacheEntryPersistence::BookmarkCapable) {
    ET_CHECK_OK_OR_RETURN_ERROR(preflight_storage_directory(
        entry_.bookmarks_.get(),
        bookmark_name(entry_.key_),
        StorageRemovalPolicy::Maintenance));
  }
  return entry_.apply_to_source(
      preflight_storage_directory, StorageRemovalPolicy::Maintenance);
}

Error CacheEntryMaintenance::remove_source() const {
  return entry_.apply_to_source(
      remove_storage_directory, StorageRemovalPolicy::Maintenance);
}

Error CacheEntryMaintenance::remove_bookmark() const {
  if (entry_.persistence_ == CacheEntryPersistence::SourceOnly) {
    return Error::Ok;
  }
  return remove_storage_directory(
      entry_.bookmarks_.get(),
      bookmark_name(entry_.key_),
      StorageRemovalPolicy::Maintenance);
}

} // namespace executorch::backends::coreai
