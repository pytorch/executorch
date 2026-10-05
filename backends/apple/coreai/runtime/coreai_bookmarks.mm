/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#import "coreai_bookmarks.h"
#import "coreai_storage.h"

#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <cstring>
#include <unistd.h>
#include <cerrno>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;
// An allocation bound, not an SDK bookmark format limit.
constexpr size_t kBookmarkByteLimit = 8 * 1024 * 1024;

bool text(id value) {
  return [value isKindOfClass:NSString.class] && [value length] > 0 &&
         [value length] <= 4096 &&
         [value rangeOfString:@"\0"].location == NSNotFound;
}
bool component(id value) {
  return text(value) && ![value containsString:@"/"] &&
         ![value containsString:@"\\"] && ![value isEqual:@"."] &&
         ![value isEqual:@".."];
}
bool digest(id value) {
  return text(value) && [value length] == 64 &&
         [value rangeOfCharacterFromSet:
                    [[NSCharacterSet
                        characterSetWithCharactersInString:@"0123456789abcdef"]
                        invertedSet]]
                 .location == NSNotFound;
}
NSString* bookmark_name(NSString* key) {
  return [key stringByAppendingString:@".bookmark"];
}
}  // namespace

Result<NSString*> bookmark_key(const Manifest& selected, NSString* platform,
                               NSString* architecture) {
  NSString* bundle = selected.path.lastPathComponent;
  NSString* exported_digest = selected.bundle_digests[bundle];
  ET_CHECK_OR_RETURN_ERROR(
      digest(exported_digest) && component(bundle) && text(platform) &&
          text(architecture),
      DelegateInvalidCompatibility,
      "Core AI bookmark key requires a selected bundle and SDK architecture");
  NSArray* fields = @[
    @"1", exported_digest, selected.aot_compiled ? @"aot" : @"source", bundle,
    platform, architecture, @"AIModelCache.default", @"default-options-v1",
    @"persistent"
  ];
  NSData* encoded = [NSJSONSerialization dataWithJSONObject:fields
                                                    options:0
                                                      error:nil];
  ET_CHECK_OR_RETURN_ERROR(encoded != nil && encoded.length <= UINT32_MAX,
                           InvalidExternalData, "Cannot encode bookmark key");
  unsigned char bytes[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(encoded.bytes, static_cast<CC_LONG>(encoded.length), bytes);
  NSMutableString* key = [NSMutableString stringWithCapacity:64];
  for (unsigned char byte : bytes) [key appendFormat:@"%02x", byte];
  return [key copy];
}

namespace {
Result<NSString*> resolve_root(NSString* path, bool create) {
  ET_CHECK_OR_RETURN_ERROR(text(path) && path.isAbsolutePath, InvalidArgument,
                           "Core AI assets require an absolute root path");
  if (!create) {
    struct stat info;
    const int result =
        retry_eintr([&] { return stat(path.fileSystemRepresentation, &info); });
    if (result < 0 && errno == ENOENT) return path;
    ET_CHECK_OR_RETURN_ERROR(result == 0 && S_ISDIR(info.st_mode),
                             AccessFailed, "Invalid Core AI assets root");
  }
  return prepare_storage_root(path, create);
}

// Returns -1 when the root does not exist.
Result<int> open_root(NSString* root) {
  const int fd = retry_eintr([&] {
    return open(root.fileSystemRepresentation,
                O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  });
  if (fd < 0 && errno == ENOENT) return -1;
  ET_CHECK_OR_RETURN_ERROR(fd >= 0, AccessFailed,
                           "Cannot open Core AI assets root");
  return fd;
}

// Returns -1 when the child does not exist and create is false.
Result<int> open_child(int root_fd, NSString* name, bool create) {
  bool created = false;
  if (create) {
    const int made = retry_eintr([&] {
      return mkdirat(root_fd, name.fileSystemRepresentation, 0700);
    });
    ET_CHECK_OR_RETURN_ERROR(made == 0 || errno == EEXIST, AccessFailed,
                             "Cannot create Core AI assets directory");
    created = made == 0;
  }
  const int fd = retry_eintr([&] {
    return openat(root_fd, name.fileSystemRepresentation,
                  O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  });
  if (fd < 0 && errno == ENOENT && !create) return -1;
  ET_CHECK_OR_RETURN_ERROR(fd >= 0, AccessFailed,
                           "Invalid Core AI assets directory");
  if (created) {
    FileDescriptor child(fd);
    ET_CHECK_OK_OR_RETURN_ERROR(sync_storage_directory(root_fd));
    return child.release();
  }
  return fd;
}

Result<bool> entry_exists(int root_fd, NSString* key) {
  for (NSString* name in @[ @"bookmarks", @"staging" ]) {
    auto opened = open_child(root_fd, name, false);
    if (!opened.ok()) return opened.error();
    FileDescriptor child(opened.get());
    if (child.get() < 0) continue;
    NSString* entry = [name isEqual:@"bookmarks"] ? bookmark_name(key) : key;
    struct stat info;
    const int result = retry_eintr([&] {
      return fstatat(child.get(), entry.fileSystemRepresentation, &info,
                     AT_SYMLINK_NOFOLLOW);
    });
    if (result == 0) return true;
    ET_CHECK_OR_RETURN_ERROR(errno == ENOENT, AccessFailed,
                             "Cannot inspect Core AI cache entry");
  }
  return false;
}
}  // namespace

Result<NSString*> resolve_bookmark_root(NSString* path, bool create) {
  return resolve_root(path, create);
}

Result<NSArray<NSString*>*> inventory_bookmark_keys(NSString* root) {
  auto resolved = resolve_root(root, false);
  if (!resolved.ok()) return resolved.error();
  auto opened = open_root(resolved.get());
  if (!opened.ok()) return opened.error();
  FileDescriptor directory(opened.get());
  if (directory.get() < 0) return @[];
  NSMutableSet<NSString*>* keys = [NSMutableSet set];
  for (NSString* name in @[ @"bookmarks", @"staging" ]) {
    auto child_result = open_child(directory.get(), name, false);
    if (!child_result.ok()) return child_result.error();
    FileDescriptor child(child_result.get());
    if (child.get() < 0) continue;
    auto children = storage_children(child.get(), true);
    if (!children.ok()) return children.error();
    for (NSString* entry in children.get()) {
      NSString* key = entry;
      if ([name isEqual:@"bookmarks"]) {
        if (![entry hasSuffix:@".bookmark"]) continue;
        key = [entry substringToIndex:entry.length - @".bookmark".length];
      }
      if (digest(key)) [keys addObject:key];
    }
  }
  return [keys.allObjects sortedArrayUsingSelector:@selector(compare:)];
}

BookmarkLock::BookmarkLock(int root, int bookmarks, int file,
                           NSString* root_path, NSString* key)
    : root_(root),
      bookmarks_(bookmarks),
      file_(file),
      root_path_(root_path),
      key_(key) {}

Result<std::unique_ptr<BookmarkLock>> lock_bookmark(NSString* root,
                                                    NSString* key,
                                                    bool create) {
  ET_CHECK_OR_RETURN_ERROR(digest(key), InvalidArgument,
                           "Invalid Core AI bookmark key");
  auto resolved = resolve_root(root, create);
  if (!resolved.ok()) return resolved.error();
  root = resolved.get();
  auto opened = open_root(root);
  if (!opened.ok()) return opened.error();
  FileDescriptor directory(opened.get());
  if (directory.get() < 0) {
    ET_CHECK_OR_RETURN_ERROR(!create, AccessFailed,
                             "Core AI assets root disappeared");
    return std::unique_ptr<BookmarkLock>();
  }
  if (!create) {
    auto exists = entry_exists(directory.get(), key);
    if (!exists.ok()) return exists.error();
    if (!exists.get()) return std::unique_ptr<BookmarkLock>();
  }
  auto bookmark_result = open_child(directory.get(), @"bookmarks", create);
  if (!bookmark_result.ok()) return bookmark_result.error();
  FileDescriptor bookmarks(bookmark_result.get());
  auto lock_result = open_child(directory.get(), @"locks", true);
  if (!lock_result.ok()) return lock_result.error();
  FileDescriptor locks(lock_result.get());
  NSString* name = [key stringByAppendingString:@".lock"];
  FileDescriptor file(
      open_storage_shared_file(locks.get(), name.fileSystemRepresentation));
  struct stat info;
  ET_CHECK_OR_RETURN_ERROR(file.get() >= 0 && fstat(file.get(), &info) == 0 &&
                               S_ISREG(info.st_mode),
                           AccessFailed, "Invalid Core AI bookmark lock");
  ET_CHECK_OR_RETURN_ERROR(
      retry_eintr([&] { return flock(file.get(), LOCK_EX); }) == 0,
      AccessFailed, "Cannot lock Core AI bookmark");
  return std::unique_ptr<BookmarkLock>(
      new BookmarkLock(directory.release(), bookmarks.release(), file.release(),
                       root, [key copy]));
}

Result<NSData*> read_bookmark(const BookmarkLock& lock) {
  if (lock.bookmarks_.get() < 0) return static_cast<NSData*>(nil);
  FileDescriptor file(openat(lock.bookmarks_.get(),
                             bookmark_name(lock.key_).fileSystemRepresentation,
                             O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
  if (file.get() < 0 && errno == ENOENT) return static_cast<NSData*>(nil);
  ET_CHECK_OR_RETURN_ERROR(file.get() >= 0, AccessFailed,
                           "Cannot open Core AI bookmark");
  struct stat info;
  ET_CHECK_OR_RETURN_ERROR(
      fstat(file.get(), &info) == 0 && S_ISREG(info.st_mode) &&
          info.st_size > 0 &&
          static_cast<uint64_t>(info.st_size) <= kBookmarkByteLimit,
      InvalidExternalData, "Invalid Core AI bookmark file");
  NSMutableData* bytes = [NSMutableData dataWithLength:info.st_size];
  size_t offset = 0;
  while (offset < bytes.length) {
    ssize_t count =
        ::read(file.get(), static_cast<char*>(bytes.mutableBytes) + offset,
               bytes.length - offset);
    if (count < 0 && errno == EINTR) continue;
    ET_CHECK_OR_RETURN_ERROR(count > 0, AccessFailed,
                             "Cannot read Core AI bookmark");
    offset += count;
  }
  return bytes;
}

Error write_bookmark(const BookmarkLock& lock, NSData* data) {
  ET_CHECK_OR_RETURN_ERROR(
      [data isKindOfClass:NSData.class] && data.length > 0 &&
          data.length <= kBookmarkByteLimit,
      InvalidExternalData, "Invalid Core AI bookmark data");
  return publish_storage_data(lock.bookmarks_.get(), bookmark_name(lock.key_),
                              data);
}

Error remove_bookmark(const BookmarkLock& lock) {
  if (lock.bookmarks_.get() < 0) return Error::Ok;
  ET_CHECK_OR_RETURN_ERROR(storage_fault(StorageOperation::Remove) == 0,
                           AccessFailed,
                           "Core AI bookmark removal interrupted");
  ET_CHECK_OR_RETURN_ERROR(
      unlinkat(lock.bookmarks_.get(),
               bookmark_name(lock.key_).fileSystemRepresentation, 0) == 0 ||
          errno == ENOENT,
      AccessFailed, "Cannot remove Core AI bookmark");
  return sync_storage_directory(lock.bookmarks_.get());
}

Error remove_bookmark_staging(const BookmarkLock& lock, bool remove) {
  return remove_storage_staging(lock.root_.get(), lock.key_, remove);
}

Result<NSString*> prepare_bookmark_staging(const BookmarkLock& lock) {
  auto staging = open_child(lock.root_.get(), @"staging", true);
  if (!staging.ok()) return staging.error();
  FileDescriptor directory(staging.get());
  return [lock.root_path_ stringByAppendingPathComponent:@"staging"];
}

}  // namespace executorch::backends::coreai
