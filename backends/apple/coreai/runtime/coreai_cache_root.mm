/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "coreai_cache_root.h"
#include "coreai_fault_injection.h"
#include "coreai_validation.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

Error ensure_excluded_from_backup(NSString* path) {
  NSURL* directory = [NSURL fileURLWithPath:path isDirectory:YES];
  NSNumber* excluded = nil;
  NSError* error = nil;
  ET_CHECK_OR_RETURN_ERROR(
      [directory getResourceValue:&excluded
                           forKey:NSURLIsExcludedFromBackupKey
                            error:&error],
      AccessFailed,
      "Cannot read Core AI backup exclusion: %s",
      error.localizedDescription.UTF8String);
  if (excluded.boolValue) {
    return Error::Ok;
  }
  ET_CHECK_OR_RETURN_ERROR(
      [directory setResourceValue:@YES
                           forKey:NSURLIsExcludedFromBackupKey
                            error:&error],
      AccessFailed,
      "Cannot exclude Core AI cache directory from backup: %s",
      error.localizedDescription.UTF8String);
  return Error::Ok;
}

Result<NSString*> canonical_path(NSString* path) {
  char* resolved = realpath(path.fileSystemRepresentation, nullptr);
  ET_CHECK_OR_RETURN_ERROR(
      resolved != nullptr,
      AccessFailed,
      "Cannot resolve Core AI cache directory");
  NSString* canonical = [NSFileManager.defaultManager
      stringWithFileSystemRepresentation:resolved
                                  length:std::strlen(resolved)];
  std::free(resolved);
  ET_CHECK_OR_RETURN_ERROR(
      canonical != nil,
      AccessFailed,
      "Cannot decode Core AI cache directory path");
  return canonical;
}

Result<NSString*> default_cache_directory_impl() {
  NSURL* caches =
      [NSFileManager.defaultManager URLsForDirectory:NSCachesDirectory
                                           inDomains:NSUserDomainMask]
          .firstObject;
  ET_CHECK_OR_RETURN_ERROR(
      caches.isFileURL, AccessFailed, "Cannot resolve Core AI cache directory");
  return [caches.path stringByAppendingPathComponent:@"executorch_coreai"];
}

} // namespace

Result<NSString*> default_cache_directory() {
  return default_cache_directory_impl();
}

Result<NSString*> cache_directory_path(const char* configured_path) {
  if (configured_path == nullptr) {
    return default_cache_directory();
  }
  NSString* path = [NSString stringWithUTF8String:configured_path];
  ET_CHECK_OR_RETURN_ERROR(
      is_nonempty_string(path) && path.isAbsolutePath && ![path hasPrefix:@"~"],
      InvalidArgument,
      "Core AI cache directory requires an absolute UTF-8 path");
  return path;
}

Result<std::optional<CacheDirectory>> resolve_cache_directory(
    NSString* path,
    StorageCreationMode mode) {
  const bool create = mode == StorageCreationMode::Create;
  ET_CHECK_OR_RETURN_ERROR(
      is_nonempty_string(path) && path.isAbsolutePath && ![path hasPrefix:@"~"],
      InvalidArgument,
      "Core AI cache directory requires an absolute path");
  struct stat link_info;
  const int link_inspected = retry_eintr(
      [&] { return lstat(path.fileSystemRepresentation, &link_info); });
  struct stat info;
  int inspected =
      retry_eintr([&] { return stat(path.fileSystemRepresentation, &info); });
  if (inspected < 0 && errno == ENOENT && !create) {
    return std::optional<CacheDirectory>();
  }
  if (inspected < 0 && errno == ENOTDIR) {
    if (create) {
      return Error::InvalidArgument;
    }
    return Error::AccessFailed;
  }
  ET_CHECK_OR_RETURN_ERROR(
      inspected == 0 || (create && errno == ENOENT),
      AccessFailed,
      "Cannot inspect Core AI cache directory");
  if (inspected == 0 && !S_ISDIR(info.st_mode)) {
    ET_LOG(Error, "Core AI cache path is not a directory");
    if (create) {
      return Error::InvalidArgument;
    }
    return Error::AccessFailed;
  }
  if (inspected < 0) {
    NSDictionary* attributes = @{
      NSFileProtectionKey :
          NSFileProtectionCompleteUntilFirstUserAuthentication,
    };
    ET_CHECK_OR_RETURN_ERROR(
        fault_point(FaultPoint::BeforeCreateDirectory) == 0,
        AccessFailed,
        "Cannot begin Core AI cache directory creation");
    NSError* error = nil;
    ET_CHECK_OR_RETURN_ERROR(
        [NSFileManager.defaultManager createDirectoryAtPath:path
                                withIntermediateDirectories:YES
                                                 attributes:attributes
                                                      error:&error],
        AccessFailed,
        "Cannot create Core AI cache directory: %s",
        error.localizedDescription.UTF8String);
  }
  NSString* resolved_path = path;
  if (link_inspected == 0 && S_ISLNK(link_info.st_mode)) {
    auto canonical = canonical_path(path);
    if (!canonical.ok()) {
      return canonical.error();
    }
    resolved_path = canonical.get();
  }
  FileDescriptor directory(retry_eintr([&] {
    return open(
        resolved_path.fileSystemRepresentation,
        O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  }));
  if (directory.get() < 0 && errno == ENOENT && !create) {
    return std::optional<CacheDirectory>();
  }
  ET_CHECK_OR_RETURN_ERROR(
      directory.get() >= 0 && fstat(directory.get(), &info) == 0 &&
          S_ISDIR(info.st_mode),
      AccessFailed,
      "Cannot open Core AI cache directory");
  if (create) {
    ET_CHECK_OK_OR_RETURN_ERROR(ensure_excluded_from_backup(resolved_path));
  }
  return std::optional<CacheDirectory>(
      std::in_place,
      CacheDirectory{[resolved_path copy], std::move(directory)});
}

} // namespace executorch::backends::coreai
