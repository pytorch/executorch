/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "coreai_cache_root.h"
#include "coreai_fault_injection.h"
#include "coreai_file.h"

#include <sys/stat.h>
#include <cerrno>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

Error ensure_excluded_from_backup(NSURL* directory) {
  NSURL* fresh = [NSURL fileURLWithPath:directory.path isDirectory:YES];
  NSNumber* excluded = nil;
  NSError* error = nil;
  ET_CHECK_OR_RETURN_ERROR(
      [fresh getResourceValue:&excluded
                       forKey:NSURLIsExcludedFromBackupKey
                        error:&error],
      AccessFailed,
      "Cannot read Core AI backup exclusion: %s",
      error.localizedDescription.UTF8String);
  if (excluded.boolValue) {
    return Error::Ok;
  }
  ET_CHECK_OR_RETURN_ERROR(
      [fresh setResourceValue:@YES
                       forKey:NSURLIsExcludedFromBackupKey
                        error:&error],
      AccessFailed,
      "Cannot exclude Core AI storage from backup: %s",
      error.localizedDescription.UTF8String);
  return Error::Ok;
}

} // namespace

Result<NSString*> default_coreai_assets_root() {
  NSURL* caches =
      [NSFileManager.defaultManager URLsForDirectory:NSCachesDirectory
                                           inDomains:NSUserDomainMask]
          .firstObject;
  ET_CHECK_OR_RETURN_ERROR(
      caches.isFileURL, AccessFailed, "Cannot resolve Core AI cache directory");
  return [caches.path stringByAppendingPathComponent:@"executorch_coreai"];
}

Result<NSString*> prepare_storage_root(
    NSString* path,
    StorageCreationMode mode) {
  const bool create = mode == StorageCreationMode::Create;
  ET_CHECK_OR_RETURN_ERROR(
      [path isKindOfClass:NSString.class] && [path hasPrefix:@"/"] &&
          [path rangeOfString:@"\0"].location == NSNotFound,
      InvalidArgument,
      "Core AI storage requires an absolute path");
  struct stat info;
  const int result =
      retry_eintr([&] { return stat(path.fileSystemRepresentation, &info); });
  ET_CHECK_OR_RETURN_ERROR(
      result == 0 || errno == ENOENT || errno == ENOTDIR,
      AccessFailed,
      "Cannot inspect Core AI storage directory");
  ET_CHECK_OR_RETURN_ERROR(
      result == 0 ? S_ISDIR(info.st_mode) : create && errno == ENOENT,
      InvalidArgument,
      "Core AI storage must be a directory");
  if (create && result != 0) {
    NSDictionary* attributes = @{
      NSFileProtectionKey :
          NSFileProtectionCompleteUntilFirstUserAuthentication,
    };
    ET_CHECK_OR_RETURN_ERROR(
        fault_point(FaultPoint::BeforeCreateDirectory) == 0,
        AccessFailed,
        "Cannot begin Core AI directory creation");
    NSError* error = nil;
    ET_CHECK_OR_RETURN_ERROR(
        [NSFileManager.defaultManager createDirectoryAtPath:path
                                withIntermediateDirectories:YES
                                                 attributes:attributes
                                                      error:&error],
        AccessFailed,
        "Cannot create Core AI storage directory: %s",
        error.localizedDescription.UTF8String);
  }
  if (create) {
    ET_CHECK_OK_OR_RETURN_ERROR(ensure_excluded_from_backup(
        [NSURL fileURLWithPath:path isDirectory:YES]));
  }
  return path;
}

} // namespace executorch::backends::coreai
