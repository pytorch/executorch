/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <Foundation/Foundation.h>
#include <executorch/runtime/core/result.h>
#include <optional>
#include "coreai_storage.h"

namespace executorch::backends::coreai {

struct CacheDirectory final {
  NSString* path;
  FileDescriptor descriptor;
};

// Resolves the default location without creating or modifying storage.
runtime::Result<NSString*> default_cache_directory();
// Null selects the default; invalid explicit paths never fall back to it.
runtime::Result<NSString*> cache_directory_path(const char* configured_path);

// Both functions resolve a symlink in the final path component once and hold
// the directory open, so later path changes cannot redirect the cache.
//
// Creates the directory if missing and excludes it from backup. A path that is
// not a directory is the caller's configuration error: InvalidArgument.
runtime::Result<CacheDirectory> create_cache_directory(NSString* path);
// Returns no value when the directory is missing. A path that is not a
// directory means existing storage is unusable: AccessFailed.
runtime::Result<std::optional<CacheDirectory>> open_cache_directory(
    NSString* path);

} // namespace executorch::backends::coreai
