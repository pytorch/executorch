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
// Resolves symlinks once and anchors the resulting directory with an owned
// descriptor. OpenExisting returns no value when the directory is missing.
runtime::Result<std::optional<CacheDirectory>> resolve_cache_directory(
    NSString* path,
    StorageCreationMode mode);

} // namespace executorch::backends::coreai
