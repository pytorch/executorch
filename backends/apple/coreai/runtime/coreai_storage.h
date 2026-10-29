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
#include <cstddef>
#include "coreai_file.h"

namespace executorch::backends::coreai {

runtime::Error sync_storage_directory(int fd);
struct StorageDirectory {
  int fd;
  bool created;
};
// Opens one child component without following symlinks. The caller owns fd;
// a missing child with OpenExisting returns {-1, false}. Does not flush
// directories.
runtime::Result<StorageDirectory>
open_storage_directory(int parent_fd, NSString* name, StorageCreationMode mode);
// Returns an independently owned descriptor for a stable coordination file.
int open_storage_shared_file(int parent_fd, const char* name);
// Creates a new file and missing parents, then syncs the file and its parent.
// On failure, attempts to remove the new file; created parents may remain.
runtime::Error write_storage_file(
    int root_fd,
    NSString* relative_path,
    const void* bytes,
    size_t size);
// Atomically replaces name with data and fully flushes the file and renamed
// entry. On error, readers see either the old or the new complete bytes.
runtime::Error publish_storage_data(int root_fd, NSString* name, NSData* data);

enum class StorageNamePolicy {
  Strict,
  SkipInvalid,
};
// Lists sorted names without changing fd's enumeration offset.
runtime::Result<NSArray<NSString*>*> storage_children(
    int fd,
    StorageNamePolicy policy = StorageNamePolicy::Strict);

} // namespace executorch::backends::coreai
