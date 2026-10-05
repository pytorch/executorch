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

namespace executorch::backends::coreai {

// Resolves the default location without creating or modifying storage.
runtime::Result<NSString*> default_coreai_assets_root();
// Requires an absolute directory path. With create, makes the directory and
// excludes it from backup; otherwise a missing directory is InvalidArgument.
runtime::Result<NSString*> prepare_storage_root(NSString* path, bool create);
runtime::Error ensure_excluded_from_backup(NSURL* directory);

runtime::Error sync_storage_directory(int fd);
// Returns an independently owned descriptor for a stable coordination file.
int open_storage_shared_file(int parent_fd, const char* name);
// Creates a new file and missing parents, then syncs the file and its parent.
runtime::Error write_storage_file(
    int root_fd,
    NSString* relative_path,
    const void* bytes,
    size_t size);
// Atomically replaces name with data. On error, readers see either the old or
// the new complete bytes.
runtime::Error publish_storage_data(int root_fd, NSString* name, NSData* data);
runtime::Result<NSArray<NSString*>*> storage_children(
    int fd,
    bool skip_invalid_names = false);
// With remove=false, only preflight the keyed tree without modifying it.
runtime::Error remove_storage_staging(int root_fd, NSString* key, bool remove);

enum class StorageOperation {
  Rename,
  BeforeSDK,
  AfterSDK,
  BeforeEvict,
  AfterEvict,
  Remove,
};
#if defined(COREAI_ASSETS_TESTING) && COREAI_ASSETS_TESTING
int storage_fault(StorageOperation operation);
namespace testing {
using StorageFaultCallback = int (*)(StorageOperation);
extern thread_local StorageFaultCallback storage_fault_callback;
} // namespace testing
#else
constexpr int storage_fault(StorageOperation) {
  return 0;
}
#endif

} // namespace executorch::backends::coreai
