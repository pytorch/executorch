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
#include <memory>
#include "coreai_file.h"

namespace executorch::backends::coreai {

// With OpenExisting, a missing root is returned unchanged for no-op
// maintenance.
runtime::Result<NSString*> resolve_bookmark_root(
    NSString* path,
    StorageCreationMode mode);

enum class CacheEntryPersistence {
  BookmarkCapable,
  StagingOnly,
};

class CacheEntryLock final {
 public:
  CacheEntryLock(const CacheEntryLock&) = delete;
  CacheEntryLock& operator=(const CacheEntryLock&) = delete;

  CacheEntryPersistence persistence() const {
    return persistence_;
  }
  // Missing is nil. Empty, oversized, inaccessible, or nonregular files fail
  // closed.
  runtime::Result<NSData*> read_bookmark() const;
  runtime::Error write_bookmark(NSData* data) const;
  // Returns the staging parent. Source preparation appends key and bundle.
  runtime::Result<NSString*> prepare_staging_root() const;

 private:
  friend runtime::Result<std::unique_ptr<CacheEntryLock>>
  acquire_cache_entry_lock(
      NSString* root,
      NSString* key,
      StorageCreationMode mode);

  CacheEntryLock(
      int root,
      int bookmarks,
      int lock_file,
      NSString* root_path,
      NSString* key,
      CacheEntryPersistence persistence);

  FileDescriptor root_;
  FileDescriptor bookmarks_;
  FileDescriptor lock_file_;
  NSString* root_path_;
  NSString* key_;
  CacheEntryPersistence persistence_;
};

// Each acquisition opens an independent descriptor on one stable stripe.
// Different keys can share a stripe; callers must not hold multiple entry locks
// at once. Stripe files are never removed by the backend.
runtime::Result<std::unique_ptr<CacheEntryLock>> lock_cache_entry(
    NSString* root,
    NSString* key);
// Missing roots and entries return a null lock without changing storage.
// Existing staging-only entries return a lock with StagingOnly persistence.
runtime::Result<std::unique_ptr<CacheEntryLock>> lock_existing_cache_entry(
    NSString* root,
    NSString* key);

} // namespace executorch::backends::coreai
