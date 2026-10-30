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
#include <optional>
#include "coreai_cache_root.h"

namespace executorch::backends::coreai {

// Invalid and unowned names are omitted from the sorted inventory.
runtime::Result<NSArray<NSString*>*> inventory_cache_keys(
    const CacheDirectory& directory);

enum class CacheBookmarkState {
  Missing,
  Corrupt,
  Present,
};

struct CacheBookmarkInspection final {
  CacheBookmarkState state;
  NSData* data;
};

enum class CacheEntryPersistence {
  BookmarkCapable,
  StagingOnly,
};

struct CacheStagingDirectory final {
  FileDescriptor descriptor;
  NSURL* url;
};

class CacheEntryMaintenance;

class LockedCacheEntry final {
 public:
  LockedCacheEntry(const LockedCacheEntry&) = delete;
  LockedCacheEntry& operator=(const LockedCacheEntry&) = delete;

  // Each acquisition owns an independent descriptor on one stable stripe.
  static runtime::Result<std::unique_ptr<LockedCacheEntry>> acquire(
      const CacheDirectory& directory,
      NSString* key);
  // Missing entries return a null handle without changing entry storage.
  static runtime::Result<std::unique_ptr<LockedCacheEntry>> acquire_existing(
      const CacheDirectory& directory,
      NSString* key);

  CacheEntryPersistence persistence() const {
    return persistence_;
  }
  NSString* key() const {
    return key_;
  }
  // Missing is nil. Empty, oversized, inaccessible, or nonregular files fail
  // closed.
  runtime::Result<NSData*> read_bookmark() const;
  runtime::Error write_bookmark(NSData* data) const;
  runtime::Result<CacheStagingDirectory> prepare_staging_directory() const;
  // Missing staging is a no-op; unexpected entry types fail before deletion.
  runtime::Error remove_staging() const;

  // The view must not outlive this locked entry.
  CacheEntryMaintenance maintenance() const;

 private:
  friend class CacheEntryMaintenance;

  LockedCacheEntry(
      FileDescriptor root,
      FileDescriptor bookmarks,
      FileDescriptor lock_file,
      NSString* root_path,
      NSString* key,
      CacheEntryPersistence persistence);
  static runtime::Result<std::unique_ptr<LockedCacheEntry>> acquire_impl(
      const CacheDirectory& directory,
      NSString* key,
      StorageCreationMode mode);
  runtime::Result<std::optional<FileDescriptor>> open_staging_directory(
      StorageCreationMode mode) const;

  FileDescriptor root_;
  FileDescriptor bookmarks_;
  FileDescriptor lock_file_;
  NSString* root_path_;
  NSString* key_;
  CacheEntryPersistence persistence_;
};

class CacheEntryMaintenance final {
 public:
  runtime::Result<CacheBookmarkInspection> inspect_bookmark() const;
  runtime::Error preflight() const;
  runtime::Error remove_staging() const;
  runtime::Error remove_bookmark() const;

 private:
  friend class LockedCacheEntry;
  explicit CacheEntryMaintenance(const LockedCacheEntry& entry)
      : entry_(entry) {}

  const LockedCacheEntry& entry_;
};

} // namespace executorch::backends::coreai
