/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <memory>
#import "coreai_assets.h"
#include "coreai_file.h"

namespace executorch::backends::coreai {

runtime::Result<NSString*> bookmark_key(const Manifest& selected,
                                        NSString* platform,
                                        NSString* architecture);
// With create=false, a missing root is returned unchanged for no-op maintenance.
runtime::Result<NSString*> resolve_bookmark_root(NSString* path,
                                                 bool create = true);
// Missing roots produce an empty inventory.
runtime::Result<NSArray<NSString*>*> inventory_bookmark_keys(NSString* root);

class BookmarkLock final {
 public:
  BookmarkLock(const BookmarkLock&) = delete;
  BookmarkLock& operator=(const BookmarkLock&) = delete;
  NSString* root_path() const { return root_path_; }

 private:
  friend runtime::Result<std::unique_ptr<BookmarkLock>> lock_bookmark(
      NSString* root, NSString* key, bool create);
  friend runtime::Result<NSData*> read_bookmark(const BookmarkLock& lock);
  friend runtime::Error write_bookmark(const BookmarkLock& lock, NSData* data);
  friend runtime::Error remove_bookmark(const BookmarkLock& lock);
  friend runtime::Error remove_bookmark_staging(const BookmarkLock& lock,
                                                bool remove);
  friend runtime::Result<NSString*> prepare_bookmark_staging(
      const BookmarkLock& lock);
  BookmarkLock(int root, int bookmarks, int file, NSString* root_path,
               NSString* key);
  FileDescriptor root_;
  FileDescriptor bookmarks_;
  FileDescriptor file_;
  NSString* root_path_;
  NSString* key_;
};

// Each acquisition opens an independent descriptor; the backend never removes
// lock files. With create=false, missing roots or entries return a null lock.
runtime::Result<std::unique_ptr<BookmarkLock>> lock_bookmark(NSString* root,
                                                             NSString* key,
                                                             bool create = true);
// Missing is nil. Empty, oversized, inaccessible or nonregular files fail
// closed.
runtime::Result<NSData*> read_bookmark(const BookmarkLock& lock);
runtime::Error write_bookmark(const BookmarkLock& lock, NSData* data);
runtime::Error remove_bookmark(const BookmarkLock& lock);
runtime::Error remove_bookmark_staging(const BookmarkLock& lock,
                                        bool remove = true);
// Prepared only for cold loads; source preparation appends key/bundle.
runtime::Result<NSString*> prepare_bookmark_staging(const BookmarkLock& lock);

}  // namespace executorch::backends::coreai
