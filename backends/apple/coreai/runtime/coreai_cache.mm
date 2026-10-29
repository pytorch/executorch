/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "coreai_cache.h"

#import "coreai_bookmarks.h"
#import "coreai_cache_key.h"
#import "coreai_cache_root.h"
#import "coreai_load_coordinator.h"
#import "coreai_manifest.h"
#include "coreai_pte.h"

#include <executorch/extension/data_loader/file_data_loader.h>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

Result<NSString*> cache_root(const char* path) {
  NSString* root = nil;
  if (path == nullptr) {
    auto default_root = default_coreai_assets_root();
    if (!default_root.ok()) {
      return default_root.error();
    }
    root = default_root.get();
  } else {
    root = [NSString stringWithUTF8String:path];
  }
  return resolve_bookmark_root(root, StorageCreationMode::OpenExisting);
}

Error clear_keys(NSString* root, NSArray<NSString*>* keys) {
  NSArray<NSString*>* ordered = [[[NSSet setWithArray:keys] allObjects]
      sortedArrayUsingSelector:@selector(compare:)];
  Error first_error = Error::Ok;
  for (NSString* key in ordered) {
    const Error error = clear_bookmark_assets(root, key, nil);
    if (error != Error::Ok) {
      ET_LOG(
          Error,
          "Core AI cache clear failed for key %s: 0x%x",
          key.UTF8String,
          static_cast<unsigned>(error));
      if (first_error == Error::Ok) {
        first_error = error;
      }
    }
  }
  return first_error;
}

Error clear_pte(runtime::DataLoader& loader, NSString* root) {
  // Keep the verified image and selected segment owners alive through eviction.
  auto pte = inspect_coreai_pte(loader);
  if (!pte.ok()) {
    return pte.error();
  }
  NSMutableSet<NSString*>* model_prefixes = [NSMutableSet set];
  for (const auto& buffer : pte->processed) {
    auto data = buffer.data_safe();
    if (!data.ok()) {
      return data.error();
    }
    ET_CHECK_OR_RETURN_ERROR(
        data.get() != nullptr && buffer.size() > 0,
        InvalidProgram,
        "Missing Core AI manifest");
    auto manifest = parse_manifest(data.get(), buffer.size());
    if (!manifest.ok()) {
      return manifest.error();
    }
    for (NSString* bundle_name in manifest->bundle_digests) {
      auto prefix = cache_model_key(manifest.get(), bundle_name);
      if (!prefix.ok()) {
        return prefix.error();
      }
      [model_prefixes addObject:prefix.get()];
    }
  }
  if (model_prefixes.count == 0) {
    return Error::Ok;
  }
  auto inventory = inventory_cache_keys(root);
  if (!inventory.ok()) {
    return inventory.error();
  }
  NSMutableArray<NSString*>* keys = [NSMutableArray array];
  for (NSString* key in inventory.get()) {
    if ([model_prefixes containsObject:cache_key_model_prefix(key)]) {
      [keys addObject:key];
    }
  }
  return clear_keys(root, keys);
}
} // namespace

Error clear_cache(const char* coreai_assets_dir) {
  @autoreleasepool {
    auto root = cache_root(coreai_assets_dir);
    if (!root.ok()) {
      return root.error();
    }
    auto keys = inventory_cache_keys(root.get());
    if (!keys.ok()) {
      return keys.error();
    }
    return clear_keys(root.get(), keys.get());
  }
}

Error clear_cache_for_pte(
    runtime::DataLoader& loader,
    const char* coreai_assets_dir) {
  @autoreleasepool {
    auto root = cache_root(coreai_assets_dir);
    if (!root.ok()) {
      return root.error();
    }
    return clear_pte(loader, root.get());
  }
}

Error clear_cache_for_pte(const char* pte_path, const char* coreai_assets_dir) {
  ET_CHECK_OR_RETURN_ERROR(
      pte_path != nullptr && pte_path[0] != '\0',
      InvalidArgument,
      "Missing PTE path");
  auto loader = extension::FileDataLoader::from(pte_path);
  if (!loader.ok()) {
    return loader.error();
  }
  return clear_cache_for_pte(loader.get(), coreai_assets_dir);
}

} // namespace executorch::backends::coreai
