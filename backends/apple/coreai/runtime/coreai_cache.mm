/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "coreai_cache.h"

#import "coreai_cache_entry.h"
#import "coreai_cache_key.h"
#import "coreai_cache_root.h"
#import "coreai_load_coordinator.h"
#import "coreai_manifest.h"
#include "coreai_pte.h"

#include <executorch/extension/data_loader/file_data_loader.h>
#include <optional>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

Result<std::optional<CacheDirectory>> cache_directory(const char* path) {
  auto configured = cache_directory_path(path);
  if (!configured.ok()) {
    return configured.error();
  }
  return resolve_cache_directory(
      configured.get(), StorageCreationMode::OpenExisting);
}

Error clear_keys(const CacheDirectory& directory, NSArray<NSString*>* keys) {
  NSArray<NSString*>* ordered = [[[NSSet setWithArray:keys] allObjects]
      sortedArrayUsingSelector:@selector(compare:)];
  Error first_error = Error::Ok;
  for (NSString* key in ordered) {
    const Error error = clear_cache_entry(directory, key, nil);
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

Error clear_pte(
    runtime::DataLoader& loader,
    const std::optional<CacheDirectory>& directory) {
  // Keep the verified image and selected segment owners alive through eviction.
  auto pte = inspect_coreai_pte(loader);
  if (!pte.ok()) {
    return pte.error();
  }
  NSMutableSet<NSString*>* bundle_identities = [NSMutableSet set];
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
      auto identity = bundle_identity(manifest.get(), bundle_name);
      if (!identity.ok()) {
        return identity.error();
      }
      [bundle_identities addObject:identity.get()];
    }
  }
  if (bundle_identities.count == 0 || !directory.has_value()) {
    return Error::Ok;
  }
  auto inventory = inventory_cache_keys(*directory);
  if (!inventory.ok()) {
    return inventory.error();
  }
  NSMutableArray<NSString*>* keys = [NSMutableArray array];
  for (NSString* key in inventory.get()) {
    if ([bundle_identities containsObject:cache_key_bundle_identity(key)]) {
      [keys addObject:key];
    }
  }
  return clear_keys(*directory, keys);
}
} // namespace

Error clear_cache(const char* coreai_cache_dir) {
  @autoreleasepool {
    auto directory = cache_directory(coreai_cache_dir);
    if (!directory.ok()) {
      return directory.error();
    }
    if (!directory->has_value()) {
      return Error::Ok;
    }
    auto keys = inventory_cache_keys(directory->value());
    if (!keys.ok()) {
      return keys.error();
    }
    return clear_keys(directory->value(), keys.get());
  }
}

Error clear_cache_for_pte(
    runtime::DataLoader& loader,
    const char* coreai_cache_dir) {
  @autoreleasepool {
    auto directory = cache_directory(coreai_cache_dir);
    if (!directory.ok()) {
      return directory.error();
    }
    return clear_pte(loader, directory.get());
  }
}

Error clear_cache_for_pte(const char* pte_path, const char* coreai_cache_dir) {
  ET_CHECK_OR_RETURN_ERROR(
      pte_path != nullptr && pte_path[0] != '\0',
      InvalidArgument,
      "Missing PTE path");
  auto loader = extension::FileDataLoader::from(pte_path);
  if (!loader.ok()) {
    return loader.error();
  }
  return clear_cache_for_pte(loader.get(), coreai_cache_dir);
}

} // namespace executorch::backends::coreai
