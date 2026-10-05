/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "coreai_cache.h"

#import "coreai_assets.h"
#import "coreai_load_coordinator.h"
#include "coreai_pte.h"
#import "coreai_storage.h"

#include <TargetConditionals.h>
#include <executorch/extension/data_loader/file_data_loader.h>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

Result<NSString*> explicit_root(const char* path) {
  NSString* root = path == nullptr ? nil : [NSString stringWithUTF8String:path];
  return resolve_bookmark_root(root, false);
}

Error clear_keys(NSString* root, NSArray<NSString*>* keys) {
  NSArray<NSString*>* ordered = [[[NSSet setWithArray:keys] allObjects]
      sortedArrayUsingSelector:@selector(compare:)];
  Error first_error = Error::Ok;
  for (NSString* key in ordered) {
    Error error = clear_bookmark_assets(root, key, nil);
    if (error != Error::Ok) {
      ET_LOG(Error, "Core AI cache clear failed for key %s: 0x%x",
             key.UTF8String, static_cast<unsigned>(error));
      if (first_error == Error::Ok) first_error = error;
    }
  }
  return first_error;
}

Error clear_pte(runtime::DataLoader& loader, NSString* root) {
  // Keep the verified image and original segment owners alive through eviction.
  auto pte = inspect_coreai_pte(loader);
  if (!pte.ok()) return pte.error();
  NSMutableArray<NSString*>* keys = [NSMutableArray array];
  NSString* architecture = nil;
#if TARGET_OS_OSX
  NSString* platform = @"macOS";
#elif TARGET_OS_IOS
  NSString* platform = @"iOS";
#else
  NSString* platform = @"unsupported";
#endif
  for (const auto& buffer : pte->processed) {
    auto data = buffer.data_safe();
    if (!data.ok()) return data.error();
    ET_CHECK_OR_RETURN_ERROR(data.get() != nullptr && buffer.size() > 0,
                             InvalidProgram, "Missing Core AI manifest");
    NSData* bytes = [NSData dataWithBytesNoCopy:const_cast<void*>(data.get())
                                       length:buffer.size()
                                 freeWhenDone:NO];
    auto manifest = parse_manifest(bytes);
    if (!manifest.ok()) return manifest.error();
    ET_CHECK_OR_RETURN_ERROR(
        [NSProcessInfo.processInfo
            isOperatingSystemAtLeastVersion:manifest->minimum_version],
        DelegateInvalidCompatibility,
        "Core AI model requires a newer operating system");
    if (architecture == nil) {
      ET_CHECK_OR_RETURN_ERROR(ETCoreAIIsAvailable(), NotSupported,
                               "Core AI requires macOS 27 or iOS 27");
      architecture = ETCoreAIDeviceArchitectureName();
    }
    auto selected = select_assets(manifest.get(), architecture, platform);
    if (!selected.ok()) return selected.error();
    auto key = bookmark_key(selected.get(), platform, architecture);
    if (!key.ok()) return key.error();
    [keys addObject:key.get()];
  }
  return clear_keys(root, keys);
}
}  // namespace

Error clear_cache(const char* coreai_assets_dir) {
  @autoreleasepool {
    auto root = explicit_root(coreai_assets_dir);
    if (!root.ok()) return root.error();
    auto keys = inventory_bookmark_keys(root.get());
    if (!keys.ok()) return keys.error();
    return clear_keys(root.get(), keys.get());
  }
}

Error clear_cache_for_pte(runtime::DataLoader& loader,
                          const char* coreai_assets_dir) {
  @autoreleasepool {
    if (coreai_assets_dir != nullptr) {
      auto root = explicit_root(coreai_assets_dir);
      if (!root.ok()) return root.error();
      return clear_pte(loader, root.get());
    }
    auto default_root = default_coreai_assets_root();
    if (!default_root.ok()) return default_root.error();
    auto root = resolve_bookmark_root(default_root.get(), false);
    if (!root.ok()) return root.error();
    return clear_pte(loader, root.get());
  }
}

Error clear_cache_for_pte(const char* pte_path, const char* coreai_assets_dir) {
  ET_CHECK_OR_RETURN_ERROR(pte_path != nullptr && pte_path[0] != '\0',
                           InvalidArgument, "Missing PTE path");
  auto loader = extension::FileDataLoader::from(pte_path);
  if (!loader.ok()) return loader.error();
  return clear_cache_for_pte(loader.get(), coreai_assets_dir);
}

}  // namespace executorch::backends::coreai
