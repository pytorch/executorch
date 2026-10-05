/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#import "coreai_load_coordinator.h"

#import "coreai_bookmarks.h"
#import "coreai_cache_key.h"
#import "coreai_error.h"
#import "coreai_fault_injection.h"
#import "coreai_source_bundle.h"

#include <TargetConditionals.h>
#include <utility>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

Result<id<ETCoreAIPreparedModel>> wait_for_acquisition(
    void (^operation)(ETCoreAIAcquisitionCompletion)) {
  ET_CHECK_OR_RETURN_ERROR(
      fault_point(FaultPoint::BeforeSDK) == 0,
      AccessFailed,
      "Core AI acquisition interrupted");
  dispatch_semaphore_t ready = dispatch_semaphore_create(0);
  __block id<ETCoreAIPreparedModel> model = nil;
  __block NSError* error = nil;
  __block ETCoreAIAcquisitionStatus status = ETCoreAIAcquisitionStatusError;
  operation(
      ^(ETCoreAIAcquisitionStatus result,
        id<ETCoreAIPreparedModel> value,
        NSError* failure) {
        status = result;
        model = value;
        error = failure;
        dispatch_semaphore_signal(ready);
      });
  // Wait without a timeout; the SDK may still be reading the staged source.
  dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
  id<ETCoreAIPreparedModel> acquired = model;
  NSError* failure = error;
  // A loader may retain the completion block after invoking it.
  model = nil;
  error = nil;
  if (failure != nil) {
    return acquisition_error(failure);
  }
  ET_CHECK_OR_RETURN_ERROR(
      fault_point(FaultPoint::AfterSDK) == 0,
      AccessFailed,
      "Core AI acquisition interrupted");
  ET_CHECK_OR_RETURN_ERROR(
      (status == ETCoreAIAcquisitionStatusHit && acquired != nil) ||
          (status == ETCoreAIAcquisitionStatusMiss && acquired == nil),
      Internal,
      "Invalid Core AI acquisition result");
  return acquired;
}

Result<id<ETCoreAIPreparedModel>> restore(
    id<ETCoreAIModelLoading> loader,
    NSData* bookmark) {
  return wait_for_acquisition(^(ETCoreAIAcquisitionCompletion completion) {
    [loader restoreModelFromBookmark:bookmark completion:completion];
  });
}

void cleanup_source(const CacheEntryLock& cache_entry) {
  const Error error = cache_entry.remove_staging();
  if (error != Error::Ok) {
    ET_LOG(
        Error,
        "Cannot clean Core AI staging: error=%u",
        static_cast<unsigned>(error));
  }
}

Error evict_one(id<ETCoreAIModelLoading> loader, NSData* bookmark) {
  ET_CHECK_OR_RETURN_ERROR(storage_fault(StorageOperation::BeforeEvict) == 0,
                           AccessFailed, "Core AI eviction interrupted");
  dispatch_semaphore_t ready = dispatch_semaphore_create(0);
  __block NSError* failure = nil;
  [loader evictModelWithBookmark:bookmark
                      completion:^(NSError* error) {
                        failure = error;
                        dispatch_semaphore_signal(ready);
                      }];
  dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
  if (failure != nil) {
    // Probe after failed deletion so our own probe cannot make the entry busy.
    @autoreleasepool {
      auto probe = restore(loader, bookmark);
      if (probe.ok() && probe.get() == nil) return Error::Ok;
    }
    return sdk_error(failure);
  }
  ET_CHECK_OR_RETURN_ERROR(storage_fault(StorageOperation::AfterEvict) == 0,
                           AccessFailed, "Core AI eviction interrupted");
  return Error::Ok;
}
}  // namespace

Result<RuntimeAssets> select_runtime_assets(const Manifest& manifest) {
  ET_CHECK_OR_RETURN_ERROR(
      [NSProcessInfo.processInfo
          isOperatingSystemAtLeastVersion:manifest.minimum_version],
      DelegateInvalidCompatibility,
      "Core AI model requires a newer operating system");
  ET_CHECK_OR_RETURN_ERROR(
      ETCoreAIIsAvailable(),
      NotSupported,
      "Core AI requires macOS 27 or iOS 27");
#if TARGET_OS_OSX
  NSString* platform = @"macOS";
#elif TARGET_OS_IOS
  NSString* platform = @"iOS";
#else
  NSString* platform = @"unsupported";
#endif
  NSString* architecture = ETCoreAIDeviceArchitectureName();
  auto selected = select_assets(manifest, architecture, platform);
  if (!selected.ok()) {
    return selected.error();
  }
  return RuntimeAssets{
      std::move(selected.get()), [platform copy], [architecture copy]};
}

Result<id<ETCoreAIPreparedModel>> acquire_bookmark_model(
    const RuntimeAssets& assets,
    const runtime::NamedDataMap* named_data,
    NSString* root,
    id<ETCoreAIModelLoading> loader) {
  ET_CHECK_OR_RETURN_ERROR(
      loader != nil, InvalidArgument, "Missing Core AI model loader");
  NSString* specialization = [loader.specializationIdentifier copy];
  auto key = cache_variant_key(
      assets.bundle, assets.platform, assets.architecture, specialization);
  if (!key.ok()) {
    return key.error();
  }
  NSString* cache_key = key.get();
  ET_LOG(Info, "Core AI cache lookup: key=%s", cache_key.UTF8String);
  auto locked = lock_cache_entry(root, cache_key);
  if (!locked.ok()) {
    return locked.error();
  }
  auto bookmark = locked.get()->read_bookmark();
  if (!bookmark.ok()) {
    return bookmark.error();
  }
  if (bookmark.get() != nil) {
    auto model = restore(loader, bookmark.get());
    if (!model.ok()) {
      return model.error();
    }
    if (model.get() != nil) {
      ET_LOG(
          Info,
          "Core AI cache hit: restored model from bookmark, key=%s",
          cache_key.UTF8String);
      cleanup_source(*locked.get());
      return model;
    }
    ET_LOG(
        Info,
        "Core AI cache miss: SDK could not restore bookmark, key=%s",
        cache_key.UTF8String);
  } else {
    ET_LOG(
        Info, "Core AI cache miss: no bookmark, key=%s", cache_key.UTF8String);
  }
  auto source = prepare_source_bundle(
      assets.bundle, named_data, *locked.get(), CorruptBundlePolicy::Replace);
  if (!source.ok()) {
    return source.error();
  }
  NSURL* url = source.get();
  auto model =
      wait_for_acquisition(^(ETCoreAIAcquisitionCompletion completion) {
        ET_LOG(
            Info,
            "Core AI specialization started: kind=%s, key=%s",
            assets.bundle.aot_compiled ? "aot" : "source",
            cache_key.UTF8String);
        [loader specializeModelAtURL:url completion:completion];
      });
  if (!model.ok()) {
    return model.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      model.get() != nil, Internal, "Core AI specialization returned no model");
  ET_LOG(
      Info, "Core AI specialization succeeded: key=%s", cache_key.UTF8String);
  ET_CHECK_OK_OR_RETURN_ERROR(
      locked.get()->write_bookmark([model.get() copyBookmarkData]));
  ET_LOG(
      Info,
      "Core AI cache bookmark saved durably: key=%s",
      cache_key.UTF8String);
  cleanup_source(*locked.get());
  return model;
}

Error evict_bookmark_model(NSString* root, NSString* key,
                           id<ETCoreAIModelLoading> loader) {
  ET_CHECK_OR_RETURN_ERROR(loader != nil, InvalidArgument,
                           "Missing Core AI model loader");
  auto locked = lock_bookmark(root, key);
  if (!locked.ok()) return locked.error();
  auto bookmark = read_bookmark(*locked.get());
  if (!bookmark.ok()) return bookmark.error();
  if (bookmark.get() == nil) return Error::Ok;
  ET_CHECK_OK_OR_RETURN_ERROR(evict_one(loader, bookmark.get()));
  return remove_bookmark(*locked.get());
}

Error clear_bookmark_assets(NSString* root, NSString* key,
                            id<ETCoreAIModelLoading> loader) {
  auto locked = lock_bookmark(root, key, false);
  if (!locked.ok()) return locked.error();
  if (locked.get() == nullptr) return Error::Ok;
  auto bookmark = read_bookmark(*locked.get());
  if (!bookmark.ok()) return bookmark.error();
  ET_CHECK_OK_OR_RETURN_ERROR(remove_bookmark_staging(*locked.get(), false));
  if (bookmark.get() != nil) {
    if (loader == nil) {
      ET_CHECK_OR_RETURN_ERROR(ETCoreAIIsAvailable(), NotSupported,
                               "Core AI requires macOS 27 or iOS 27");
      loader = ETCoreAICreateModelLoader();
      ET_CHECK_OR_RETURN_ERROR(loader != nil, Internal,
                               "Cannot create Core AI model loader");
    }
    ET_CHECK_OK_OR_RETURN_ERROR(evict_one(loader, bookmark.get()));
  }
  ET_CHECK_OK_OR_RETURN_ERROR(remove_bookmark_staging(*locked.get()));
  return bookmark.get() == nil ? Error::Ok : remove_bookmark(*locked.get());
}

}  // namespace executorch::backends::coreai
