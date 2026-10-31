/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#import "coreai_load_coordinator.h"

#import "coreai_cache_entry.h"
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

Error require_coreai_available() {
  ET_CHECK_OR_RETURN_ERROR(
      ETCoreAIIsAvailable(),
      NotSupported,
      "Core AI requires macOS 27 or iOS 27");
  return Error::Ok;
}

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

void cleanup_source(const LockedCacheEntry& cache_entry) {
  const Error error = cache_entry.remove_source();
  if (error != Error::Ok) {
    ET_LOG(
        Error,
        "Cannot remove Core AI source bundle: error=%u",
        static_cast<unsigned>(error));
  }
}

Error evict_one(id<ETCoreAIModelLoading> loader, NSData* bookmark) {
  ET_CHECK_OR_RETURN_ERROR(
      fault_point(FaultPoint::BeforeEvict) == 0,
      AccessFailed,
      "Core AI eviction interrupted");
  dispatch_semaphore_t ready = dispatch_semaphore_create(0);
  __block NSError* failure = nil;
  [loader evictModelWithBookmark:bookmark
                      completion:^(NSError* error) {
                        failure = error;
                        dispatch_semaphore_signal(ready);
                      }];
  dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
  NSError* eviction_failure = failure;
  // A loader may retain the completion block after invoking it.
  failure = nil;
  if (eviction_failure != nil) {
    // A confirmed restore miss proves that a failed eviction is already done.
    @autoreleasepool {
      auto probe = restore(loader, bookmark);
      if (probe.ok() && probe.get() == nil) {
        return Error::Ok;
      }
    }
    return acquisition_error(eviction_failure);
  }
  ET_CHECK_OR_RETURN_ERROR(
      fault_point(FaultPoint::AfterEvict) == 0,
      AccessFailed,
      "Core AI eviction interrupted");
  return Error::Ok;
}
} // namespace

Result<SelectedHostBundle> select_host_bundle(const Manifest& manifest) {
  ET_CHECK_OR_RETURN_ERROR(
      [NSProcessInfo.processInfo
          isOperatingSystemAtLeastVersion:manifest.minimum_version],
      DelegateInvalidCompatibility,
      "Core AI model requires a newer operating system");
  ET_CHECK_OK_OR_RETURN_ERROR(require_coreai_available());
#if TARGET_OS_OSX
  NSString* platform = @"macOS";
#elif TARGET_OS_IOS
  NSString* platform = @"iOS";
#else
  NSString* platform = @"unsupported";
#endif
  NSString* architecture = ETCoreAIDeviceArchitectureName();
  auto selected = select_bundle(manifest, architecture, platform);
  if (!selected.ok()) {
    return selected.error();
  }
  return SelectedHostBundle{
      std::move(selected.get()), [platform copy], [architecture copy]};
}

Result<id<ETCoreAIPreparedModel>> acquire_model(
    const SelectedHostBundle& host_bundle,
    const runtime::NamedDataMap* named_data,
    const CacheDirectory& cache_directory,
    id<ETCoreAIModelLoading> loader) {
  ET_CHECK_OR_RETURN_ERROR(
      loader != nil, InvalidArgument, "Missing Core AI model loader");
  NSString* specialization = [loader.specializationIdentifier copy];
  auto key = cache_variant_key(
      host_bundle.bundle,
      host_bundle.platform,
      host_bundle.architecture,
      specialization);
  if (!key.ok()) {
    return key.error();
  }
  NSString* cache_key = key.get();
  ET_LOG(Info, "Core AI cache lookup: key=%s", cache_key.UTF8String);
  auto locked = LockedCacheEntry::acquire(cache_directory, cache_key);
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
      host_bundle.bundle,
      named_data,
      *locked.get(),
      CorruptBundlePolicy::Replace);
  if (!source.ok()) {
    return source.error();
  }
  NSURL* url = source.get();
  auto model =
      wait_for_acquisition(^(ETCoreAIAcquisitionCompletion completion) {
        ET_LOG(
            Info,
            "Core AI specialization started: kind=%s, key=%s",
            host_bundle.bundle.aot_compiled ? "aot" : "source",
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

Error clear_cache_entry(
    const CacheDirectory& cache_directory,
    NSString* key,
    id<ETCoreAIModelLoading> loader) {
  auto locked = LockedCacheEntry::acquire_existing(cache_directory, key);
  if (!locked.ok()) {
    return locked.error();
  }
  if (locked.get() == nullptr) {
    return Error::Ok;
  }
  ET_LOG(Info, "Core AI cache clear started: key=%s", key.UTF8String);
  auto maintenance = locked.get()->maintenance();
  auto bookmark = maintenance.inspect_bookmark();
  if (!bookmark.ok()) {
    return bookmark.error();
  }
  ET_CHECK_OK_OR_RETURN_ERROR(maintenance.preflight());
  if (bookmark->state != CacheBookmarkState::Present) {
    ET_LOG(
        Info,
        "Core AI cache key %s has a %s bookmark; SDK eviction could not be "
        "confirmed; attempting local cleanup",
        key.UTF8String,
        bookmark->state == CacheBookmarkState::Missing ? "missing" : "corrupt");
  } else {
    if (loader == nil) {
      ET_CHECK_OK_OR_RETURN_ERROR(require_coreai_available());
      // Eviction targets the bookmark, so the loader's specialization is
      // irrelevant.
      loader = ETCoreAICreateModelLoader(ETCoreAIComputeUnitDefault);
      ET_CHECK_OR_RETURN_ERROR(
          loader != nil, Internal, "Cannot create Core AI model loader");
    }
    ET_LOG(Info, "Core AI SDK eviction requested: key=%s", key.UTF8String);
    ET_CHECK_OK_OR_RETURN_ERROR(evict_one(loader, bookmark->data));
    ET_LOG(
        Info,
        "Core AI SDK cache entry absence confirmed: key=%s",
        key.UTF8String);
  }
  ET_CHECK_OK_OR_RETURN_ERROR(maintenance.remove_source());
  ET_CHECK_OK_OR_RETURN_ERROR(maintenance.remove_bookmark());
  ET_LOG(Info, "Core AI local cache entry cleared: key=%s", key.UTF8String);
  return Error::Ok;
}

} // namespace executorch::backends::coreai
