/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#import "coreai_load_coordinator.h"
#import "coreai_storage.h"

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;

Error sdk_error(NSError* error) {
  ET_LOG(Error, "Core AI SDK error (%s, %ld): %s", error.domain.UTF8String,
         static_cast<long>(error.code), error.description.UTF8String);
  return Error::Internal;
}

Result<id<ETCoreAIPreparedModel>> acquire(
    void (^operation)(ETCoreAIAcquisitionCompletion)) {
  ET_CHECK_OR_RETURN_ERROR(storage_fault(StorageOperation::BeforeSDK) == 0,
                           AccessFailed, "Core AI acquisition interrupted");
  dispatch_semaphore_t ready = dispatch_semaphore_create(0);
  __block id<ETCoreAIPreparedModel> model = nil;
  __block NSError* error = nil;
  __block ETCoreAIAcquisitionStatus status = ETCoreAIAcquisitionStatusError;
  operation(^(ETCoreAIAcquisitionStatus result, id<ETCoreAIPreparedModel> value,
              NSError* failure) {
    status = result;
    model = value;
    error = failure;
    dispatch_semaphore_signal(ready);
  });
  // Do not abandon asynchronous source access on cancellation or timeout.
  dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
  id<ETCoreAIPreparedModel> acquired = model;
  // A loader may retain the completion block after invoking it.
  model = nil;
  if (error != nil) return sdk_error(error);
  ET_CHECK_OR_RETURN_ERROR(storage_fault(StorageOperation::AfterSDK) == 0,
                           AccessFailed, "Core AI acquisition interrupted");
  ET_CHECK_OR_RETURN_ERROR(
      (status == ETCoreAIAcquisitionStatusHit && acquired != nil) ||
          (status == ETCoreAIAcquisitionStatusMiss && acquired == nil),
      Internal, "Invalid Core AI acquisition result");
  return acquired;
}

Result<id<ETCoreAIPreparedModel>> restore(id<ETCoreAIModelLoading> loader,
                                          NSData* bookmark) {
  return acquire(^(ETCoreAIAcquisitionCompletion completion) {
    [loader restoreModelFromBookmark:bookmark completion:completion];
  });
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

Result<id<ETCoreAIPreparedModel>> acquire_bookmark_model(
    const Manifest& selected, const runtime::NamedDataMap* named_data,
    NSString* root, NSString* platform, NSString* architecture,
    id<ETCoreAIModelLoading> loader) {
  ET_CHECK_OR_RETURN_ERROR(loader != nil, InvalidArgument,
                           "Missing Core AI model loader");
  auto key = bookmark_key(selected, platform, architecture);
  if (!key.ok()) return key.error();
  auto locked = lock_bookmark(root, key.get());
  if (!locked.ok()) return locked.error();
  auto bookmark = read_bookmark(*locked.get());
  if (!bookmark.ok()) return bookmark.error();
  if (bookmark.get() != nil) {
    auto model = restore(loader, bookmark.get());
    if (!model.ok()) return model.error();
    if (model.get() != nil) return model;
  }
  auto staging = prepare_bookmark_staging(*locked.get());
  if (!staging.ok()) return staging.error();
  auto source =
      prepare_source_bundle(selected, named_data, staging.get(), key.get());
  if (!source.ok()) return source.error();
  NSURL* url = source.get();
  auto model = acquire(^(ETCoreAIAcquisitionCompletion completion) {
    [loader specializeModelAtURL:url completion:completion];
  });
  if (!model.ok()) return model.error();
  ET_CHECK_OR_RETURN_ERROR(model.get() != nil, Internal,
                           "Core AI specialization returned no model");
  ET_CHECK_OK_OR_RETURN_ERROR(
      write_bookmark(*locked.get(), [model.get() copyBookmarkData]));
  // SDK pins do not prove source independence; leave staged sources in place.
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
