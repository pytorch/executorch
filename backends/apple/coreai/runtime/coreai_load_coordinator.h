/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/runtime/core/named_data_map.h>
#import "ETCoreAIBridge.h"
#import "coreai_cache_root.h"
#import "coreai_manifest.h"

#include <mutex>

// A loaded model and the lock that serializes execution on it; delegates that
// share the model share this object. The SDK does not document concurrent
// execution on one model as safe, and it produced wrong outputs in testing.
@interface ETCoreAILoadedModel : NSObject
@property(nonatomic, readonly) id<ETCoreAIPreparedModel> model;
- (instancetype)init NS_UNAVAILABLE;
- (std::mutex&)executionMutex;
@end

namespace executorch::backends::coreai {

struct SelectedHostBundle final {
  SelectedBundle bundle;
  NSString* const platform;
  NSString* const architecture;
};

runtime::Result<SelectedHostBundle> select_host_bundle(
    const Manifest& manifest);

// The cache entry lock outlives all SDK callbacks and bookmark publication.
// Cleanup after a restore hit or durable publication is best-effort.
runtime::Result<id<ETCoreAIPreparedModel>> acquire_model(
    const SelectedHostBundle& host_bundle,
    const runtime::NamedDataMap* named_data,
    const CacheDirectory& cache_directory,
    id<ETCoreAIModelLoading> loader);

// With share, returns the model another delegate in this process already holds
// for the same cache directory and cache key, otherwise acquires one and
// records it. Distinct cache directories never share, so each keeps its own
// bookmark. Entries last only while some caller retains the result.
runtime::Result<ETCoreAILoadedModel*> acquire_loaded_model(
    const SelectedHostBundle& host_bundle,
    const runtime::NamedDataMap* named_data,
    const CacheDirectory& cache_directory,
    id<ETCoreAIModelLoading> loader,
    bool share);

// Callers must keep affected loads and sessions quiescent across processes.
runtime::Error clear_cache_entry(
    const CacheDirectory& cache_directory,
    NSString* key,
    id<ETCoreAIModelLoading> loader);

} // namespace executorch::backends::coreai
