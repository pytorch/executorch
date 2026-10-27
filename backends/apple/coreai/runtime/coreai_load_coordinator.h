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
#import "coreai_manifest.h"

namespace executorch::backends::coreai {

struct RuntimeAssets final {
  SelectedBundle bundle;
  NSString* const platform;
  NSString* const architecture;
};

runtime::Result<RuntimeAssets> select_runtime_assets(const Manifest& manifest);

// The cache entry lock outlives all SDK callbacks and bookmark publication.
// Cleanup after a restore hit or durable publication is best-effort.
runtime::Result<id<ETCoreAIPreparedModel>> acquire_bookmark_model(
    const RuntimeAssets& assets,
    const runtime::NamedDataMap* named_data,
    NSString* root,
    id<ETCoreAIModelLoading> loader);

// Callers must keep affected loads and sessions quiescent across processes.
runtime::Error clear_bookmark_assets(
    NSString* root,
    NSString* key,
    id<ETCoreAIModelLoading> loader);

} // namespace executorch::backends::coreai
