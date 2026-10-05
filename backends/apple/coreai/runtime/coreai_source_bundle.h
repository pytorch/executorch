/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <Foundation/Foundation.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::coreai {

class CacheEntryLock;
class SelectedBundle;

enum class CorruptBundlePolicy {
  Reject,
  Replace,
};

// The cache entry remains locked for this call. Reuse validates the exact tree
// structure and sizes, not file contents.
runtime::Result<NSURL*> prepare_source_bundle(
    const SelectedBundle& bundle,
    const runtime::NamedDataMap* named_data,
    const CacheEntryLock& cache_entry,
    NSString* key,
    CorruptBundlePolicy corrupt_policy = CorruptBundlePolicy::Reject);

} // namespace executorch::backends::coreai
