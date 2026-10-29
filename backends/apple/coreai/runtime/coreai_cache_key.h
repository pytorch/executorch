/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#import <Foundation/Foundation.h>
#include <executorch/runtime/core/result.h>
#include <cstddef>

namespace executorch::backends::coreai {

class Manifest;
class SelectedBundle;

bool is_cache_key(NSString* key);
NSString* cache_key_model_prefix(NSString* key);

// Model identity excludes device and specialization so maintenance can find all
// variants of one exported bundle.
runtime::Result<NSString*> cache_model_key(
    const Manifest& manifest,
    NSString* bundle_name);
runtime::Result<NSString*> cache_model_key(const SelectedBundle& bundle);
// specialization_identity is stable, versioned plain data supplied by the SDK
// boundary. This module does not depend on SDK types.
runtime::Result<NSString*> cache_variant_key(
    const SelectedBundle& bundle,
    NSString* platform,
    NSString* architecture,
    NSString* specialization_identity);

constexpr size_t kCacheEntryLockStripeCount = 256;
runtime::Result<NSString*> cache_entry_lock_name(NSString* key);

} // namespace executorch::backends::coreai
