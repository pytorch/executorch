/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/runtime/core/error.h>

namespace executorch::runtime {
class DataLoader;
}

namespace executorch::backends::coreai {

/**
 * Clears tracked SDK entries, keyed staging, and bookmarks in a dedicated
 * Core AI assets directory. Null selects the same default as loading; supplied
 * paths must be absolute. Missing roots are successful no-ops.
 * PTEs, stable lock files, and unkeyed temporaries are not removed.
 * SDK entries without saved bookmarks cannot be targeted.
 *
 * Callers must unload affected models and prevent concurrent loading,
 * inference, and maintenance across processes until this call returns.
 * Independent keys are attempted in sorted order, returning the first error;
 * clearing is not a transaction. SDK failures preserve an entry's files unless
 * the SDK confirms that entry is absent.
 */
[[nodiscard]] runtime::Error clear_cache(
    const char* coreai_assets_dir = nullptr);

/**
 * Clears current-platform/SDK-architecture cache entries referenced by every
 * CoreAI delegate in the PTE, after validating all selected manifests. Invalid
 * or incompatible delegates fail before any entries are cleared. Does not
 * initialize delegates or materialize Core AI assets.
 *
 * A null directory selects the same Caches default as backend initialization.
 * Supplied directories must be valid explicit roots; they never fall back to
 * the default. The directory is never derived from the PTE's location.
 * Copied PTEs can share cache keys. The quiescence and failure contract of
 * clear_cache also applies. Keep the loader alive and its data unchanged until
 * the call returns.
 */
[[nodiscard]] runtime::Error clear_cache_for_pte(
    runtime::DataLoader& loader,
    const char* coreai_assets_dir = nullptr);

/** FileDataLoader convenience overload with the same contract. */
[[nodiscard]] runtime::Error clear_cache_for_pte(
    const char* pte_path,
    const char* coreai_assets_dir = nullptr);

} // namespace executorch::backends::coreai
