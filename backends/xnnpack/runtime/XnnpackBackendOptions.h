/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/xnnpack/runtime/XNNPACKBackend.h>
#include <executorch/backends/xnnpack/runtime/XNNWeightsCache.h>
#include <executorch/backends/xnnpack/runtime/XNNWeightsCacheManager.h>
#include <executorch/backends/xnnpack/runtime/XNNWorkspaceManager.h>
#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

namespace executorch::backends::xnnpack {

class XnnpackBackendOptions {
 public:
  // Get a single option by key. The key field of the option must be set.
  runtime::Error get_option(runtime::BackendOption& option) const;

  // Set a single option by key. Validates type and domain.
  runtime::Error set_option(const runtime::BackendOption& option);

  // Resolve the effective weight cache setting for a delegate, applying
  // any runtime spec override.
  bool resolve_weight_cache(
      const ET_RUNTIME_NAMESPACE::BackendInitContext& context) const;

  // Resolve the effective workspace sharing mode for a delegate, applying
  // any runtime spec override. Returns InvalidArgument for out-of-range values.
  runtime::Result<WorkspaceSharingMode> resolve_sharing_mode(
      const ET_RUNTIME_NAMESPACE::BackendInitContext& context) const;

  WorkspaceSharingMode get_sharing_mode() const;
  XNNWorkspaceManager& workspace_manager();
  const XNNWorkspaceManager& workspace_manager() const;

  // Returns a copy of the most-recently-set packed cache path. Copied
  // under path_mutex_ to avoid tearing while set_option is concurrently
  // running. Callers receive a snapshot; subsequent set_option calls
  // do not affect the returned string.
  std::string get_packed_cache_path() const;
  void set_packed_cache_path(const std::string& path);

  // Returns a shared XNNWeightsCache via the backend-owned manager.
  // Same non-empty path → same shared instance. Different paths →
  // independent instances. Empty path → one shared heap-only instance
  // across all empty-path callers (so XNNPACK's in-memory name dedup
  // still works).
  //
  // The caller MUST hold the returned instance's XNNWeightsCache::mutex()
  // around every cache-method call, including the XNNPACK callbacks
  // invoked during xnn_create_runtime.
  runtime::Result<std::shared_ptr<delegate::XNNWeightsCache>>
  get_or_create_weights_cache(const std::string& cache_file_path);

  // Returns the manager directly. Useful for tests and for the
  // `save_weight_cache_on_disk` option side-effect path. Production
  // callers should prefer get_or_create_weights_cache().
  XNNWeightsCacheManager& weights_cache_manager();
  const XNNWeightsCacheManager& weights_cache_manager() const;

  // Walks every live cache instance the manager has handed out and
  // invokes save_packed_index() on each (under each instance's own
  // mutex). Used by set_option() when `save_weight_cache_on_disk` is
  // requested. Returns the first error encountered; continues
  // attempting every instance.
  runtime::Error save_weights_cache_locked();

 private:
  XNNWorkspaceManager workspace_manager_;
  XNNWeightsCacheManager weights_cache_manager_;

#ifdef ENABLE_XNNPACK_SHARED_WORKSPACE
  std::atomic<WorkspaceSharingMode> sharing_mode_{WorkspaceSharingMode::Global};
#else
  std::atomic<WorkspaceSharingMode> sharing_mode_{
      WorkspaceSharingMode::Disabled};
#endif

#ifdef ENABLE_XNNPACK_WEIGHTS_CACHE
  std::atomic<bool> weight_cache_enabled_{true};
#else
  std::atomic<bool> weight_cache_enabled_{false};
#endif

  // The most-recently-set packed cache path. Today's callers (Cria
  // runner) set this via set_option(packed_cache_path_option_key) per
  // PTE, then invoke init() which reads it and calls
  // get_or_create_weights_cache(). path_mutex_ serializes the
  // set / get pair so a concurrent set_option from another caller
  // doesn't tear the string mid-read.
  //
  // This is a transport for the path option only — the path itself
  // is owned per-cache-instance inside XNNWeightsCache, set once by
  // the manager before publishing the shared_ptr.
  mutable std::mutex path_mutex_;
  std::string packed_cache_path_;
};

} // namespace executorch::backends::xnnpack
