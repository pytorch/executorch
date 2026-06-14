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
#include <executorch/backends/xnnpack/runtime/XNNWorkspaceManager.h>
#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <atomic>
#include <mutex>

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

  const std::string& get_packed_cache_path() const;
  void set_packed_cache_path(const std::string& path);

  // Shared XNNWeightsCache (one instance per backend, like the workspace
  // manager). The cache itself is not internally synchronized; callers
  // MUST hold weights_cache_mutex() around every weights_cache() call —
  // including reading the reference and calling any method on it. The
  // same mutex also protects packed_cache_path_, so a typical
  // load/init/compile sequence holds one lock for the whole block:
  //
  //   std::lock_guard lock(options.weights_cache_mutex());
  //   options.weights_cache().set_packed_cache_path(
  //       options.get_packed_cache_path());
  //   options.weights_cache().initialize_for_runtime(...);
  //   XNNCompiler::compileModel(..., &options.weights_cache(), ...);
  //
  // The mutex is intentionally exposed (rather than wrapping every
  // method) because XNNCompiler needs a raw cache pointer to pass into
  // XNNPACK callbacks that fire during xnn_create_runtime; those
  // callbacks must run under the same lock as the surrounding init.
  delegate::XNNWeightsCache& weights_cache();
  std::mutex& weights_cache_mutex();

  // Invokes save_packed_index() on the cache while holding the cache
  // mutex. Returns the cache's error code; the caller does not need to
  // grab the mutex itself. This is the entry point used by set_option()
  // when `save_weight_cache_on_disk_option_key` is requested.
  runtime::Error save_weights_cache_locked();

 private:
  XNNWorkspaceManager workspace_manager_;

  // Weights cache is shared across all delegate instances. Owned here so
  // that all backend-option-keyed state (workspace manager, weights cache,
  // packed-cache path) lives in a single place; XnnpackBackend holds an
  // XnnpackBackendOptions and delegates synchronization to its mutex.
  // Protects weights_cache_ AND packed_cache_path_ (init reads the path
  // while holding this lock and hands it to the cache).
  std::mutex weights_cache_mutex_;
  delegate::XNNWeightsCache weights_cache_;

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

  std::string packed_cache_path_;
};

} // namespace executorch::backends::xnnpack
