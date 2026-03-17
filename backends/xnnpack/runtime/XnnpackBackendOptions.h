/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/xnnpack/runtime/XNNPACKBackend.h>
#include <executorch/backends/xnnpack/runtime/XNNWorkspaceManager.h>
#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <atomic>

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

 private:
  XNNWorkspaceManager workspace_manager_;

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
};

} // namespace executorch::backends::xnnpack
