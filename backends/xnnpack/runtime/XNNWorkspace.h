/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/result.h>
#include <xnnpack.h>

#include <memory>
#include <mutex>
#include <utility>

namespace executorch::backends::xnnpack {

using WorkspacePtr =
    std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)>;

/// A lightweight wrapper around an underlying xnn_workspace_t instance, bundled
/// with appropriate synchronization.
class XNNWorkspace {
 public:
  XNNWorkspace(WorkspacePtr workspace) : workspace_(std::move(workspace)){};
  XNNWorkspace(const XNNWorkspace&) = delete;
  XNNWorkspace& operator=(const XNNWorkspace&) = delete;
  // Not moveable due to std::mutex.
  XNNWorkspace(XNNWorkspace&&) = delete;
  XNNWorkspace& operator=(XNNWorkspace&&) = delete;

  std::pair<std::unique_lock<std::mutex>, xnn_workspace_t> acquire() {
    auto lock = std::unique_lock<std::mutex>(mutex_);
    return {std::move(lock), workspace_.get()};
  }

  // Return the workspace pointer withot acquiring the lock. This should be used
  // carefully, as it can lead to crashes or data corruption if the workspace is
  // used concurrently.s
  xnn_workspace_t unsafe_get_workspace() {
    return workspace_.get();
  }

  static runtime::Result<std::shared_ptr<XNNWorkspace>> create() {
    // Because this class can't be moved, we need to construct it in-place.
    xnn_workspace_t workspace = nullptr;
    auto status = xnn_create_workspace(&workspace);
    if (status != xnn_status_success) {
      ET_LOG(
          Error,
          "Failed to create XNN workspace, XNNPACK status: 0x%x",
          (unsigned int)status);
      return runtime::Error::Internal;
    }

    return std::make_shared<XNNWorkspace>(
        WorkspacePtr(workspace, &xnn_release_workspace));
  }

 private:
  std::mutex mutex_;
  WorkspacePtr workspace_;
};

} // namespace executorch::backends::xnnpack
