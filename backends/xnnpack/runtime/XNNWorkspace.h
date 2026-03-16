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

#include <atomic>
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
  XNNWorkspace(WorkspacePtr workspace)
      : id_(next_id_++), workspace_(std::move(workspace)){};
  XNNWorkspace(const XNNWorkspace&) = delete;
  XNNWorkspace& operator=(const XNNWorkspace&) = delete;
  // Not moveable due to std::mutex.
  XNNWorkspace(XNNWorkspace&&) = delete;
  XNNWorkspace& operator=(XNNWorkspace&&) = delete;

  std::pair<std::unique_lock<std::mutex>, xnn_workspace_t> acquire() {
    if (!lock_required_) {
      return {std::unique_lock<std::mutex>{}, workspace_.get()};
    }
    auto lock = std::unique_lock<std::mutex>(mutex_);
    return {std::move(lock), workspace_.get()};
  }

  // Return the workspace pointer withot acquiring the lock. This should be used
  // carefully, as it can lead to crashes or data corruption if the workspace is
  // used concurrently.s
  xnn_workspace_t unsafe_get_workspace() {
    return workspace_.get();
  }

  // Returns a unique ID for this workspace instance. This can be used to
  // distinguish between different workspace objects even if they happen to
  // have the same raw pointer due to memory reuse.
  uint64_t id() const {
    return id_;
  }

  void disable_locking() {
    lock_required_ = false;
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
  static inline std::atomic<uint64_t> next_id_{0};
  std::mutex mutex_;
  uint64_t id_;
  bool lock_required_ = true;
  WorkspacePtr workspace_;
};

} // namespace executorch::backends::xnnpack
