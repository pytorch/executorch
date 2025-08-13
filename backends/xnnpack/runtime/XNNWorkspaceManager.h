/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/xnnpack/runtime/XNNPACKBackend.h>
#include <executorch/backends/xnnpack/runtime/XNNWorkspace.h>
#include <executorch/runtime/core/result.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace executorch::backends::xnnpack {

/**
 * XNNWorkspaceManager manages XNNPACK workspaces based on the configured
 * workspace sharing mode.
 *
 * It supports three modes:
 * - Disabled: Each delegate instance gets its own workspace
 * - PerModel: All delegate instances in a model share a workspace
 * - Global: All delegate instances across all models share a workspace
 */
class XNNWorkspaceManager {
 public:
  XNNWorkspaceManager();
  ~XNNWorkspaceManager() = default;

  /**
   * Set the workspace sharing mode.
   *
   * @param mode The workspace sharing mode to set.
   * @return Error::Ok if the mode was set successfully.
   */
  runtime::Error set_sharing_mode(WorkspaceSharingMode mode);

  /**
   * Get the current workspace sharing mode.
   *
   * @return The current workspace sharing mode.
   */
  WorkspaceSharingMode get_sharing_mode() const;

  /**
   * Retrieve a workspace for the given program ID, depending on the sharing
   * mode. A workspace will be created if needed.
   *
   * @param program_id The ID of the program requesting a workspace.
   * @return A Result containing a shared_ptr to the workspace, or an error.
   */
  runtime::Result<std::shared_ptr<XNNWorkspace>> get_or_create_workspace(
      uintptr_t program_id) const;

 private:
  // The active sharing mode. Changes to this affect only models loaded after
  // the change.
  std::atomic<WorkspaceSharingMode> sharing_mode_;

  // A mutex guarding global_workspace_ and model_workspaces_. Note that this
  // mutex only guards the top-level definitions, not the contents of the
  // workspace. The contents of the workspace are guarded by the workspace's own
  // mutex in the XNNWorkspace class.
  mutable std::mutex workspace_meta_mutex_;

  // A global workspace for all delegate instances, if global sharing is
  // enabled. Lazy initialized. Stored as a weak pointer to allow automatic
  // cleanup when all references are released.
  mutable std::weak_ptr<XNNWorkspace> global_workspace_;

  // A map from program id to workspace for delegate instances, if per model
  // sharing is enabled. Workspaces are owned by the executor instances via
  // shared_ptr. They are tracked here via weak pointers to allow automatic
  // cleanup when the executors are destroyed while being retrievable when
  // instantiating new executors.
  mutable std::unordered_map<uintptr_t, std::weak_ptr<XNNWorkspace>>
      model_workspaces_;

  // Retrieve the global workspace, lazy initializing it if needed.
  runtime::Result<std::shared_ptr<XNNWorkspace>>
  get_or_create_global_workspace() const;

  // Get or create a workspace for the given program ID.
  runtime::Result<std::shared_ptr<XNNWorkspace>> get_or_create_model_workspace(
      uintptr_t program_id) const;
};

} // namespace executorch::backends::xnnpack
