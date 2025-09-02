/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNWorkspaceManager.h>
#include <executorch/runtime/core/error.h>
#include <cinttypes> // For PRIuPTR

namespace executorch::backends::xnnpack {

using executorch::runtime::Error;
using executorch::runtime::Result;

XNNWorkspaceManager::XNNWorkspaceManager() {
#ifdef ENABLE_XNNPACK_SHARED_WORKSPACE
  sharing_mode_ = WorkspaceSharingMode::Global;
#else
  sharing_mode_ = WorkspaceSharingMode::Disabled;
#endif // ENABLE_XNNPACK_SHARED_WORKSPACE
}

runtime::Error XNNWorkspaceManager::set_sharing_mode(
    WorkspaceSharingMode mode) {
  // Validate that the mode is valid
  if (static_cast<int>(mode) < 0 ||
      static_cast<int>(mode) >= static_cast<int>(WorkspaceSharingMode::Count)) {
    ET_LOG(
        Error,
        "XNNPACK workspace sharing mode must be between 0 and %d, inclusive, but was %d.",
        static_cast<int>(WorkspaceSharingMode::Count) - 1,
        static_cast<int>(mode));
    return runtime::Error::InvalidArgument;
  }

  sharing_mode_ = mode;
  return runtime::Error::Ok;
}

WorkspaceSharingMode XNNWorkspaceManager::get_sharing_mode() const {
  return sharing_mode_.load();
}

Result<std::shared_ptr<XNNWorkspace>>
XNNWorkspaceManager::get_or_create_workspace(uintptr_t program_id) const {
  auto mode = sharing_mode_.load();

  // Get or create the workspace according to the current sharing mode.
  if (mode == WorkspaceSharingMode::Disabled) {
    ET_LOG(Debug, "Instantiating workspace.");
    auto create_result = XNNWorkspace::create();
    if (!create_result.ok()) {
      return create_result.error();
    }

    return create_result.get();
  } else if (mode == WorkspaceSharingMode::PerModel) {
    return get_or_create_model_workspace(program_id);
  } else if (mode == WorkspaceSharingMode::Global) {
    return get_or_create_global_workspace();
  } else {
    ET_LOG(
        Error, "Invalid workspace sharing mode: %d.", static_cast<int>(mode));
    return Error::Internal;
  }
}

Result<std::shared_ptr<XNNWorkspace>>
XNNWorkspaceManager::get_or_create_global_workspace() const {
  std::scoped_lock<std::mutex> lock(workspace_meta_mutex_);

  // Check for an existing (live) global workspace.
  std::shared_ptr<XNNWorkspace> workspace = {};
  if (auto live_workspace = global_workspace_.lock()) {
    workspace = live_workspace;
  }

  // Allocate a new workspace if needed.
  if (!workspace) {
    auto create_result = XNNWorkspace::create();
    if (!create_result.ok()) {
      return create_result.error();
    }
    workspace = create_result.get();
    ET_LOG(
        Debug,
        "Created global workspace %p.",
        workspace->unsafe_get_workspace());
    global_workspace_ = workspace;
  }

  return workspace;
}

Result<std::shared_ptr<XNNWorkspace>>
XNNWorkspaceManager::get_or_create_model_workspace(uintptr_t program_id) const {
  std::scoped_lock<std::mutex> lock(workspace_meta_mutex_);

  // Check for an existing (live) workspace for this program.
  auto match = model_workspaces_.find(program_id);
  std::shared_ptr<XNNWorkspace> workspace = {};
  if (match != model_workspaces_.end()) {
    if (auto live_workspace = match->second.lock()) {
      workspace = live_workspace;
    }
  }

  // Allocate a new workspace if needed.
  if (!workspace) {
    auto create_result = XNNWorkspace::create();
    if (!create_result.ok()) {
      return create_result.error();
    }
    workspace = create_result.get();
    ET_LOG(
        Debug,
        "Created workspace %p for program %" PRIuPTR ".",
        workspace->unsafe_get_workspace(),
        program_id);
    model_workspaces_.insert(
        {program_id, std::weak_ptr<XNNWorkspace>(workspace)});
  }

  return workspace;
}

} // namespace executorch::backends::xnnpack
