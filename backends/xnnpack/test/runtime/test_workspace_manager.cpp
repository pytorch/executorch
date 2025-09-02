/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/XNNWorkspace.h>
#include <executorch/backends/xnnpack/runtime/XNNWorkspaceManager.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

#include <xnnpack.h>

using namespace ::testing;

using executorch::backends::xnnpack::WorkspaceSharingMode;
using executorch::backends::xnnpack::XNNWorkspace;
using executorch::backends::xnnpack::XNNWorkspaceManager;
using executorch::runtime::Error;
using executorch::runtime::Result;

class XNNWorkspaceManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Log calls will abort if PAL is not initialized.
    executorch::runtime::runtime_init();

    // Initialize a new workspace manager for each test.
    workspace_manager_ = std::make_unique<XNNWorkspaceManager>();
  }

  std::unique_ptr<XNNWorkspaceManager> workspace_manager_;
};

TEST_F(XNNWorkspaceManagerTest, SetAndGetSharingMode) {
  // Test setting and getting the sharing mode
  EXPECT_EQ(
      workspace_manager_->set_sharing_mode(WorkspaceSharingMode::Disabled),
      Error::Ok);
  EXPECT_EQ(
      workspace_manager_->get_sharing_mode(), WorkspaceSharingMode::Disabled);

  EXPECT_EQ(
      workspace_manager_->set_sharing_mode(WorkspaceSharingMode::PerModel),
      Error::Ok);
  EXPECT_EQ(
      workspace_manager_->get_sharing_mode(), WorkspaceSharingMode::PerModel);

  EXPECT_EQ(
      workspace_manager_->set_sharing_mode(WorkspaceSharingMode::Global),
      Error::Ok);
  EXPECT_EQ(
      workspace_manager_->get_sharing_mode(), WorkspaceSharingMode::Global);
}

TEST_F(XNNWorkspaceManagerTest, SetInvalidSharingMode) {
  // First set a valid mode to ensure we're starting from a known state.
  EXPECT_EQ(
      workspace_manager_->set_sharing_mode(WorkspaceSharingMode::Disabled),
      Error::Ok);
  EXPECT_EQ(
      workspace_manager_->get_sharing_mode(), WorkspaceSharingMode::Disabled);

  // Try to set an invalid mode.
  WorkspaceSharingMode invalid_mode = static_cast<WorkspaceSharingMode>(70);
  EXPECT_EQ(
      workspace_manager_->set_sharing_mode(invalid_mode),
      Error::InvalidArgument);

  // The mode should not have changed.
  EXPECT_EQ(
      workspace_manager_->get_sharing_mode(), WorkspaceSharingMode::Disabled);
}

TEST_F(XNNWorkspaceManagerTest, DisabledMode) {
  // Verify that each call retrieves a new workspace when sharing is disabled.
  workspace_manager_->set_sharing_mode(WorkspaceSharingMode::Disabled);

  uintptr_t program_id = 12345;
  auto workspace1_result =
      workspace_manager_->get_or_create_workspace(program_id);
  ASSERT_TRUE(workspace1_result.ok());
  auto workspace1 = workspace1_result.get();

  auto workspace2_result =
      workspace_manager_->get_or_create_workspace(program_id);
  ASSERT_TRUE(workspace2_result.ok());
  auto workspace2 = workspace2_result.get();

  auto workspace3_result =
      workspace_manager_->get_or_create_workspace(program_id + 1);
  ASSERT_TRUE(workspace3_result.ok());
  auto workspace3 = workspace3_result.get();

  EXPECT_NE(workspace1, workspace2);
  EXPECT_NE(workspace1, workspace3);
  EXPECT_NE(workspace2, workspace3);
  EXPECT_NE(
      workspace1->unsafe_get_workspace(), workspace2->unsafe_get_workspace());
  EXPECT_NE(
      workspace1->unsafe_get_workspace(), workspace3->unsafe_get_workspace());
  EXPECT_NE(
      workspace2->unsafe_get_workspace(), workspace3->unsafe_get_workspace());
}

TEST_F(XNNWorkspaceManagerTest, PerModelMode) {
  // In PerModel mode, calls with the same program_id should return the same
  // workspace.
  workspace_manager_->set_sharing_mode(WorkspaceSharingMode::PerModel);

  // Get two workspaces with the same program ID and one different.
  uintptr_t program_id = 12345;
  auto workspace1_result =
      workspace_manager_->get_or_create_workspace(program_id);
  ASSERT_TRUE(workspace1_result.ok());
  auto workspace1 = workspace1_result.get();

  auto workspace2_result =
      workspace_manager_->get_or_create_workspace(program_id);
  ASSERT_TRUE(workspace2_result.ok());
  auto workspace2 = workspace2_result.get();

  auto workspace3_result =
      workspace_manager_->get_or_create_workspace(program_id + 1);
  ASSERT_TRUE(workspace3_result.ok());
  auto workspace3 = workspace3_result.get();

  // Workspace 1 and 2 should be the same, but different from workspace 3.
  EXPECT_EQ(workspace1, workspace2);
  EXPECT_EQ(
      workspace1->unsafe_get_workspace(), workspace2->unsafe_get_workspace());

  EXPECT_NE(workspace1, workspace3);
  EXPECT_NE(
      workspace1->unsafe_get_workspace(), workspace3->unsafe_get_workspace());
}

TEST_F(XNNWorkspaceManagerTest, GlobalMode) {
  // In Global mode, all calls should return the same workspace.
  workspace_manager_->set_sharing_mode(WorkspaceSharingMode::Global);

  // Get workspaces with different program IDs
  uintptr_t program_id1 = 12345;
  auto workspace1_result =
      workspace_manager_->get_or_create_workspace(program_id1);
  ASSERT_TRUE(workspace1_result.ok());
  auto workspace1 = workspace1_result.get();

  uintptr_t program_id2 = 67890;
  auto workspace2_result =
      workspace_manager_->get_or_create_workspace(program_id2);
  ASSERT_TRUE(workspace2_result.ok());
  auto workspace2 = workspace2_result.get();

  EXPECT_EQ(workspace1, workspace2);
  EXPECT_EQ(
      workspace1->unsafe_get_workspace(), workspace2->unsafe_get_workspace());
}

TEST_F(XNNWorkspaceManagerTest, PerModelModeCleanup) {
  // Test that workspaces are properly cleaned up when shared_ptr is destroyed
  workspace_manager_->set_sharing_mode(WorkspaceSharingMode::PerModel);

  uintptr_t program_id = 12345;
  xnn_workspace_t raw_workspace1 = nullptr;

  // Create a scope to control the lifetime of workspace1
  {
    auto workspace1_result =
        workspace_manager_->get_or_create_workspace(program_id);
    ASSERT_TRUE(workspace1_result.ok());
    auto workspace1 = workspace1_result.get();

    // Store the raw pointer for later comparison
    raw_workspace1 = workspace1->unsafe_get_workspace();

    // Let workspace1 go out of scope and be destroyed
  }

  // Get a new workspace with the same program ID
  auto workspace2_result =
      workspace_manager_->get_or_create_workspace(program_id);
  ASSERT_TRUE(workspace2_result.ok());
  auto workspace2 = workspace2_result.get();

  // Since the previous workspace was destroyed, we should get a new one.
  EXPECT_NE(workspace2->unsafe_get_workspace(), raw_workspace1);
}

TEST_F(XNNWorkspaceManagerTest, GlobalModeCleanup) {
  // Test that global workspaces are properly cleaned up when all users
  // are destroyed.
  workspace_manager_->set_sharing_mode(WorkspaceSharingMode::Global);

  uintptr_t program_id = 12345;
  xnn_workspace_t raw_workspace1 = nullptr;

  // Create a scope to control the lifetime of workspace1
  {
    auto workspace1_result =
        workspace_manager_->get_or_create_workspace(program_id);
    ASSERT_TRUE(workspace1_result.ok());
    auto workspace1 = workspace1_result.get();

    // Store the raw pointer for later comparison
    raw_workspace1 = workspace1->unsafe_get_workspace();

    // Let workspace1 go out of scope and be destroyed
  }

  // Get a new workspace (program ID doesn't matter in Global mode)
  auto workspace2_result =
      workspace_manager_->get_or_create_workspace(program_id);
  ASSERT_TRUE(workspace2_result.ok());
  auto workspace2 = workspace2_result.get();

  // Since the previous workspace was destroyed, we should get a new one.
  EXPECT_NE(workspace2->unsafe_get_workspace(), raw_workspace1);
}

TEST_F(XNNWorkspaceManagerTest, SwitchingModes) {
  // Test switching between different sharing modes

  // Start with Disabled mode
  workspace_manager_->set_sharing_mode(WorkspaceSharingMode::Disabled);

  // Get a workspace
  uintptr_t program_id = 12345;
  auto workspace1_result =
      workspace_manager_->get_or_create_workspace(program_id);
  ASSERT_TRUE(workspace1_result.ok());
  auto workspace1 = workspace1_result.get();

  // Switch to PerModel mode
  workspace_manager_->set_sharing_mode(WorkspaceSharingMode::PerModel);

  // Get another workspace with the same program ID
  auto workspace2_result =
      workspace_manager_->get_or_create_workspace(program_id);
  ASSERT_TRUE(workspace2_result.ok());
  auto workspace2 = workspace2_result.get();

  // Should be a different workspace
  EXPECT_NE(workspace1, workspace2);

  // Get another workspace with the same program ID in PerModel mode
  auto workspace3_result =
      workspace_manager_->get_or_create_workspace(program_id);
  ASSERT_TRUE(workspace3_result.ok());
  auto workspace3 = workspace3_result.get();

  // Should be the same workspace as workspace2
  EXPECT_EQ(workspace2, workspace3);

  // Switch to Global mode
  workspace_manager_->set_sharing_mode(WorkspaceSharingMode::Global);

  // Get another workspace
  auto workspace4_result =
      workspace_manager_->get_or_create_workspace(program_id);
  ASSERT_TRUE(workspace4_result.ok());
  auto workspace4 = workspace4_result.get();

  // Should be a different workspace since we switched modes
  EXPECT_NE(workspace3, workspace4);

  // Get a workspace with a different program ID in Global mode
  uintptr_t different_program_id = 67890;
  auto workspace5_result =
      workspace_manager_->get_or_create_workspace(different_program_id);
  ASSERT_TRUE(workspace5_result.ok());
  auto workspace5 = workspace5_result.get();

  // Should be the same workspace as workspace4
  EXPECT_EQ(workspace4, workspace5);
}
