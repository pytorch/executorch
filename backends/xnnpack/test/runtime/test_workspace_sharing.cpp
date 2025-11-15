/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/XNNPACKBackend.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/runtime.h>

#include <optional>

using namespace ::testing;

using executorch::backends::xnnpack::workspace_sharing_mode_option_key;
using executorch::backends::xnnpack::WorkspaceSharingMode;
using executorch::backends::xnnpack::xnnpack_backend_key;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;

TensorPtr create_input_tensor(float val);
void run_and_validate_two_models(
    std::optional<WorkspaceSharingMode> mode1 = std::nullopt,
    std::optional<WorkspaceSharingMode> mode2 = std::nullopt);
void set_and_check_workspace_sharing_mode(WorkspaceSharingMode mode);

TEST(WorkspaceSharing, SetMode) {
  // Try setting and reading back the mode a few times.
  set_and_check_workspace_sharing_mode(WorkspaceSharingMode::Disabled);
  set_and_check_workspace_sharing_mode(WorkspaceSharingMode::PerModel);
  set_and_check_workspace_sharing_mode(WorkspaceSharingMode::Global);
}

TEST(WorkspaceSharing, SetInvalidMode) {
  // Make sure we can't set an invalid mode.

  // Set to an initial known value.
  set_and_check_workspace_sharing_mode(WorkspaceSharingMode::PerModel);

  // Set to a bad value.
  BackendOptions<1> backend_options;
  backend_options.set_option(workspace_sharing_mode_option_key, 70);

  auto status = executorch::runtime::set_option(
      xnnpack_backend_key, backend_options.view());
  ASSERT_EQ(status, Error::InvalidArgument);

  // Make sure the option is still set to a valid value.
  BackendOption read_option;
  ASSERT_GT(sizeof(read_option.key), strlen(workspace_sharing_mode_option_key));
  strcpy(read_option.key, workspace_sharing_mode_option_key);
  read_option.value = -1;
  status = get_option(xnnpack_backend_key, read_option);

  ASSERT_TRUE(
      std::get<int>(read_option.value) ==
      static_cast<int>(WorkspaceSharingMode::PerModel));
}

TEST(WorkspaceSharing, RunWithDisabledMode) {
  // Load and run some PTEs with workspace sharing disabled.
  run_and_validate_two_models(WorkspaceSharingMode::Disabled);
}

TEST(WorkspaceSharing, RunWithPerModelMode) {
  // Load and run some PTEs with per-model workspace sharing.
  run_and_validate_two_models(WorkspaceSharingMode::PerModel);
}

TEST(WorkspaceSharing, RunWithGlobalMode) {
  // Load and run some PTEs with global workspace sharing.
  run_and_validate_two_models(WorkspaceSharingMode::Global);
}

TEST(WorkspaceSharing, RunWithModeSwitch) {
  // Check each pair of modes, loading one model in one mode and the other in
  // the other mode.

  std::array<WorkspaceSharingMode, 3> modes = {
      WorkspaceSharingMode::Disabled,
      WorkspaceSharingMode::PerModel,
      WorkspaceSharingMode::Global};

  for (auto i = 0; i < modes.size(); ++i) {
    for (auto j = i + 1; j < modes.size(); ++j) {
      run_and_validate_two_models(modes[i], modes[j]);
    }
  }
}

TensorPtr create_input_tensor(float val) {
  // Create an f32 tensor with shape [10, 10, 10], matching the input of the
  // test models.
  std::vector<float> data(1000, val);

  // Note that the tensor pointer takes ownership of the data vector.
  return executorch::extension::make_tensor_ptr({10, 10, 10}, std::move(data));
}

void run_and_validate_two_models(
    std::optional<WorkspaceSharingMode> mode1,
    std::optional<WorkspaceSharingMode> mode2) {
  // Load and run two models, verifying that the output tensors are correct,
  // optionally setting sharing mode.

  if (mode1) {
    set_and_check_workspace_sharing_mode(*mode1);
  }

  Module mod1(std::getenv("ET_XNNPACK_GENERATED_ADD_LARGE_PTE_PATH"));

  auto a = create_input_tensor(1.0);
  auto b = create_input_tensor(2.0);
  auto c = create_input_tensor(3.0);

  auto result = mod1.forward({a, b, c});
  EXPECT_TRUE(result.ok());

  // Expected output is 2a + 2b + c.
  auto output_val = 1.0 * 2 + 2.0 * 2 + 3.0;
  auto& output_tensor = result.get()[0].toTensor();
  for (auto i = 0; i < output_tensor.numel(); ++i) {
    ASSERT_EQ(output_tensor.const_data_ptr<float>()[i], output_val);
  }

  if (mode2) {
    set_and_check_workspace_sharing_mode(*mode2);
  }

  Module mod2(std::getenv("ET_XNNPACK_GENERATED_SUB_LARGE_PTE_PATH"));

  auto result2 = mod2.forward({a, b, c});
  EXPECT_TRUE(result2.ok());

  // Expected output is zero (the subtract operations cancel out).
  auto& output_tensor2 = result2.get()[0].toTensor();
  for (auto i = 0; i < output_tensor2.numel(); ++i) {
    ASSERT_EQ(output_tensor2.const_data_ptr<float>()[i], 0);
  }

  // Run mod1 again to validate that it gives correct results in the second mode
  auto result3 = mod1.forward({a, b, c});
  EXPECT_TRUE(result3.ok());

  // Expected output is still 2a + 2b + c
  auto& output_tensor3 = result3.get()[0].toTensor();
  for (auto i = 0; i < output_tensor3.numel(); ++i) {
    ASSERT_EQ(output_tensor3.const_data_ptr<float>()[i], output_val);
  }
}

void set_and_check_workspace_sharing_mode(WorkspaceSharingMode mode) {
  executorch::runtime::runtime_init();

  BackendOptions<1> backend_options;
  backend_options.set_option(
      workspace_sharing_mode_option_key, static_cast<int>(mode));

  auto status = executorch::runtime::set_option(
      xnnpack_backend_key, backend_options.view());
  ASSERT_EQ(status, Error::Ok);

  // Read the option back to sanity check.
  BackendOption read_option;
  ASSERT_GT(sizeof(read_option.key), strlen(workspace_sharing_mode_option_key));
  strcpy(read_option.key, workspace_sharing_mode_option_key);
  read_option.value = -1;
  status = get_option(xnnpack_backend_key, read_option);

  ASSERT_TRUE(std::get<int>(read_option.value) == static_cast<int>(mode));
}
