/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/gpu_shared/runtime/SharedGpuCompileSpec.h>
#include <executorch/backends/vulkan/runtime/SharedGpuContextAdapter.h>

using executorch::backends::gpu_shared::SharedContextMode;
using executorch::backends::gpu_shared::SharedGpuCompileSpec;
using executorch::backends::vulkan::shared::
    determine_shared_adapter_resolution_action;
using executorch::backends::vulkan::shared::SharedAdapterResolutionAction;

namespace {

TEST(SharedGpuContextAdapterTest, DisabledSpecDisablesSharedResolution) {
  SharedGpuCompileSpec spec;
  spec.context_mode = SharedContextMode::kDisabled;
  EXPECT_EQ(
      determine_shared_adapter_resolution_action(spec, false),
      SharedAdapterResolutionAction::kDisabled);
}

TEST(SharedGpuContextAdapterTest, LookupOnlyWithoutContextFails) {
  SharedGpuCompileSpec spec;
  spec.context_mode = SharedContextMode::kLookupOnly;
  EXPECT_EQ(
      determine_shared_adapter_resolution_action(spec, false),
      SharedAdapterResolutionAction::kErrorMissing);
}

TEST(SharedGpuContextAdapterTest, LookupOnlyWithContextUsesExisting) {
  SharedGpuCompileSpec spec;
  spec.context_mode = SharedContextMode::kLookupOnly;
  EXPECT_EQ(
      determine_shared_adapter_resolution_action(spec, true),
      SharedAdapterResolutionAction::kUseExisting);
}

TEST(SharedGpuContextAdapterTest, LookupOrCreateWithoutContextCreatesNew) {
  SharedGpuCompileSpec spec;
  spec.context_mode = SharedContextMode::kLookupOrCreate;
  EXPECT_EQ(
      determine_shared_adapter_resolution_action(spec, false),
      SharedAdapterResolutionAction::kCreateNew);
}

TEST(SharedGpuContextAdapterTest, LookupOrCreateWithContextUsesExisting) {
  SharedGpuCompileSpec spec;
  spec.context_mode = SharedContextMode::kLookupOrCreate;
  EXPECT_EQ(
      determine_shared_adapter_resolution_action(spec, true),
      SharedAdapterResolutionAction::kUseExisting);
}

TEST(SharedGpuContextAdapterTest, CreateOnlyWithoutContextCreatesNew) {
  SharedGpuCompileSpec spec;
  spec.context_mode = SharedContextMode::kCreateOnly;
  EXPECT_EQ(
      determine_shared_adapter_resolution_action(spec, false),
      SharedAdapterResolutionAction::kCreateNew);
}

TEST(SharedGpuContextAdapterTest, CreateOnlyWithExistingContextFails) {
  SharedGpuCompileSpec spec;
  spec.context_mode = SharedContextMode::kCreateOnly;
  EXPECT_EQ(
      determine_shared_adapter_resolution_action(spec, true),
      SharedAdapterResolutionAction::kErrorAlreadyLoaded);
}

} // namespace
