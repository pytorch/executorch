/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUExecutionOptions.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {
namespace {

TEST(WebGPUExecutionOptionsTest, DefaultsToPreservingOutputs) {
  EXPECT_EQ(
      current_webgpu_execution_options().discardable_output_data, nullptr);
}

TEST(WebGPUExecutionOptionsTest, NestedScopesRestorePriorValue) {
  int outer_output = 0;
  int inner_output = 0;
  {
    ScopedWebGPUExecutionOptions outer({&outer_output, true});
    EXPECT_EQ(
        current_webgpu_execution_options().discardable_output_data,
        &outer_output);
    {
      ScopedWebGPUExecutionOptions inner({&inner_output, true});
      EXPECT_EQ(
          current_webgpu_execution_options().discardable_output_data,
          &inner_output);
    }
    EXPECT_EQ(
        current_webgpu_execution_options().discardable_output_data,
        &outer_output);
  }
  EXPECT_EQ(
      current_webgpu_execution_options().discardable_output_data, nullptr);
}

TEST(WebGPUExecutionOptionsTest, ExceptionRestoresPriorValue) {
  int output = 0;
  EXPECT_THROW(
      with_webgpu_execution_options(
          {&output, true},
          []() -> void { throw std::runtime_error("expected"); }),
      std::runtime_error);
  EXPECT_EQ(
      current_webgpu_execution_options().discardable_output_data, nullptr);
}

TEST(WebGPUExecutionOptionsTest, BooleanReturnRestoresPriorValue) {
  int output = 0;
  const bool result =
      with_webgpu_execution_options({&output, true}, []() { return false; });
  EXPECT_FALSE(result);
  EXPECT_EQ(
      current_webgpu_execution_options().discardable_output_data, nullptr);
}

TEST(WebGPUExecutionOptionsTest, ResolvesOnlyOneExactDelegateOutput) {
  int method_output = 0;
  int delegate_intermediate = 0;
  const std::vector<const void*> delegate_outputs = {&delegate_intermediate};

  EXPECT_EQ(
      resolve_webgpu_graph_execution_options(
          delegate_outputs, WebGPUExecutionOptions{&method_output, true})
          .suppress_output_ordinal,
      kNoOutputOrdinal);
  EXPECT_EQ(
      resolve_webgpu_graph_execution_options(
          {&delegate_intermediate, &method_output},
          WebGPUExecutionOptions{&method_output, true})
          .suppress_output_ordinal,
      1);
  EXPECT_EQ(
      resolve_webgpu_graph_execution_options(
          {&method_output, &method_output},
          WebGPUExecutionOptions{&method_output, true})
          .suppress_output_ordinal,
      kNoOutputOrdinal);
  EXPECT_EQ(
      resolve_webgpu_graph_execution_options(
          {&method_output}, WebGPUExecutionOptions{&method_output, false})
          .suppress_output_ordinal,
      kNoOutputOrdinal);
}

TEST(WebGPUExecutionPlanTest, DefaultPlanPreservesDispatchesAndOutputs) {
  const std::vector<SuppressibleOutput> suppressible = {{9, 1, 4, 6}};
  const WebGPUExecutionPlan plan = plan_webgpu_execution(
      6, 2, ExecuteConfig{}, suppressible, WebGPUGraphExecutionOptions{});

  EXPECT_EQ(
      plan.dispatch_chunks,
      (std::vector<std::vector<size_t>>{{0, 1, 2, 3, 4, 5}}));
  EXPECT_EQ(plan.copy_outputs, (std::vector<bool>{true, true}));
}

TEST(WebGPUExecutionPlanTest, SuppressionIsPerOutputAndSupportsChunking) {
  const std::vector<SuppressibleOutput> suppressible = {{9, 1, 4, 6}};
  const ExecuteConfig config = {2, 1};
  const WebGPUExecutionPlan plan = plan_webgpu_execution(
      6, 2, config, suppressible, WebGPUGraphExecutionOptions{1});

  EXPECT_EQ(
      plan.dispatch_chunks,
      (std::vector<std::vector<size_t>>{{0}, {1, 2}, {3}}));
  EXPECT_EQ(plan.copy_outputs, (std::vector<bool>{true, false}));
}

TEST(WebGPUExecutionPlanTest, RejectsInvalidSuppressibleRange) {
  const std::vector<SuppressibleOutput> suppressible = {{9, 0, 3, 7}};
  EXPECT_THROW(
      plan_webgpu_execution(
          6, 1, ExecuteConfig{}, suppressible, WebGPUGraphExecutionOptions{0}),
      std::runtime_error);
}

TEST(WebGPUExecutionPlanTest, AllSuppressedHasNoSyntheticDispatchChunk) {
  const std::vector<SuppressibleOutput> suppressible = {{9, 0, 0, 2}};
  const WebGPUExecutionPlan plan = plan_webgpu_execution(
      2, 1, ExecuteConfig{}, suppressible, WebGPUGraphExecutionOptions{0});

  EXPECT_TRUE(plan.dispatch_chunks.empty());
  EXPECT_EQ(plan.copy_outputs, (std::vector<bool>{false}));
}

TEST(WebGPUExecutionPlanTest, CopyOnlyPlanRetainsOneSubmissionChunk) {
  const WebGPUExecutionPlan plan = plan_webgpu_execution(
      0, 1, ExecuteConfig{}, {}, WebGPUGraphExecutionOptions{});

  EXPECT_EQ(plan.dispatch_chunks, (std::vector<std::vector<size_t>>{{}}));
  EXPECT_EQ(plan.copy_outputs, (std::vector<bool>{true}));
}

TEST(WebGPUExecutionPlanTest, FiltersDisabledDispatchesAcrossChunks) {
  const std::vector<bool> enabled = {true, false, true, false, true, true};
  const WebGPUExecutionPlan plan = plan_webgpu_execution(
      6, 1, ExecuteConfig{2, 1}, {}, WebGPUGraphExecutionOptions{}, enabled);

  EXPECT_EQ(
      plan.dispatch_chunks,
      (std::vector<std::vector<size_t>>{{0}, {2}, {4}, {5}}));
  EXPECT_EQ(plan.copy_outputs, (std::vector<bool>{true}));
}

TEST(WebGPUExecutionPlanTest, RejectsMismatchedEnabledDispatches) {
  EXPECT_THROW(
      plan_webgpu_execution(
          3,
          1,
          ExecuteConfig{},
          {},
          WebGPUGraphExecutionOptions{},
          {true, false}),
      std::runtime_error);
}

} // namespace
} // namespace executorch::backends::webgpu
