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
  const auto options = current_webgpu_execution_options();
  EXPECT_EQ(options.discardable_output_data, nullptr);
  EXPECT_FALSE(options.single_compute_pass);
  EXPECT_EQ(options.max_compute_dispatches_per_pass, 0u);
}

TEST(WebGPUExecutionOptionsTest, SinglePassOptionsSurviveResolution) {
  WebGPUExecutionOptions options;
  options.single_compute_pass = true;
  options.max_compute_dispatches_per_pass = 7;

  const auto no_suppression =
      resolve_webgpu_graph_execution_options({}, options);
  EXPECT_TRUE(no_suppression.single_compute_pass);
  EXPECT_EQ(no_suppression.max_compute_dispatches_per_pass, 7u);

  int output = 0;
  options.discardable_output_data = &output;
  options.exact_method_certificate_verified = false;
  const auto uncertified =
      resolve_webgpu_graph_execution_options({&output}, options);
  EXPECT_TRUE(uncertified.single_compute_pass);
  EXPECT_EQ(uncertified.max_compute_dispatches_per_pass, 7u);
  EXPECT_EQ(uncertified.suppress_output_ordinal, kNoOutputOrdinal);
}

TEST(WebGPUExecutionOptionsTest, SinglePassScopeRestoresEveryField) {
  WebGPUExecutionOptions options;
  options.single_compute_pass = true;
  options.max_compute_dispatches_per_pass = 3;
  {
    ScopedWebGPUExecutionOptions scope(options);
    const auto current = current_webgpu_execution_options();
    EXPECT_TRUE(current.single_compute_pass);
    EXPECT_EQ(current.max_compute_dispatches_per_pass, 3u);
  }
  const auto restored = current_webgpu_execution_options();
  EXPECT_FALSE(restored.single_compute_pass);
  EXPECT_EQ(restored.max_compute_dispatches_per_pass, 0u);
}

TEST(WebGPUExecutionOptionsTest, PassCapZeroIsUnlimited) {
  EXPECT_FALSE(webgpu_pass_cap_reached(0, 0));
  EXPECT_FALSE(webgpu_pass_cap_reached(1000, 0));
}

TEST(WebGPUExecutionOptionsTest, PassCapClosesAtExactDispatchCount) {
  EXPECT_FALSE(webgpu_pass_cap_reached(2, 3));
  EXPECT_TRUE(webgpu_pass_cap_reached(3, 3));
  EXPECT_TRUE(webgpu_pass_cap_reached(4, 3));
}

TEST(WebGPUExecutionOptionsTest, CommandInventoryTracksComputeRunsAndCopies) {
  std::vector<WebGPUCommandRecord> commands(5);
  commands[0].identity = "first";
  commands[1].identity = "disabled";
  commands[1].enabled = false;
  commands[1].zero_grid = true;
  commands[2].kind = WebGPUCommandKind::GraphCopy;
  commands[3].identity = "second";
  commands[4].kind = WebGPUCommandKind::OutputCopy;

  const auto inventory = build_webgpu_command_inventory(commands);
  EXPECT_EQ(inventory.static_dispatch_records, 4u);
  EXPECT_EQ(inventory.active_compute_count, 2u);
  EXPECT_EQ(inventory.zero_grid_compute_count, 1u);
  EXPECT_EQ(inventory.graph_copy_count, 1u);
  EXPECT_EQ(inventory.output_copy_count, 1u);
  EXPECT_EQ(inventory.maximal_compute_runs, 2u);
  EXPECT_NE(inventory.canonical_commands_json.find("first"), std::string::npos);
  EXPECT_NE(inventory.canonical_commands_json.find("second"), std::string::npos);
}

TEST(WebGPUExecutionOptionsTest, ComputePassCountHonorsCopiesAndCap) {
  std::vector<WebGPUCommandRecord> commands(7);
  commands[0].identity = "a";
  commands[1].identity = "b";
  commands[2].identity = "disabled";
  commands[2].enabled = false;
  commands[3].kind = WebGPUCommandKind::GraphCopy;
  commands[4].identity = "c";
  commands[5].identity = "d";
  commands[6].identity = "e";

  EXPECT_EQ(count_webgpu_compute_passes(commands, false, 0), 5u);
  EXPECT_EQ(count_webgpu_compute_passes(commands, true, 0), 2u);
  EXPECT_EQ(count_webgpu_compute_passes(commands, true, 2), 3u);
  EXPECT_THROW(
      count_webgpu_compute_passes(commands, false, 2), std::invalid_argument);
}

TEST(WebGPUExecutionOptionsTest, DisabledCopyDoesNotSplitComputeRun) {
  std::vector<WebGPUCommandRecord> commands(3);
  commands[0].identity = "first";
  commands[1].kind = WebGPUCommandKind::GraphCopy;
  commands[1].enabled = false;
  commands[2].identity = "second";
  EXPECT_EQ(count_webgpu_compute_passes(commands, true, 0), 1u);
}

TEST(WebGPUExecutionOptionsTest, AttestationSerializesObservedPassCounts) {
  WebGPUExecutionAttestation attestation;
  attestation.execution_ordinal = 3;
  attestation.requested = true;
  attestation.applied = true;
  attestation.completed = true;
  attestation.encoded_compute_passes = 2;
  attestation.queue_submit_count = 1;
  attestation.max_compute_dispatches_per_pass = 4;

  const std::string json = serialize_webgpu_execution_attestation(attestation);
  EXPECT_NE(json.find("\"executionOrdinal\":3"), std::string::npos);
  EXPECT_NE(json.find("\"encodedComputePasses\":2"), std::string::npos);
  EXPECT_NE(
      json.find("\"maxComputeDispatchesPerPass\":4"), std::string::npos);
  EXPECT_NE(json.find("\"completed\":true"), std::string::npos);
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
