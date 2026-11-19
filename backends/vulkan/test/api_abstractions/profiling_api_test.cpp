// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/test/api_abstractions/VulkanTest.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <future>
#include <thread>
#include <vector>

namespace vkcompute {
namespace {

using ProfilingApiTest = VulkanTest;

TEST(ProfilingValueTest, TimestampVectorsSupportStandardAlgorithms) {
  const std::vector<vkapi::ShaderTimestamp> samples = {
      {"first", 1, 2, 3, {{8, 1, 1}, {4, 1, 1}}},
      {"second", 4, 5, 6, {{16, 1, 1}, {8, 1, 1}}}};
  auto reordered = samples;
  std::reverse(reordered.begin(), reordered.end());
  EXPECT_EQ(reordered.front().kernel_name, "second");
  reordered.erase(reordered.begin());
  reordered.insert(reordered.begin(), samples.back());
  reordered.back() = samples.front();
  reordered.back().metadata.gwg[0] = 32;
  EXPECT_EQ(reordered.back().metadata.gwg[0], 32);
  EXPECT_EQ(samples.front().metadata.gwg[0], 8);
}

TEST(ProfilingValueTest, OrderViolationsCountOutOfOrderTimestamps) {
  const vkapi::ShaderMetadata metadata{{1, 1, 1}, {1, 1, 1}};
  EXPECT_EQ(vkapi::count_order_violations({}), 0u);
  EXPECT_EQ(
      vkapi::count_order_violations(
          {{"a", 1, 2, 3, metadata}, {"b", 3, 4, 5, metadata}}),
      0u);
  // Barrier finish after shader execution finish.
  EXPECT_EQ(
      vkapi::count_order_violations(
          {{"a", 1, 4, 3, metadata}, {"b", 5, 6, 7, metadata}}),
      1u);
  // The second dispatch's window starts before the first one finishes.
  EXPECT_EQ(
      vkapi::count_order_violations(
          {{"a", 1, 2, 5, metadata}, {"b", 4, 6, 7, metadata}}),
      1u);
}

bool timestamps_available(ComputeGraph& graph) {
  uint32_t family_count = 0;
  const VkPhysicalDevice physical =
      graph.context()->adapter_ptr()->physical_handle();
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical, &family_count, families.data());
  return families.at(graph.context()->queue().family_index).timestampValidBits >
      0 &&
      graph.context()->adapter_ptr()->timestamp_period() > 0;
}

void disable_profiling(
    api::Context& context,
    std::promise<void>& started,
    std::promise<void>& completed) {
  started.set_value();
  context.set_shader_profiling_enabled(false);
  completed.set_value();
}

TEST_F(ProfilingApiTest, EmptyQueryPoolHasNoTimestampTicks) {
  ComputeGraph graph{GraphConfig{}};
  EXPECT_TRUE(
      graph.context()->querypool().get_shader_timestamp_ticks().empty());
}

TEST_F(ProfilingApiTest, ProfilingModeChangesWaitForCommandRecording) {
  ComputeGraph graph{GraphConfig{}};
  auto& context = *graph.context();
  std::promise<void> started;
  std::promise<void> completed;
  auto entered = started.get_future();
  auto finished = completed.get_future();
  auto recording = context.dispatch_lock();
  std::thread worker(
      disable_profiling,
      std::ref(context),
      std::ref(started),
      std::ref(completed));
  entered.wait();
  const auto status = finished.wait_for(std::chrono::milliseconds(100));
  recording.unlock();
  worker.join();
  EXPECT_EQ(status, std::future_status::timeout);
}

TEST_F(ProfilingApiTest, UnfinishedAndUnsubmittedCapturesHaveNoTimestampTicks) {
  ComputeGraph probe{GraphConfig{}};
  if (!timestamps_available(probe)) {
    GTEST_SKIP() << "GPU timestamps are unavailable";
  }
  for (const bool complete : {false, true}) {
    SCOPED_TRACE(complete);
    GraphConfig config;
    config.enable_querypool = true;
    ComputeGraph graph(config);
    auto& context = *graph.context();
    context.initialize_querypool();
    {
      auto resetting = context.dispatch_lock();
      context.set_cmd();
      context.cmd_reset_querypool();
      vkapi::VulkanFence fence(context.device());
      context.submit_cmd_to_gpu(fence.get_submit_handle());
      fence.wait();
    }
    context.flush();
    auto recording = context.dispatch_lock();
    context.set_cmd();
    context.report_previous_work_finish(
        "unsubmitted",
        GlobalWorkGrid({1, 1, 1}, kTextureExtentsWorkGrid),
        LocalWorkGroup(1, 1, 1));
    if (complete) {
      context.report_shader_exec_finish();
    }
    EXPECT_TRUE(context.querypool().get_shader_timestamp_ticks().empty());
  }
}

TEST_F(ProfilingApiTest, TimestampTogglePreservesExecutionAndArgumentMetadata) {
  executorch::runtime::runtime_init();
  ComputeGraph probe{GraphConfig{}};
  const bool has_timestamps = timestamps_available(probe);
  for (const bool enabled : {true, false, true}) {
    SCOPED_TRACE(enabled);
    GraphConfig config;
    config.enable_querypool = has_timestamps;
    config.execute_threshold_node_count = 64;
    ComputeGraph graph(config);
    graph.context()->set_shader_profiling_enabled(enabled);
    const auto a =
        graph.add_input_tensor({1, 1, 1, 16}, vkapi::kFloat, utils::kBuffer);
    const auto b =
        graph.add_input_tensor({1, 1, 1, 16}, vkapi::kFloat, utils::kBuffer);
    const auto out =
        graph.add_tensor({1, 1, 1, 16}, vkapi::kFloat, utils::kBuffer);
    VK_GET_OP_FN("aten.add.Tensor")
    (graph, {a.value, b.value, graph.add_scalar(1.0), out});
    const auto& args = graph.execute_nodes().back()->args();
    ASSERT_EQ(args.size(), 2);
    EXPECT_EQ(args[0].refs, (std::vector<ValueRef>{out}));
    EXPECT_EQ(args[0].access, vkapi::kWrite);
    EXPECT_EQ(args[1].refs, (std::vector<ValueRef>{a.value, b.value}));
    EXPECT_EQ(args[1].access, vkapi::kRead);
    const auto ostage = graph.set_output_tensor(out);
    graph.prepare();
    graph.prepack();
    graph.propagate_resize();
    std::vector<float> input(16, 0.25f);
    graph.maybe_cast_and_copy_into_staging(
        a.staging, input.data(), input.size(), vkapi::kFloat);
    graph.maybe_cast_and_copy_into_staging(
        b.staging, input.data(), input.size(), vkapi::kFloat);
    graph.execute();
    std::vector<float> actual(16);
    graph.maybe_cast_and_copy_from_staging(
        ostage, actual.data(), actual.size(), vkapi::kFloat);
    EXPECT_EQ(actual, std::vector<float>(16, 0.5f));
    const auto ticks =
        graph.context()->querypool().get_shader_timestamp_ticks();
    if (enabled && has_timestamps) {
      ASSERT_EQ(ticks.size(), graph.execute_nodes().size());
      const auto target =
          std::find_if(ticks.begin(), ticks.end(), [](const auto& entry) {
            return entry.kernel_name.find("binary_op_") == 0;
          });
      ASSERT_NE(target, ticks.end());
      EXPECT_GE(
          target->barrier_finish_ticks, target->previous_work_finish_ticks);
      EXPECT_GE(target->shader_exec_finish_ticks, target->barrier_finish_ticks);
      EXPECT_GT(target->metadata.gwg[0], 0);
      EXPECT_GT(target->metadata.lwg[0], 0);
    } else {
      EXPECT_TRUE(ticks.empty());
    }
  }
}

TEST_F(ProfilingApiTest, ShaderProfilingWidensBarriersToFullBarriers) {
  executorch::runtime::runtime_init();
  ComputeGraph probe{GraphConfig{}};
  if (!timestamps_available(probe)) {
    GTEST_SKIP() << "GPU timestamps are unavailable";
  }
  GraphConfig config;
  config.enable_querypool = true;
  ComputeGraph graph(config);
  graph.prepare();
  api::Context& context = *graph.context();
  vkapi::PipelineBarrier narrow{};
  narrow.stage = {
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};

  context.set_shader_profiling_enabled(false);
  vkapi::PipelineBarrier unprofiled = narrow;
  context.serialize_if_profiling(unprofiled);
  EXPECT_EQ(unprofiled.stage.src, narrow.stage.src);
  EXPECT_EQ(unprofiled.stage.dst, narrow.stage.dst);
  EXPECT_EQ(unprofiled.global_access.src, 0u);
  EXPECT_EQ(unprofiled.global_access.dst, 0u);

  context.set_shader_profiling_enabled(true);
  vkapi::PipelineBarrier profiled = narrow;
  context.serialize_if_profiling(profiled);
  const vkapi::PipelineBarrier full = vkapi::full_pipeline_barrier();
  EXPECT_EQ(profiled.stage.src, narrow.stage.src | full.stage.src);
  EXPECT_EQ(profiled.stage.dst, narrow.stage.dst | full.stage.dst);
  EXPECT_EQ(profiled.global_access.src, full.global_access.src);
  EXPECT_EQ(profiled.global_access.dst, full.global_access.dst);
}

TEST_F(ProfilingApiTest, ProfiledTimestampsAreRewrittenOnReplay) {
  executorch::runtime::runtime_init();
  ComputeGraph probe{GraphConfig{}};
  if (!timestamps_available(probe)) {
    GTEST_SKIP() << "GPU timestamps are unavailable";
  }
  GraphConfig config;
  config.enable_querypool = true;
  ComputeGraph graph(config);
  graph.context()->set_shader_profiling_enabled(true);
  const std::vector<int64_t> sizes = {1, 64, 64, 64};
  const auto a = graph.add_input_tensor(sizes, vkapi::kFloat, utils::kBuffer);
  const auto b = graph.add_input_tensor(sizes, vkapi::kFloat, utils::kBuffer);
  const auto out_a = graph.add_tensor(sizes, vkapi::kFloat, utils::kBuffer);
  const auto out_b = graph.add_tensor(sizes, vkapi::kFloat, utils::kBuffer);
  VK_GET_OP_FN("aten.add.Tensor")
  (graph, {a.value, a.value, graph.add_scalar(1.0), out_a});
  VK_GET_OP_FN("aten.add.Tensor")
  (graph, {b.value, b.value, graph.add_scalar(1.0), out_b});
  graph.set_output_tensor(out_a);
  graph.set_output_tensor(out_b);
  graph.prepare();
  graph.prepack();
  graph.propagate_resize();
  graph.execute();
  const auto ticks = graph.context()->querypool().get_shader_timestamp_ticks();
  ASSERT_GE(ticks.size(), 2);
  EXPECT_EQ(vkapi::count_order_violations(ticks), 0u);
  graph.context()->querypool().extract_results();
  EXPECT_EQ(graph.context()->querypool().num_order_violations(), 0u);
  // The second execute replays the recorded command buffers, which must reset
  // and rewrite every query.
  graph.execute();
  const auto replayed =
      graph.context()->querypool().get_shader_timestamp_ticks();
  ASSERT_EQ(replayed.size(), ticks.size());
  EXPECT_EQ(vkapi::count_order_violations(replayed), 0u);
  EXPECT_GT(
      replayed.front().previous_work_finish_ticks,
      ticks.back().shader_exec_finish_ticks);
}

} // namespace
} // namespace vkcompute
