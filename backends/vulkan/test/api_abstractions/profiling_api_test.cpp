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
      {"first", 1, 2, {{8, 1, 1}, {4, 1, 1}}},
      {"second", 3, 4, {{16, 1, 1}, {8, 1, 1}}}};
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

bool timestamps_available(ComputeGraph& graph) {
  uint32_t family_count = 0;
  const VkPhysicalDevice physical =
      graph.context()->adapter_ptr()->physical_handle();
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical, &family_count, families.data());
  return families.at(graph.context()->queue().family_index).timestampValidBits >
      0 && graph.context()->adapter_ptr()->timestamp_period() > 0;
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
    context.report_shader_dispatch_start(
        "unsubmitted", GlobalWorkGrid({1, 1, 1}, kTextureExtentsWorkGrid),
        LocalWorkGroup(1, 1, 1));
    if (complete) {
      context.report_shader_dispatch_end();
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
            return entry.kernel_name.find("binary_add_") == 0;
          });
      ASSERT_NE(target, ticks.end());
      EXPECT_GE(target->end_ticks, target->start_ticks);
      EXPECT_GT(target->metadata.gwg[0], 0);
      EXPECT_GT(target->metadata.lwg[0], 0);
    } else {
      EXPECT_TRUE(ticks.empty());
    }
  }
}

} // namespace
} // namespace vkcompute
