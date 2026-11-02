// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Timing.h>

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/GpuClock.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/TensorUtils.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Timestamps.h>

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>
#include <algorithm>
#include <chrono>
#include <utility>
#include <vector>

namespace vkcompute::shader_profiling {

namespace {

using Clock = std::chrono::steady_clock;

Report read_gpu_clock(GpuClock* clock) {
  const auto frequency = clock ? clock->read_mhz() : std::nullopt;
  return frequency ? Report(*frequency) : Report();
}

Report profile_operator_shaders(
    const Json& spec,
    ComputeGraph& graph,
    const Timestamps& timestamps,
    const std::function<void(uint32_t)>& encode_batch,
    uint32_t count,
    uint32_t query_limit,
    uint32_t warmup,
    uint32_t runs,
    GpuClock* gpu_clock) {
  auto& context = *graph.context();
  context.flush();
  context.initialize_querypool();
  context.set_shader_profiling_enabled(true);
  context.set_cmd(true);
  context.cmd_reset_querypool();
  const bool memory = spec.value("show_memory_throughput", false);
  const bool compute = spec.value("show_compute_throughput", false);
  std::vector<WorkAmounts> work;
  if (memory || compute) {
    for (const auto& node : graph.execute_nodes()) {
      const size_t before =
          context.querypool().get_shader_timestamp_data().size();
      node->encode(&graph);
      const auto records = context.querypool().get_shader_timestamp_data();
      const size_t emitted =
          records.size() > before ? records.size() - before : 0;
      for (size_t i = before; i < records.size(); ++i) {
        if (emitted == 1) {
          work.push_back(estimate_work(
              {{"shader_name", records[i].kernel_name},
               {"gwg_size",
                {{"method", "operator"}, {"value", records[i].metadata.gwg}}},
               {"lwg_size",
                {{"method", "operator"}, {"value", records[i].metadata.lwg}}},
               {"bindings", dispatch_tensor_bindings(graph, *node)}},
              memory,
              compute));
        } else {
          work.push_back(
              {std::nullopt,
               std::nullopt,
               {"Binding metadata unavailable for a node with multiple dispatches"}});
        }
      }
    }
  } else {
    encode_batch(1);
  }
  const size_t shaders_per_invocation =
      context.querypool().get_shader_timestamp_data().size();
  if ((memory || compute) && work.size() != shaders_per_invocation) {
    work.assign(
        shaders_per_invocation,
        {std::nullopt, std::nullopt, {"Shader binding metadata unavailable"}});
  }
  if (shaders_per_invocation > query_limit / 2) {
    throw UnsupportedRoute(
        "Operator dispatch chain exceeds timestamp query capacity");
  }
  count = shaders_per_invocation == 0
      ? 1u
      : std::min(
            count,
            static_cast<uint32_t>(query_limit / (2 * shaders_per_invocation)));
  if (count > 1) {
    context.flush();
    context.set_cmd(true);
    context.cmd_reset_querypool();
    encode_batch(count);
  }
  vkapi::VulkanFence fence(context.device());
  auto replay = [&]() {
    context.submit_cmd_to_gpu(fence.get_submit_handle());
    fence.wait();
  };
  for (uint32_t i = 0; i < warmup; ++i) {
    replay();
  }
  Report dispatches = Report::array();
  Report clock_samples = Report::array();
  std::vector<std::vector<double>> samples;
  for (uint32_t i = 0; i < runs; ++i) {
    const Report clock_start = read_gpu_clock(gpu_clock);
    replay();
    if (gpu_clock) {
      clock_samples.push_back({clock_start, read_gpu_clock(gpu_clock)});
    }
    const auto shaders = context.querypool().get_shader_timestamp_ticks();
    std::vector<double> durations;
    for (size_t j = 0; j < shaders.size(); ++j) {
      const auto& shader = shaders[j];
      if (i == 0 && j < shaders_per_invocation) {
        dispatches.push_back(
            {{"dispatch_index", j},
             {"shader_name", shader.kernel_name},
             {"gwg_size",
              {{"method", "operator"}, {"value", shader.metadata.gwg}}},
             {"lwg_size",
              {{"method", "operator"}, {"value", shader.metadata.lwg}}}});
      }
      durations.push_back(
          timestamps.elapsed_us(shader.start_ticks, shader.end_ticks));
    }
    samples.push_back(std::move(durations));
  }
  Report result = {
      {"shader_profile_invocations_per_batch", count},
      {"shader_breakdown",
       shader_breakdown_report(dispatches, samples, count, spec, work)}};
  if (gpu_clock) {
    result["shader_profile_batch_observed_gpu_clock_freq_MHz"] =
        std::move(clock_samples);
  }
  return result;
}

} // namespace

Report measure(
    const Json& spec,
    ComputeGraph& graph,
    const std::function<void(uint32_t)>& encode_batch,
    const Json& work_spec) {
  api::Context& context = *graph.context();
  GpuClock* gpu_clock = nullptr;
#if defined(__ANDROID__)
  GpuClock android_gpu_clock;
  gpu_clock = &android_gpu_clock;
#endif
  Report wall_clock_samples = Report::array();
  Report gpu_clock_samples = Report::array();
  Timestamps timestamps(context);
  vkapi::VulkanFence fence(context.device());
  const WorkAmounts work = estimate_work(
      work_spec,
      spec.value("show_memory_throughput", false),
      spec.value("show_compute_throughput", false));
  auto encode = [&](uint32_t count, bool timed) {
    context.set_shader_profiling_enabled(false);
    context.flush();
    context.set_cmd(true);
    if (timed) {
      timestamps.begin(context.extract_cmd());
    }
    encode_batch(count);
    if (timed) {
      timestamps.end(context.extract_cmd());
    }
  };
  auto replay = [&]() {
    const auto start = Clock::now();
    context.submit_cmd_to_gpu(fence.get_submit_handle());
    fence.wait();
    return std::chrono::duration<double, std::micro>(Clock::now() - start)
        .count();
  };
  auto measured_replay = [&](Report& clock_samples) {
    const Report clock_start = read_gpu_clock(gpu_clock);
    const double wall = replay();
    if (gpu_clock) {
      clock_samples.push_back({clock_start, read_gpu_clock(gpu_clock)});
    }
    return wall;
  };

  uint32_t count = 1;
  const uint32_t limit =
      spec.value("max_dispatches", spec.contains("operator") ? 200u : 65536u);
  const double target_us = spec.value("target_ms", 50.0) * 1000;
  for (;;) {
    encode(count, false);
    replay();
    const double wall = replay();
    if (wall >= target_us * 0.8 || count == limit)
      break;
    const uint32_t next = choose_dispatch_count(wall / count, target_us, limit);
    count = std::min(limit, std::max(count + 1, std::min(next, count * 8)));
  }
  const uint32_t warmup = spec.value("warmup_runs", 3u);
  const uint32_t runs = spec.value("benchmark_runs", 10u);
  encode(count, false);
  std::vector<double> wall_samples;
  std::vector<double> gpu_samples;
  for (uint32_t i = 0; i < warmup; ++i) {
    replay();
  }
  for (uint32_t i = 0; i < runs; ++i) {
    wall_samples.push_back(measured_replay(wall_clock_samples));
  }
  if (timestamps.available()) {
    encode(count, true);
    for (uint32_t i = 0; i < warmup; ++i) {
      replay();
    }
    for (uint32_t i = 0; i < runs; ++i) {
      measured_replay(gpu_clock_samples);
      gpu_samples.push_back(timestamps.elapsed_us());
    }
  }
  Report result = {
      {"device", graph.device_name()},
      {"target_ms", target_us / 1000},
      {"max_dispatches", limit},
      {spec.contains("operator") ? "operator_invocations_per_batch"
                                 : "dispatches_per_batch",
       count},
      {"warmup_runs", warmup},
      {"benchmark_runs", runs},
      {"gpu_timestamps_available", timestamps.available()},
      {"timestamp_period_ns", timestamps.period_ns()},
      {"timestamp_valid_bits", timestamps.valid_bits()}};
  const Report measurements = measurement_report(
      count, spec.contains("operator"), gpu_samples, wall_samples);
  for (const auto& measurement : measurements.items()) {
    result[measurement.key()] = measurement.value();
    if (gpu_clock && measurement.key() == "gpu_timestamp_batch_us") {
      result["gpu_batch_observed_gpu_clock_freq_MHz"] =
          std::move(gpu_clock_samples);
    }
    if (gpu_clock && measurement.key() == "wall_time_batch_us") {
      result["wall_time_batch_observed_gpu_clock_freq_MHz"] =
          std::move(wall_clock_samples);
    }
  }
  Report throughput =
      throughput_report(spec, work, count, gpu_samples, wall_samples);
  if (!throughput.empty()) {
    result["throughput"] = std::move(throughput);
  }
  if (spec.contains("operator") && timestamps.available()) {
    result.update(profile_operator_shaders(
        spec,
        graph,
        timestamps,
        encode_batch,
        count,
        graph.graphconfig().context_config.query_pool_config.max_query_count,
        warmup,
        runs,
        gpu_clock));
  }
  return result;
}

} // namespace vkcompute::shader_profiling
