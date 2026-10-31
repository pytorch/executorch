// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Report.h>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace vkcompute::shader_profiling {
namespace {

const Json kBoth = {
    {"show_memory_throughput", true},
    {"show_compute_throughput", true}};

TEST(
    BenchmarkThroughputReport,
    CalculatesRatesPerSampleWithCorrectBinaryAndFlopUnits) {
  const WorkAmounts work{1073741824.0, 2000000000.0, {}};
  const auto report =
      throughput_report(kBoth, work, 2, {2000000, 4000000}, {4000000, 8000000});
  const auto& gpu = report.at("gpu_timestamp");
  const auto& memory = gpu.at("memory_gib_per_s");
  EXPECT_DOUBLE_EQ(memory.at("min"), 0.5);
  EXPECT_DOUBLE_EQ(memory.at("max"), 1.0);
  EXPECT_DOUBLE_EQ(memory.at("mean"), 0.75);
  EXPECT_NEAR(memory.at("p90").get<double>(), 0.95, 1e-12);
  EXPECT_DOUBLE_EQ(gpu.at("memory_mib_per_s").at("mean"), 768);
  EXPECT_DOUBLE_EQ(gpu.at("compute_gflops").at("mean"), 1.5);
  EXPECT_EQ(report.at("gpu_timestamp_batch"), gpu);
  EXPECT_EQ(
      report.at("wall_time").at("memory_gib_per_s"),
      Report::array({0.5, 0.25}));
  EXPECT_EQ(
      report.at("wall_time").at("compute_gflops"), Report::array({1.0, 0.5}));
  EXPECT_EQ(report.at("wall_time_batch"), report.at("wall_time"));
}

TEST(BenchmarkThroughputReport, FlagsControlOnlyRequestedMetrics) {
  const WorkAmounts work{1024.0, 4.0, {}};
  EXPECT_TRUE(throughput_report({}, work, 1, {1}, {2}).empty());
  const auto memory =
      throughput_report({{"show_memory_throughput", true}}, work, 1, {1}, {2});
  EXPECT_TRUE(memory.contains("memory_bytes"));
  EXPECT_FALSE(memory.contains("compute_flops"));
  EXPECT_FALSE(memory.at("gpu_timestamp").contains("compute_gflops"));
  const auto compute =
      throughput_report({{"show_compute_throughput", true}}, work, 1, {1}, {2});
  EXPECT_FALSE(compute.contains("memory_bytes"));
  EXPECT_TRUE(compute.contains("compute_flops"));
  EXPECT_FALSE(compute.at("gpu_timestamp").contains("memory_gib_per_s"));
}

TEST(BenchmarkThroughputReport, UnknownCountsAreNullAndDoNotEraseValidMetrics) {
  const WorkAmounts work{std::nullopt, 4.0, {"No memory counter"}};
  const auto report = throughput_report(kBoth, work, 1, {2}, {4});
  EXPECT_TRUE(report.at("memory_bytes").is_null());
  EXPECT_TRUE(report.at("gpu_timestamp").at("memory_gib_per_s").is_null());
  EXPECT_DOUBLE_EQ(
      report.at("gpu_timestamp").at("compute_gflops").at("mean"), 0.002);
  EXPECT_EQ(report.at("diagnostics"), Report::array({"No memory counter"}));
}

TEST(
    BenchmarkThroughputReport,
    ZeroWorkAndUnavailableTimestampsRemainDistinct) {
  const WorkAmounts work{0.0, 0.0, {}};
  const auto report = throughput_report(kBoth, work, 1, {}, {1, 2});
  EXPECT_TRUE(report.at("gpu_timestamp").at("compute_gflops").is_null());
  EXPECT_EQ(report.at("wall_time").at("compute_gflops"), Report::array({0, 0}));
  EXPECT_DOUBLE_EQ(
      throughput_report(kBoth, work, 1, {1}, {})
          .at("gpu_timestamp")
          .at("compute_gflops")
          .at("mean"),
      0);
}

TEST(BenchmarkThroughputReport, InvalidIntervalsCannotGenerateInfiniteRates) {
  const WorkAmounts work{4.0, 4.0, {}};
  const auto report = throughput_report(kBoth, work, 1, {0, 1}, {0, 2});
  EXPECT_TRUE(report.at("gpu_timestamp").at("compute_gflops").is_null());
  EXPECT_TRUE(report.at("wall_time").at("compute_gflops")[0].is_null());
  EXPECT_DOUBLE_EQ(report.at("wall_time").at("compute_gflops")[1], 0.002);
  EXPECT_THROW(
      throughput_report(kBoth, work, 0, {1}, {2}), std::invalid_argument);
}

TEST(
    BenchmarkThroughputReport,
    ShaderBreakdownUsesWorkAndSamplesForEachDispatch) {
  const Report dispatches =
      Report::array({{{"shader_name", "first"}}, {{"shader_name", "second"}}});
  const std::vector<WorkAmounts> work = {
      {1048576.0, 1000000000.0, {}}, {2097152.0, 3000000000.0, {}}};
  const auto result = shader_breakdown_report(
      dispatches, {{1000000, 2000000, 1000000, 2000000}}, 2, kBoth, work);
  EXPECT_DOUBLE_EQ(
      result[0]
          .at("throughput")
          .at("gpu_timestamp")
          .at("memory_mib_per_s")
          .at("mean"),
      1);
  EXPECT_DOUBLE_EQ(
      result[1]
          .at("throughput")
          .at("gpu_timestamp")
          .at("memory_mib_per_s")
          .at("mean"),
      1);
  EXPECT_DOUBLE_EQ(
      result[0]
          .at("throughput")
          .at("gpu_timestamp")
          .at("compute_gflops")
          .at("mean"),
      1);
  EXPECT_DOUBLE_EQ(
      result[1]
          .at("throughput")
          .at("gpu_timestamp")
          .at("compute_gflops")
          .at("mean"),
      1.5);
  EXPECT_FALSE(result[0].at("throughput").contains("wall_time"));
}

} // namespace
} // namespace vkcompute::shader_profiling
