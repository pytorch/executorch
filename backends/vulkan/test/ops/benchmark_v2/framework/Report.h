// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vkcompute::shader_profiling {

using Report = nlohmann::ordered_json;

struct TableOptions {
  size_t name_max_chars = 32;
  size_t inputs_max_chars = 64;
};

std::string format_report(const Report& report);
std::string format_condensed_report(const Report& report);
std::string format_table(
    const Report& reports,
    const TableOptions& options = {});
Report latency_statistics(std::vector<double> samples);
Report measurement_report(
    uint32_t repetitions,
    bool operator_case,
    const std::vector<double>& gpu_batches,
    const std::vector<double>& wall_batches);
Report throughput_report(
    const Json& spec,
    const WorkAmounts& work,
    uint32_t repetitions,
    const std::vector<double>& gpu_batches,
    const std::vector<double>& wall_batches);
Report shader_breakdown_report(
    const Report& dispatches,
    const std::vector<std::vector<double>>& samples,
    uint32_t repetitions = 1,
    const Json& throughput_flags = Json::object(),
    const std::vector<WorkAmounts>& work = {});
Report shader_report(
    const Json& spec,
    const Json& gwg,
    const Json& lwg,
    const Json& actual_dtypes,
    const Report& constants);
Report operator_report(
    const Json& spec,
    const Json& actual_dtypes,
    const Json& actual_accesses = Json::object());
Report error_report(const Json& spec, const std::string& error);

} // namespace vkcompute::shader_profiling
