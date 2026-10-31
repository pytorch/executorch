// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/WorkgroupUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

#include <stdexcept>
#include <vector>

namespace vkcompute::shader_profiling {
namespace {

GlobalWorkGrid pick_gwg(
    const Json& setting,
    ComputeGraph& graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args) {
  if (setting.is_array()) {
    const auto dims = setting.get<std::vector<uint32_t>>();
    return GlobalWorkGrid({dims[0], dims[1], dims[2]}, kExplicitWorkGrid);
  }
  static const std::map<std::string, decltype(&default_pick_gwg)> pickers = {
      {"default_pick_gwg", default_pick_gwg}};
  const std::string name = setting.get<std::string>();
  const auto picker = pickers.find(name);
  if (picker == pickers.end()) {
    throw std::invalid_argument("Unknown gwg_size picker: " + name);
  }
  if (args.empty() || !(args.front().access & vkapi::WRITE)) {
    throw std::invalid_argument(
        "gwg_size picker requires a write or read_write tensor binding");
  }
  return picker->second(&graph, shader, args, {});
}

LocalWorkGroup pick_lwg(
    const Json& setting,
    ComputeGraph& graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args) {
  if (setting.is_array()) {
    const auto dims = setting.get<std::vector<uint32_t>>();
    return LocalWorkGroup(dims[0], dims[1], dims[2]);
  }
  static const std::map<std::string, decltype(&default_pick_lwg)> pickers = {
      {"default_pick_lwg", default_pick_lwg},
      {"pick_required_lwg", pick_required_lwg},
      {"pick_xy_square_lwg", pick_xy_square_lwg},
      {"pick_xz_square_lwg", pick_xz_square_lwg},
      {"pick_120shape_lwg", pick_120shape_lwg}};
  const std::string name = setting.get<std::string>();
  const auto picker = pickers.find(name);
  if (picker == pickers.end()) {
    throw std::invalid_argument("Unknown lwg_size picker: " + name);
  }
  return picker->second(&graph, shader, gwg, args, {});
}

} // namespace

WorkgroupSizes pick_workgroup_sizes(
    const Json& spec,
    ComputeGraph& graph,
    const vkapi::ShaderInfo& shader,
    const std::map<std::string, ValueRef>& tensors) {
  std::vector<ArgGroup> args;
  std::vector<ArgGroup> read_args;
  for (const Json& binding : spec.at("bindings")) {
    if (!binding.contains("tensor")) {
      continue;
    }
    const ValueRef ref = tensors.at(binding.at("tensor").get<std::string>());
    const std::string access = binding.value("access", "read");
    if (access == "read") {
      read_args.emplace_back(ref, vkapi::kRead);
    } else {
      args.emplace_back(
          ref, access == "write" ? vkapi::kWrite : vkapi::kReadWrite);
    }
  }
  for (const ArgGroup& arg : read_args) {
    args.emplace_back(arg.refs, arg.access);
  }
  GlobalWorkGrid gwg = pick_gwg(spec.at("gwg_size"), graph, shader, args);
  const LocalWorkGroup lwg =
      pick_lwg(spec.at("lwg_size"), graph, shader, gwg, args);
  auto* adapter = graph.context()->adapter_ptr();
  lwg.validate(
      adapter->max_compute_workgroup_size(),
      adapter->max_compute_workgroup_invocations());
  if (gwg.is_linear() && gwg.required_lwg_size() != lwg) {
    // Rewrap the linear workload using the selected local geometry.
    gwg = GlobalWorkGrid(
        {utils::safe_downcast<uint32_t>(
             graph.numel_of(args.front().refs.front())),
         1,
         1},
        kLinearWorkGrid);
    gwg.wrap_linear_dispatch(adapter->max_compute_workgroup_count(), lwg);
  }
  gwg.validate(lwg, adapter->max_compute_workgroup_count());
  return {gwg, lwg};
}

} // namespace vkcompute::shader_profiling
