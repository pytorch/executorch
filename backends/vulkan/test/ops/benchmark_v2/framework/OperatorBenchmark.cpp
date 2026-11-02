// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/OperatorBenchmark.h>

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/TensorUtils.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Timing.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Validation.h>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace vkcompute::shader_profiling {

namespace {

Json tensor_accesses(
    ComputeGraph& graph,
    size_t first,
    size_t end,
    const std::map<std::string, ValueRef>& tensors,
    ValueRef primary_output) {
  std::map<ValueRef, vkapi::MemoryAccessFlags> first_access;
  std::map<ValueRef, vkapi::MemoryAccessFlags> all_access;
  for (size_t i = first; i < end; ++i) {
    std::map<ValueRef, vkapi::MemoryAccessFlags> node_access;
    for (const auto& group : graph.execute_nodes()[i]->args()) {
      for (ValueRef ref : group.refs) {
        node_access[ref] |= group.access;
      }
    }
    for (const auto& entry : node_access) {
      first_access.emplace(entry);
      all_access[entry.first] |= entry.second;
    }
  }
  Json result = Json::object();
  for (const auto& tensor : tensors) {
    const ValueRef ref = tensor.second;
    // Mutable input caches can be written before their unchanged prefix is
    // read.
    const bool reads_existing = (first_access[ref] & vkapi::kRead) ||
        (ref != primary_output && (all_access[ref] & vkapi::kRead));
    result[tensor.first] = !(all_access[ref] & vkapi::kWrite) ? "read"
        : reads_existing                                      ? "read_write"
                                                              : "write";
  }
  return result;
}

ValueRef operator_arg(
    const Json& arg,
    ComputeGraph& graph,
    const std::map<std::string, ValueRef>& tensors) {
  if (arg.is_null()) {
    return graph.add_none();
  }
  if (arg.contains("tensor")) {
    return tensors.at(arg.at("tensor").get<std::string>());
  }
  if (arg.contains("int")) {
    return graph.add_scalar(arg.at("int").get<int64_t>());
  }
  if (arg.contains("float")) {
    return graph.add_scalar(arg.at("float").get<double>());
  }
  if (arg.contains("bool")) {
    return graph.add_scalar(arg.at("bool").get<bool>());
  }
  if (arg.contains("int_list")) {
    return graph.add_scalar_list(
        arg.at("int_list").get<std::vector<int64_t>>());
  }
  return graph.add_string(arg.at("string").get<std::string>());
}

} // namespace

std::vector<ValueRef> build_operator_args(
    const Json& spec,
    ComputeGraph& graph,
    const std::map<std::string, ValueRef>& tensors) {
  std::vector<ValueRef> args;
  for (const Json& arg : spec.at("args"))
    args.push_back(operator_arg(arg, graph, tensors));
  return args;
}

Report benchmark_operator(
    const Json& spec,
    const TensorData& host_data,
    bool validation_only) {
  GraphConfig config;
  config.force_resize = true;
  if (!validation_only) {
    config.descriptor_pool_safety_factor *= spec.value("max_dispatches", 200u);
  }
  ComputeGraph graph(config);
  const auto tensors = create_tensors(spec, graph, host_data, validation_only);
  const auto args = build_operator_args(spec, graph, tensors);
  const size_t first = graph.execute_nodes().size();
  const Json route = build_operator_route(spec, graph, args);
  const size_t end = graph.execute_nodes().size();
  if (first == end) {
    throw std::invalid_argument("Operator produced no execute nodes");
  }
  const Json accesses =
      tensor_accesses(graph, first, end, tensors, args.back());
  std::map<std::string, ValueRef> outputs;
  if (validation_only) {
    outputs = add_validation_outputs(spec, graph, tensors, accesses);
  }
  graph.prepare();
  graph.prepack();
  graph.propagate_resize();
  if (validation_only) {
    copy_validation_inputs(spec, graph, tensors, host_data);
    graph.execute();
    Report result =
        operator_report(spec, actual_tensor_dtypes(graph, tensors), accesses);
    if (spec.contains("route"))
      result["route"] = route;
    result["validation"] =
        validate_outputs(spec, graph, tensors, outputs, host_data);
    return result;
  }
  auto encode_batch = [&](uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
      for (size_t j = first; j < end; ++j) {
        graph.execute_nodes()[j]->encode(&graph);
      }
    }
  };
  // Warm once to establish the layout state used at each replay boundary.
  auto& context = *graph.context();
  context.set_cmd();
  encode_batch(1);
  vkapi::VulkanFence fence(context.device());
  context.submit_cmd_to_gpu(fence.get_submit_handle(), true);
  fence.wait();
  Report result =
      operator_report(spec, actual_tensor_dtypes(graph, tensors), accesses);
  if (spec.contains("route")) {
    result["route"] = route;
  }
  result["execute_nodes_per_invocation"] = end - first;
  result["execute_nodes"] = Report::array();
  for (size_t i = first; i < end; ++i) {
    result["execute_nodes"].push_back(graph.execute_nodes()[i]->name());
  }
  result.update(measure(spec, graph, encode_batch, Json(result)));
  return result;
}

} // namespace vkcompute::shader_profiling
