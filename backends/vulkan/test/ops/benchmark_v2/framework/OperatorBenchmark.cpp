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
#include <utility>
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
  Report execute_nodes = Report::array();
  for (size_t i = first; i < end; ++i) {
    execute_nodes.push_back(graph.execute_nodes()[i]->name());
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
    result["execute_nodes"] = std::move(execute_nodes);
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
  result["execute_nodes"] = std::move(execute_nodes);
  result.update(measure(spec, graph, encode_batch, Json(result)));
  return result;
}

Json validate_operator_dynamic_resize(const Json& spec, ComputeGraph& graph) {
  const Json& parameters = spec.at("parameters");
  const std::vector<Json> operator_cases =
      parse_configs(parameters.at("operator_case"));
  reference::require(
      operator_cases.size() == 1,
      "Dynamic validation operator_case must expand to one case");
  Json operator_spec = operator_cases.front();
  const std::string output_name = parameters.at("output_name");
  Json input_spec = operator_spec;
  input_spec["tensors"].erase(output_name);
  Json output_spec = operator_spec;
  output_spec["tensors"] = {
      {output_name, operator_spec.at("tensors").at(output_name)}};
  auto host_data = prepare_tensor_data(input_spec);
  const auto input_tensors = create_tensors(input_spec, graph, host_data, true);
  auto output_data = prepare_tensor_data(output_spec);
  const auto output_tensors =
      create_tensors(output_spec, graph, output_data, false);
  host_data.merge(output_data);
  auto tensors = input_tensors;
  tensors.insert(output_tensors.begin(), output_tensors.end());
  for (const auto& input : graph.inputs())
    reference::require(
        input.value != tensors.at(output_name),
        "Dynamic validation output must not be staged as an input");
  const auto args = build_operator_args(operator_spec, graph, tensors);
  build_operator_route(operator_spec, graph, args);
  Json accesses = Json::object();
  for (const auto& tensor : tensors)
    accesses[tensor.first] = tensor.first == output_name ? "write" : "read";
  const auto staging =
      add_validation_outputs(operator_spec, graph, tensors, accesses);
  graph.prepare();
  graph.prepack();
  const auto resized_bytes = [&](const std::string& name, const Json& sizes) {
    return tensor_numel(sizes) * host_data.at(name).size() /
        tensor_numel(operator_spec.at("tensors").at(name).at("sizes"));
  };
  Json validations = Json::array();
  for (const Json& step : spec.at("validation_steps")) {
    Json resized = operator_spec;
    auto data = host_data;
    const Json& sizes = step.at("tensor_sizes");
    reference::require(
        !sizes.contains(output_name),
        "Dynamic validation steps must not resize the output manually");
    for (const auto& item : sizes.items()) {
      const auto shape = item.value().get<std::vector<int64_t>>();
      const ValueRef ref = input_tensors.at(item.key());
      bool found = false;
      for (size_t i = 0; i < graph.inputs().size(); ++i)
        if (graph.inputs().at(i).value == ref) {
          graph.resize_input(i, shape);
          found = true;
          break;
        }
      reference::require(
          found, "Dynamic validation size target is not a staged input");
      resized["tensors"][item.key()]["sizes"] = shape;
      const size_t bytes = resized_bytes(item.key(), item.value());
      reference::require(
          bytes <= data.at(item.key()).size(),
          "Dynamic validation shape exceeds upper input allocation");
      data.at(item.key()).resize(bytes);
    }
    resized["tensors"][output_name]["sizes"] = step.at("expected_output_sizes");
    graph.propagate_resize();
    reference::require(
        graph.sizes_of(tensors.at(output_name)) ==
            step.at("expected_output_sizes").get<std::vector<int64_t>>(),
        "Operator output did not resize");
    const size_t output_bytes =
        resized_bytes(output_name, step.at("expected_output_sizes"));
    reference::require(
        output_bytes <= data.at(output_name).size(),
        "Dynamic validation output exceeds upper allocation");
    data.at(output_name).resize(output_bytes);
    copy_validation_inputs(resized, graph, input_tensors, data);
    graph.execute();
    if (step.value("validate_output", true)) {
      const Json validation =
          validate_outputs(resized, graph, tensors, staging, data);
      reference::require(
          validation.at("status") == "PASS",
          "Dynamic resize validation failed: " + validation.dump());
      validations.push_back(validation);
    }
  }
  return {
      {"status", "PASS"},
      {"validation_steps", spec.at("validation_steps").size()},
      {"validated_outputs", validations.size()}};
}

} // namespace vkcompute::shader_profiling
