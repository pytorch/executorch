// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Validation.h>

#include <chrono>
#include <set>
#include <stdexcept>

namespace vkcompute::shader_profiling {

void copy_validation_inputs(
    const Json& spec,
    ComputeGraph& graph,
    const std::map<std::string, ValueRef>& tensors,
    const std::map<std::string, std::vector<uint8_t>>& host_data) {
  for (const auto& tensor : tensors) {
    const Json& description = spec.at("tensors").at(tensor.first);
    if (description.value("storage", "buffer") == "constant") {
      continue;
    }
    auto dtype = scalar_type(description.value("dtype", "float32"));
    if (dtype == vkapi::kHalf) {
      dtype = vkapi::kFloat;
    } else if (dtype == vkapi::kInt8x4) {
      dtype = vkapi::kChar;
    }
    graph.maybe_cast_and_copy_into_staging(
        graph.staging_of(tensor.second),
        host_data.at(tensor.first).data(),
        tensor_numel(description.at("sizes")),
        dtype);
  }
}

std::map<std::string, ValueRef> add_validation_outputs(
    const Json& spec,
    ComputeGraph& graph,
    const std::map<std::string, ValueRef>& tensors,
    const Json& accesses) {
  std::set<std::string> outputs = validation_output_names(spec);
  const bool explicitly_selected = !outputs.empty();
  if (outputs.empty() && spec.contains("outputs")) {
    for (const Json& name : spec.at("outputs")) {
      outputs.insert(name.get<std::string>());
    }
  } else if (outputs.empty()) {
    for (const auto& access : accesses.items()) {
      if (access.value() != "read") {
        outputs.insert(access.key());
      }
    }
  }
  if (outputs.empty()) {
    throw std::invalid_argument(
        "Validation requires at least one output tensor");
  }
  std::map<std::string, ValueRef> staging;
  for (const auto& name : outputs) {
    if (explicitly_selected) {
      reference::require(
          accesses.contains(name) && accesses.at(name) != "read",
          "Selected validation tensor is not writable: " + name);
    }
    const ValueRef tensor = tensors.at(name);
    auto dtype = graph.dtype_of(tensor);
    if (dtype == vkapi::kHalf) {
      dtype = vkapi::kFloat;
    } else if (dtype == vkapi::kInt8x4) {
      dtype = vkapi::kChar;
    }
    staging.emplace(name, graph.set_output_tensor(tensor, dtype));
  }
  return staging;
}

namespace {

template <typename T>
std::vector<double> read_staging(
    ComputeGraph& graph,
    ValueRef staging,
    size_t count,
    vkapi::ScalarType dtype) {
  std::vector<T> data(count);
  graph.maybe_cast_and_copy_from_staging(staging, data.data(), count, dtype);
  return std::vector<double>(data.begin(), data.end());
}

} // namespace

Report validate_outputs(
    const Json& spec,
    ComputeGraph& graph,
    const std::map<std::string, ValueRef>& tensors,
    const std::map<std::string, ValueRef>& staging,
    const std::map<std::string, std::vector<uint8_t>>& host_data) {
  ReferenceOutputs actual;
  for (const auto& output : staging) {
    const ValueRef tensor = tensors.at(output.first);
    if (graph.sizes_of(tensor) !=
        spec.at("tensors")
            .at(output.first)
            .at("sizes")
            .get<std::vector<int64_t>>()) {
      throw std::invalid_argument(
          "Output tensor shape does not match declared shape: " + output.first);
    }
    const size_t count = tensor_numel(graph.sizes_of(tensor));
    const auto dtype = graph.dtype_of(tensor);
    if (dtype == vkapi::kChar || dtype == vkapi::kInt8x4) {
      actual[output.first] =
          read_staging<int8_t>(graph, output.second, count, vkapi::kChar);
    } else if (dtype == vkapi::kByte) {
      actual[output.first] =
          read_staging<uint8_t>(graph, output.second, count, vkapi::kByte);
    } else if (dtype == vkapi::kInt) {
      actual[output.first] =
          read_staging<int32_t>(graph, output.second, count, vkapi::kInt);
    } else {
      actual[output.first] =
          read_staging<float>(graph, output.second, count, vkapi::kFloat);
    }
  }
  const Json dtypes = actual_tensor_dtypes(graph, tensors);
  const auto start = std::chrono::steady_clock::now();
  Json reference_spec = spec;
  Json shaders = Json::array();
  for (const auto& node : graph.execute_nodes()) {
    shaders.push_back(node->name());
  }
  reference_spec["reference_context"] = {
      {"int8_dot_product", graph.can_use_int8_dot_product()},
      {"shaders", shaders}};
  const ReferenceOutputs expected =
      reference_outputs(reference_spec, dtypes, host_data);
  const double reference_us = std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
  Report result = compare_outputs(spec, expected, actual, dtypes);
  result["reference_wall_time_us"] = reference_us;
  return result;
}

} // namespace vkcompute::shader_profiling
