// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

#include <cmath>
#include <map>
#include <stdexcept>

namespace vkcompute::shader_profiling {
namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

void associate(
    std::unordered_map<std::string, WorkCountFunction>& table,
    const std::string& name,
    WorkCountFunction function,
    const std::string& kind) {
  require(
      !name.empty() && function,
      "Work count registration requires a nonempty name and callable");
  require(
      table.emplace(name, function).second,
      "Duplicate " + kind + " work count association: " + name);
}

WorkCountFunction lookup(
    const std::unordered_map<std::string, WorkCountFunction>& table,
    const std::string& name,
    const std::string& kind) {
  const auto entry = table.find(name);
  require(
      entry != table.end(),
      "No " + kind + " work count registered for " + name);
  return entry->second;
}

WorkCountFunction lookup_shader(
    const std::unordered_map<std::string, WorkCountFunction>& exact,
    const std::unordered_map<std::string, WorkCountFunction>& prefixes,
    const std::string& name,
    const std::string& kind) {
  const auto entry = exact.find(name);
  if (entry != exact.end()) {
    return entry->second;
  }
  WorkCountFunction function = nullptr;
  size_t length = 0;
  for (const auto& prefix : prefixes) {
    if (prefix.first.size() > length &&
        name.compare(0, prefix.first.size(), prefix.first) == 0) {
      function = prefix.second;
      length = prefix.first.size();
    }
  }
  require(
      function != nullptr, "No " + kind + " work count registered for " + name);
  return function;
}

bool tensor_type(const Json& tensor) {
  const std::string type = tensor.value("type", "");
  return type == "buffer" || type == "texture2d" || type == "texture3d" ||
      type == "constant";
}

} // namespace

WorkCountFunction ThroughputRegistry::operator_memory(
    const std::string& name) const {
  const auto function = lookup(operator_memory_, name, "operator memory");
  return function;
}
WorkCountFunction ThroughputRegistry::operator_compute(
    const std::string& name) const {
  const auto function = lookup(operator_compute_, name, "operator compute");
  return function;
}
WorkCountFunction ThroughputRegistry::shader_memory(
    const std::string& name) const {
  const auto function = lookup_shader(
      shader_memory_, shader_memory_prefixes_, name, "shader memory");
  return function;
}
WorkCountFunction ThroughputRegistry::shader_compute(
    const std::string& name) const {
  const auto function = lookup_shader(
      shader_compute_, shader_compute_prefixes_, name, "shader compute");
  return function;
}

void ThroughputRegistry::register_operator_memory(
    const std::string& name,
    WorkCountFunction function) {
  associate(operator_memory_, name, function, "operator memory");
}
void ThroughputRegistry::register_operator_compute(
    const std::string& name,
    WorkCountFunction function) {
  associate(operator_compute_, name, function, "operator compute");
}
void ThroughputRegistry::register_shader_memory(
    const std::string& name,
    WorkCountFunction function) {
  associate(shader_memory_, name, function, "shader memory");
}
void ThroughputRegistry::register_shader_compute(
    const std::string& name,
    WorkCountFunction function) {
  associate(shader_compute_, name, function, "shader compute");
}

void ThroughputRegistry::register_shader_memory_prefix(
    const std::string& prefix,
    WorkCountFunction function) {
  associate(shader_memory_prefixes_, prefix, function, "shader memory prefix");
}

void ThroughputRegistry::register_shader_compute_prefix(
    const std::string& prefix,
    WorkCountFunction function) {
  associate(
      shader_compute_prefixes_, prefix, function, "shader compute prefix");
}

ThroughputRegistry& throughput_registry() {
  static ThroughputRegistry registry;
  return registry;
}

WorkAmounts estimate_work(const Json& spec, bool memory, bool compute) {
  WorkAmounts amounts;
  const auto calculate = [&](bool memory_count) -> std::optional<double> {
    try {
      require(
          spec.contains("operator_name") != spec.contains("shader_name"),
          "Work counts require exactly one operator_name or shader_name");
      const bool op = spec.contains("operator_name");
      const std::string name = spec.at(op ? "operator_name" : "shader_name");
      const auto& registry = throughput_registry();
      const auto function = op
          ? (memory_count ? registry.operator_memory(name)
                          : registry.operator_compute(name))
          : (memory_count ? registry.shader_memory(name)
                          : registry.shader_compute(name));
      const double count = function(spec);
      require(
          std::isfinite(count) && count >= 0,
          "Work count must be finite and nonnegative for " + name);
      return count;
    } catch (const std::exception& error) {
      amounts.diagnostics.push_back(
          std::string(
              memory_count ? "Memory throughput: " : "Compute throughput: ") +
          error.what());
      return std::nullopt;
    }
  };
  if (memory) {
    amounts.memory_bytes = calculate(true);
  }
  if (compute) {
    amounts.compute_flops = calculate(false);
  }
  return amounts;
}

namespace throughput {

double tensor_elements(const Json& tensor) {
  return double(tensor_numel(tensor.at("sizes")));
}

double tensor_bytes(const Json& tensor) {
  static const std::map<std::string, size_t> bytes_per_element = {
      {"float32", 4},
      {"float16", 2},
      {"int32", 4},
      {"int8", 1},
      {"uint8", 1},
      {"int8x4", 1}};
  const std::string dtype = tensor.at("dtype");
  const auto element_size = bytes_per_element.find(dtype);
  require(
      element_size != bytes_per_element.end(),
      "Unknown tensor dtype for memory work count: " + dtype);
  return tensor_elements(tensor) * element_size->second;
}

double tensor_traffic(const Json& spec) {
  struct Traffic {
    double bytes;
    unsigned access;
  };
  std::map<std::string, Traffic> tensors;
  const auto add = [&](const Json& tensor, const std::string& fallback_name) {
    require(
        tensor_type(tensor), "Unknown tensor storage for memory work count");
    const std::string access = tensor.at("access");
    require(
        access == "read" || access == "write" || access == "read_write",
        "Unknown tensor access for memory work count: " + access);
    const unsigned flags = access == "read" ? 1 : access == "write" ? 2 : 3;
    const std::string name =
        tensor.value("ref", tensor.value("name", fallback_name));
    const double bytes = tensor_bytes(tensor);
    auto entry = tensors.emplace(name, Traffic{bytes, flags});
    require(
        entry.first->second.bytes == bytes,
        "Aliased tensor byte counts disagree: " + name);
    entry.first->second.access |= flags;
  };
  if (spec.contains("operator_name")) {
    const Json& args = spec.at("args");
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i].contains("tensor")) {
        add(args[i].at("tensor"), "arg_" + std::to_string(i));
      }
    }
  } else {
    for (const auto& binding : spec.at("bindings").items()) {
      if (tensor_type(binding.value()) || binding.value().contains("sizes")) {
        add(binding.value(), binding.key());
      }
    }
  }
  require(!tensors.empty(), "Memory work count requires tensor bindings");
  double total = 0;
  for (const auto& tensor : tensors) {
    const auto& traffic = tensor.second;
    total +=
        traffic.bytes * ((traffic.access & 1) + ((traffic.access >> 1) & 1));
  }
  return total;
}

const Json& tensor_argument(const Json& spec, size_t index) {
  const Json& tensor = spec.at("args").at(index).at("tensor");
  require(
      tensor.is_object() && tensor.contains("sizes"),
      "Work count requires resolved tensor argument metadata");
  return tensor;
}

const Json& output_tensor(const Json& spec) {
  if (spec.contains("operator_name")) {
    return tensor_argument(spec, spec.at("args").size() - 1);
  }
  const Json* result = nullptr;
  for (const auto& binding : spec.at("bindings").items()) {
    const Json& tensor = binding.value();
    if (!tensor_type(tensor)) {
      continue;
    }
    const std::string access = tensor.at("access");
    if (access == "write" || access == "read_write") {
      if (!result ||
          tensor.value("binding_index", SIZE_MAX) <
              result->value("binding_index", SIZE_MAX)) {
        result = &tensor;
      }
    }
  }
  require(
      result != nullptr,
      "Compute work count requires an output tensor binding");
  return *result;
}

} // namespace throughput

} // namespace vkcompute::shader_profiling
