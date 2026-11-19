// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>

namespace vkcompute::shader_profiling {

namespace {

void register_association(
    std::unordered_map<std::string, std::string>& table,
    const std::string& name,
    const std::string& reference_name,
    const std::string& kind) {
  reference::require(
      !name.empty() && !reference_name.empty(),
      kind + " and reference names must be nonempty");
  reference::require(
      table.emplace(name, reference_name).second,
      "Duplicate " + kind + " reference association: " + name);
}

const std::string& lookup_association(
    const std::unordered_map<std::string, std::string>& table,
    const std::string& name,
    const std::string& kind) {
  const auto entry = table.find(name);
  reference::require(
      entry != table.end(),
      "No correctness reference registered for " + kind + " " + name);
  return entry->second;
}

} // namespace

ReferenceFunction ReferenceRegistry::get_reference(
    const std::string& name) const {
  const auto entry = functions_.find(name);
  reference::require(
      entry != functions_.end(), "Unknown reference function: " + name);
  return entry->second;
}

ReferenceOutputsFunction ReferenceRegistry::get_reference_outputs(
    const std::string& name) const {
  const auto entry = output_functions_.find(name);
  return entry == output_functions_.end() ? nullptr : entry->second;
}

bool ReferenceRegistry::has_reference(const std::string& name) const {
  return functions_.count(name) || output_functions_.count(name);
}

const std::string& ReferenceRegistry::operator_reference(
    const std::string& name) const {
  const auto& reference = lookup_association(operators_, name, "operator");
  reference::require(
      has_reference(reference), "Unknown reference function: " + reference);
  return reference;
}

const std::string& ReferenceRegistry::shader_reference(
    const std::string& name) const {
  const auto& reference = lookup_association(shaders_, name, "shader");
  reference::require(
      has_reference(reference), "Unknown reference function: " + reference);
  return reference;
}

void ReferenceRegistry::register_reference(
    const std::string& name,
    ReferenceFunction function) {
  reference::require(!name.empty(), "Reference name must be nonempty");
  reference::require(
      function != nullptr, "Reference function must not be null");
  reference::require(
      !has_reference(name), "Duplicate reference function: " + name);
  functions_.emplace(name, function);
}

void ReferenceRegistry::register_reference_outputs(
    const std::string& name,
    ReferenceOutputsFunction function) {
  reference::require(!name.empty(), "Reference name must be nonempty");
  reference::require(
      function != nullptr, "Reference function must not be null");
  reference::require(
      !has_reference(name), "Duplicate reference function: " + name);
  output_functions_.emplace(name, function);
}

void ReferenceRegistry::register_operator_reference(
    const std::string& name,
    const std::string& reference) {
  register_association(operators_, name, reference, "operator");
}

void ReferenceRegistry::register_shader_reference(
    const std::string& name,
    const std::string& reference) {
  register_association(shaders_, name, reference, "shader");
}

ReferenceRegistry& reference_registry() {
  static ReferenceRegistry registry;
  return registry;
}

} // namespace vkcompute::shader_profiling
