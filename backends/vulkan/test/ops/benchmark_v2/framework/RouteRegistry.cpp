// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

namespace vkcompute::shader_profiling {

RouteBuildFunction RouteRegistry::operator_route(
    const std::string& op,
    const std::string& route) const {
  const auto it = operators_.find(op);
  if (it == operators_.end() || !it->second.count(route)) {
    throw std::invalid_argument("Unknown route for " + op + ": " + route);
  }
  return it->second.at(route);
}

RouteInspectFunction RouteRegistry::operator_inspector(
    const std::string& op) const {
  const auto it = inspectors_.find(op);
  return it == inspectors_.end() ? nullptr : it->second;
}

void RouteRegistry::register_operator_route(
    const std::string& op,
    const std::string& route,
    RouteBuildFunction function) {
  if (op.empty() || route.empty() || route == "auto" || !function) {
    throw std::invalid_argument(
        "Route requires operator/route names and a function; auto is reserved");
  }
  if (!operators_[op].emplace(route, function).second) {
    throw std::invalid_argument("Duplicate route for " + op + ": " + route);
  }
}

void RouteRegistry::register_operator_inspector(
    const std::string& op,
    RouteInspectFunction function) {
  if (op.empty() || !function) {
    throw std::invalid_argument("Route inspector requires a name and function");
  }
  if (!inspectors_.emplace(op, function).second) {
    throw std::invalid_argument("Duplicate route inspector: " + op);
  }
}

RouteRegistry& route_registry() {
  static RouteRegistry registry;
  return registry;
}

Json build_operator_route(
    const Json& spec,
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  const std::string op = spec.at("operator");
  const Json requested =
      spec.value("route", Json{{"name", "auto"}, {"options", Json::object()}});
  const std::string route = requested.at("name");
  const Json& options = requested.at("options");
  auto& registry = route_registry();
  Json resolved;
  if (route == "auto") {
    if (!options.empty()) {
      throw std::invalid_argument("auto route does not accept options");
    }
    const size_t first = graph.execute_nodes().size();
    VK_GET_OP_FN(op)(graph, args);
    if (const auto inspector = registry.operator_inspector(op)) {
      resolved = inspector(graph, args, first, graph.execute_nodes().size());
    }
  } else {
    const auto builder = registry.operator_route(op, route);
    resolved = builder(graph, args, options);
  }
  if (route != "auto" && resolved.is_null()) {
    throw std::invalid_argument("Route builder returned no resolved metadata");
  }
  if (!resolved.is_null() &&
      (!resolved.is_object() || !resolved.contains("name") ||
       !resolved.at("name").is_string() ||
       resolved.at("name").get<std::string>().empty() ||
       !resolved.contains("options") || !resolved.at("options").is_object())) {
    throw std::invalid_argument("Route returned invalid resolved metadata");
  }
  return {{"requested", requested}, {"resolved", resolved}};
}

} // namespace vkcompute::shader_profiling
