// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vkcompute::shader_profiling {

class UnsupportedRoute : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

using RouteBuildFunction =
    Json (*)(ComputeGraph&, const std::vector<ValueRef>&, const Json&);
using RouteInspectFunction =
    Json (*)(ComputeGraph&, const std::vector<ValueRef>&, size_t, size_t);

class RouteRegistry final {
 private:
  std::unordered_map<
      std::string,
      std::unordered_map<std::string, RouteBuildFunction>>
      operators_;
  std::unordered_map<std::string, RouteInspectFunction> inspectors_;

 public:
  RouteBuildFunction operator_route(
      const std::string& op,
      const std::string& route) const;
  RouteInspectFunction operator_inspector(const std::string& op) const;
  void register_operator_route(
      const std::string& op,
      const std::string& route,
      RouteBuildFunction function);
  void register_operator_inspector(
      const std::string& op,
      RouteInspectFunction function);
};

class RouteRegisterInit final {
 public:
  explicit RouteRegisterInit(void (*initialize)()) {
    initialize();
  }
};

RouteRegistry& route_registry();
Json build_operator_route(
    const Json& spec,
    ComputeGraph& graph,
    const std::vector<ValueRef>& args);

} // namespace vkcompute::shader_profiling

#define REGISTER_OPERATOR_ROUTE(op, route, function)                       \
  ::vkcompute::shader_profiling::route_registry().register_operator_route( \
      #op, #route, &function)
#define REGISTER_ROUTE_INSPECTOR(op, function)                                 \
  ::vkcompute::shader_profiling::route_registry().register_operator_inspector( \
      #op, &function)
#define REGISTER_OPERATOR_ROUTES                                           \
  static void register_routes();                                           \
  static const ::vkcompute::shader_profiling::RouteRegisterInit route_reg( \
      &register_routes);                                                   \
  static void register_routes()
