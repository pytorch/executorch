// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>

#include <gtest/gtest.h>

namespace vkcompute::shader_profiling {
namespace {

Json first_builder(
    ComputeGraph&,
    const std::vector<ValueRef>&,
    const Json& options) {
  return {{"name", "first"}, {"options", options}};
}
Json second_builder(
    ComputeGraph&,
    const std::vector<ValueRef>&,
    const Json& options) {
  return {{"name", "second"}, {"options", options}};
}
Json inspect_builder(
    ComputeGraph&,
    const std::vector<ValueRef>&,
    size_t,
    size_t) {
  return {{"name", "first"}, {"options", Json::object()}};
}

REGISTER_OPERATOR_ROUTES {
  REGISTER_OPERATOR_ROUTE(test_operator, candidate, first_builder);
  REGISTER_ROUTE_INSPECTOR(test_operator, inspect_builder);
}

TEST(BenchmarkRouteRegistry, RoutesAreScopedToOperators) {
  RouteRegistry registry;
  registry.register_operator_route("op_a", "direct", first_builder);
  registry.register_operator_route("op_b", "direct", second_builder);
  registry.register_operator_route("op_a", "other", second_builder);
  EXPECT_EQ(registry.operator_route("op_a", "direct"), first_builder);
  EXPECT_EQ(registry.operator_route("op_b", "direct"), second_builder);
  EXPECT_EQ(registry.operator_route("op_a", "other"), second_builder);
}

TEST(BenchmarkRouteRegistry, BuildersCanBeSharedAcrossRoutes) {
  RouteRegistry registry;
  registry.register_operator_route("op_a", "direct", first_builder);
  registry.register_operator_route("op_a", "other", first_builder);
  registry.register_operator_route("op_b", "direct", first_builder);
  EXPECT_EQ(registry.operator_route("op_a", "direct"), first_builder);
  EXPECT_EQ(registry.operator_route("op_a", "other"), first_builder);
  EXPECT_EQ(registry.operator_route("op_b", "direct"), first_builder);
}

TEST(BenchmarkRouteRegistry, DuplicateRoutesPreserveOriginals) {
  RouteRegistry registry;
  registry.register_operator_route("op", "direct", first_builder);
  EXPECT_THROW(
      registry.register_operator_route("op", "direct", second_builder),
      std::invalid_argument);
  EXPECT_EQ(registry.operator_route("op", "direct"), first_builder);
}

TEST(BenchmarkRouteRegistry, InspectorsAreOptionalAndUnique) {
  RouteRegistry registry;
  EXPECT_EQ(registry.operator_inspector("op"), nullptr);
  registry.register_operator_inspector("op", inspect_builder);
  EXPECT_EQ(registry.operator_inspector("op"), inspect_builder);
  EXPECT_THROW(
      registry.register_operator_inspector("op", inspect_builder),
      std::invalid_argument);
}

TEST(BenchmarkRouteRegistry, RejectsMissingAndInvalidRegistrations) {
  RouteRegistry registry;
  EXPECT_THROW(registry.operator_route("op", "missing"), std::invalid_argument);
  registry.register_operator_route("op", "direct", first_builder);
  EXPECT_THROW(registry.operator_route("op", "missing"), std::invalid_argument);
  EXPECT_THROW(
      registry.register_operator_route("", "direct", first_builder),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_operator_route("op", "", first_builder),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_operator_route("op", "direct", nullptr),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_operator_route("op", "auto", first_builder),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_operator_inspector("", inspect_builder),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_operator_inspector("op", nullptr),
      std::invalid_argument);
}

TEST(BenchmarkRouteRegistry, StaticRegistrationsAreAvailable) {
  const auto& registry = route_registry();
  EXPECT_EQ(
      registry.operator_route("test_operator", "candidate"), first_builder);
  EXPECT_EQ(registry.operator_inspector("test_operator"), inspect_builder);
}

} // namespace
} // namespace vkcompute::shader_profiling
