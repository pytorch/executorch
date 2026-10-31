// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/api_abstractions/VulkanTest.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/CaseCheckRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>
#include <gtest/gtest.h>
namespace vkcompute::shader_profiling {
using GpuCaseCheckRegistry = VulkanTest;

namespace {
Json pass_check(const Json&, ComputeGraph&) {
  return {{"status", "PASS"}};
}
Json fail_check(const Json&, ComputeGraph&) {
  return {{"status", "FAIL"}, {"error", "intentional"}};
}
Json unsupported_shader_check(const Json&, ComputeGraph&) {
  throw vkapi::ShaderNotSupportedError(
      "unsupported_test_shader", vkapi::VulkanExtension::INTEGER_DOT_PRODUCT);
}
Json unsupported_route_check(const Json&, ComputeGraph&) {
  throw UnsupportedRoute("Unsupported test route");
}
Json error_check(const Json&, ComputeGraph&) {
  throw std::runtime_error("Ordinary test error");
}
REGISTER_CASE_CHECKS {
  REGISTER_CASE_CHECK(test_case_check_pass, pass_check);
  REGISTER_CASE_CHECK(test_case_check_fail, fail_check);
  REGISTER_CASE_CHECK(
      test_case_check_unsupported_shader, unsupported_shader_check);
  REGISTER_CASE_CHECK(
      test_case_check_unsupported_route, unsupported_route_check);
  REGISTER_CASE_CHECK(test_case_check_error, error_check);
}
TEST(CaseCheckRegistry, RejectsNullAndDuplicateChecks) {
  CaseCheckRegistry registry;
  registry.register_check("check", pass_check);
  EXPECT_EQ(registry.get_check("check"), pass_check);
  EXPECT_THROW(
      registry.register_check("check", fail_check), std::invalid_argument);
  EXPECT_EQ(registry.get_check("check"), pass_check);
  EXPECT_THROW(registry.register_check("", pass_check), std::invalid_argument);
  EXPECT_THROW(registry.register_check("null", nullptr), std::invalid_argument);
  EXPECT_THROW(registry.get_check("missing"), std::invalid_argument);
}
TEST_F(GpuCaseCheckRegistry, ExecutesRegisteredChecksAndSurfacesFailure) {
  const auto pass = run_case_check(
      {{"name", "pass"},
       {"check", "test_case_check_pass"},
       {"legacy_case", {{"custom", true}}}});
  EXPECT_EQ(pass.at("validation").at("status"), "PASS");
  EXPECT_FALSE(pass.contains("legacy_case"));
  const auto fail = run_case_check({{"check", "test_case_check_fail"}});
  EXPECT_EQ(fail.at("validation").at("status"), "FAIL");
  EXPECT_EQ(fail.at("validation").at("error"), "intentional");
}
TEST(CaseCheckRegistry, ParsesOnlyUnambiguousStandaloneChecks) {
  const auto parsed = parse_configs(
      {{"check", "test_case_check_pass"}, {"parameters", {{"value", 3}}}});
  ASSERT_EQ(parsed.size(), 1);
  EXPECT_EQ(parsed[0].at("parameters").at("value"), 3);
  EXPECT_THROW(parse_configs({{"check", ""}}), std::invalid_argument);
  EXPECT_THROW(
      parse_configs({{"check", "x"}, {"operator", "op"}}),
      std::invalid_argument);
  EXPECT_THROW(
      parse_configs({{"check", "x"}, {"args_sweep", Json::array()}}),
      std::invalid_argument);
}
TEST_F(GpuCaseCheckRegistry, SkipsNotSupportedExceptionsWithoutRequirements) {
  for (const char* name :
       {"test_case_check_unsupported_shader",
        "test_case_check_unsupported_route"}) {
    const auto result = run_case_check({{"check", name}});
    EXPECT_EQ(result.at("status"), "SKIP");
    EXPECT_EQ(result.at("validation").at("status"), "SKIP");
    EXPECT_FALSE(result.at("skip_reason").get<std::string>().empty());
  }
}
TEST_F(GpuCaseCheckRegistry, OrdinaryExceptionsRemainFailures) {
  const auto result = run_case_check({{"check", "test_case_check_error"}});
  EXPECT_EQ(result.at("validation").at("status"), "FAIL");
  EXPECT_EQ(result.at("validation").at("error"), "Ordinary test error");
  EXPECT_FALSE(result.contains("skip_reason"));
}
} // namespace
} // namespace vkcompute::shader_profiling
