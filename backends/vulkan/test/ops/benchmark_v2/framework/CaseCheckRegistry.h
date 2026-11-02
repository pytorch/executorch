// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Report.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>

#include <unordered_map>

namespace vkcompute::shader_profiling {
using CaseCheckFunction = Json (*)(const Json&, ComputeGraph&);
class CaseCheckRegistry final {
 private:
  std::unordered_map<std::string, CaseCheckFunction> checks_;

 public:
  void register_check(const std::string& name, CaseCheckFunction function);
  CaseCheckFunction get_check(const std::string& name) const;
};
CaseCheckRegistry& case_check_registry();
Report run_case_check(const Json& spec);
} // namespace vkcompute::shader_profiling

#define REGISTER_CASE_CHECK(name, function)                            \
  ::vkcompute::shader_profiling::case_check_registry().register_check( \
      #name, &function)
#define REGISTER_CASE_CHECKS                                    \
  static void register_case_checks();                           \
  static const ::vkcompute::shader_profiling::RouteRegisterInit \
      case_checks_reg(&register_case_checks);                   \
  static void register_case_checks()
