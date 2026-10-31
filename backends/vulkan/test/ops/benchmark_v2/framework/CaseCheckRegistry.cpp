// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/CaseCheckRegistry.h>

#include <stdexcept>

namespace vkcompute::shader_profiling {

void CaseCheckRegistry::register_check(
    const std::string& name,
    CaseCheckFunction function) {
  if (name.empty() || !function)
    throw std::invalid_argument("Case check requires a name and callable");
  if (!checks_.emplace(name, function).second)
    throw std::invalid_argument("Duplicate case check: " + name);
}

CaseCheckFunction CaseCheckRegistry::get_check(const std::string& name) const {
  const auto entry = checks_.find(name);
  if (entry == checks_.end())
    throw std::invalid_argument("Unknown case check: " + name);
  return entry->second;
}

CaseCheckRegistry& case_check_registry() {
  static CaseCheckRegistry registry;
  return registry;
}

Report run_case_check(const Json& spec) {
  const std::string name = spec.at("check");
  Report result{{"case_name", spec.value("name", name)}, {"check_name", name}};
  try {
    GraphConfig config;
    config.force_resize = true;
    ComputeGraph graph(config);
    Json validation = case_check_registry().get_check(name)(spec, graph);
    const std::string status = validation.value("status", "");
    if (status != "PASS" && status != "FAIL" && status != "SKIP")
      throw std::invalid_argument(
          "Case check must return a PASS, FAIL, or SKIP status");
    result["validation"] = Report(validation);
    if (status == "SKIP") {
      result["status"] = "SKIP";
      result["skip_reason"] =
          validation.value("reason", "Case check unsupported");
    }
  } catch (const UnsupportedRoute& error) {
    result["status"] = "SKIP";
    result["skip_reason"] = error.what();
    result["validation"] = {{"status", "SKIP"}};
  } catch (const vkapi::ShaderNotSupportedError& error) {
    result["status"] = "SKIP";
    result["skip_reason"] = error.what();
    result["validation"] = {{"status", "SKIP"}};
  } catch (const std::exception& error) {
    result["validation"] = {{"status", "FAIL"}, {"error", error.what()}};
  }
  return result;
}

} // namespace vkcompute::shader_profiling
