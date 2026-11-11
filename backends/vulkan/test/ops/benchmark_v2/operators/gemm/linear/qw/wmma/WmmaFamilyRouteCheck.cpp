// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/wmma/WmmaLinear.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/CaseCheckRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/OperatorBenchmark.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Reference.h>

#include <string>
#include <vector>

namespace vkcompute::shader_profiling {

namespace {

// The kernels the graph dispatches in its current state, from the shader
// profiling records; empty without timestamp queries (e.g. SwiftShader).
std::vector<std::string> dispatched_kernels(ComputeGraph& graph) {
  api::Context& context = *graph.context();
  context.flush();
  context.initialize_querypool();
  if (!context.querypool()) {
    return {};
  }
  context.set_shader_profiling_enabled(true);
  context.set_cmd(true);
  context.cmd_reset_querypool();
  for (const auto& node : graph.execute_nodes()) {
    node->encode(&graph);
  }
  vkapi::VulkanFence fence(context.device());
  context.submit_cmd_to_gpu(fence.get_submit_handle(), true);
  fence.wait();
  std::vector<std::string> kernels;
  for (const auto& record : context.querypool().get_shader_timestamp_data()) {
    kernels.push_back(record.kernel_name);
  }
  context.set_shader_profiling_enabled(false);
  context.flush();
  return kernels;
}

bool is_family_kernel(const std::string& kernel) {
  return kernel.find("_wmma") != std::string::npos;
}

} // namespace

// operator_dynamic_resize, plus the kernel each step dispatches on this
// device. A step's "expected_kernels" maps a family name (CoopmatFamily.h) to
// the prefix of the kernel it must dispatch; a family without an entry, and
// every device that matches no family, must dispatch no family kernel.
Json check_wmma_family_route(const Json& spec, ComputeGraph& graph) {
  const std::string family =
      coopmat_family_name(coopmat_family(coopmat_device_key(&graph)));
  bool asserted = false;
  Json result = validate_operator_dynamic_resize(
      spec, graph, [&](const Json& step, ComputeGraph& step_graph) {
        const std::vector<std::string> kernels = dispatched_kernels(step_graph);
        Json expected = nullptr;
        if (step.contains("expected_kernels") &&
            step.at("expected_kernels").contains(family)) {
          expected = step.at("expected_kernels").at(family);
        }
        if (!kernels.empty()) {
          asserted = true;
          bool found = false;
          for (const std::string& kernel : kernels) {
            if (expected.is_string()) {
              found |= kernel.rfind(expected.get<std::string>(), 0) == 0;
            } else {
              reference::require(
                  !is_family_kernel(kernel),
                  "Family kernel " + kernel + " dispatched on family " +
                      family + ", which expects none for this step");
            }
          }
          reference::require(
              !expected.is_string() || found,
              "Family " + family + " did not dispatch " + expected.dump() +
                  "; dispatched: " + Json(kernels).dump());
        }
        return Json{{"kernels", kernels}, {"expected", expected}};
      });
  result["family"] = family;
  result["route_asserted"] = asserted;
  return result;
}

REGISTER_CASE_CHECKS {
  REGISTER_CASE_CHECK(wmma_family_route, check_wmma_family_route);
}

} // namespace vkcompute::shader_profiling
