// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/Q8taConv2dPW.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/im2col/Q8taConv2dIm2Col.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/CaseCheckRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/OperatorBenchmark.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Validation.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/q8ta/Q8taValidationChecks.h>
#include <stdexcept>
namespace vkcompute::shader_profiling {
Json check_q8ta_unsigned_dot_selection(const Json& spec, ComputeGraph& graph) {
  const auto& parameters = spec.at("parameters");
  const int64_t expected_limit = parameters.at("accumulator_limit_bytes");
  reference::require(
      expected_limit == kMaxUnsignedDotAccumulatorBytes,
      "Unsigned dot accumulator boundary changed");
  const auto& adapter = *graph.context()->adapter_ptr();
  const bool prefers_unsigned = adapter.accelerates_unsigned_packed4x8_dot() &&
      !adapter.accelerates_signed_packed4x8_dot();
  Json decisions = Json::array();
  for (const Json& size : parameters.at("test_kernel_sizes")) {
    const int64_t k = size.get<int64_t>();
    const bool selected = can_use_unsigned_pw_dot(adapter, k);
    reference::require(
        selected == (prefers_unsigned && k <= expected_limit),
        "Unsigned dot selection violated capability or accumulator limit");
    decisions.push_back({{"kernel_bytes", k}, {"unsigned_selected", selected}});
  }
  return {
      {"status", "PASS"},
      {"prefers_unsigned", prefers_unsigned},
      {"decisions", decisions}};
}
Json check_q8ta_streaming_dynamic_shrink(
    const Json& spec,
    ComputeGraph& graph) {
  const Json& parameters = spec.at("parameters");
  const Json& plan_args = parameters.at("stream_plan");
  const auto plan = make_q8ta_conv2d_stream_plan(
      plan_args.at("batch").get<int64_t>(),
      plan_args.at("flattened_kernel_size").get<int64_t>(),
      plan_args.at("out_height").get<int64_t>(),
      plan_args.at("out_width").get<int64_t>(),
      plan_args.at("scratch_budget_bytes").get<int64_t>());
  const Json& expected = parameters.at("expected_plan");
  reference::require(
      plan.feasible &&
          plan.num_tiles == expected.at("num_tiles").get<int64_t>() &&
          plan.rows_per_tile >= expected.at("min_rows_per_tile").get<int64_t>(),
      "Dynamic shrink case did not create dead stream tiles");
  const int64_t resized_bytes =
      plan_args.at("flattened_kernel_size").get<int64_t>() *
      plan.rows_per_tile *
      utils::align_up_4(parameters.at("resized_width").get<int64_t>());
  reference::require(
      resized_bytes <= plan_args.at("scratch_budget_bytes").get<int64_t>(),
      "Resized im2col scratch exceeds stream budget");
  const Json operator_tensors =
      parse_configs(parameters.at("operator_case")).front().at("tensors");
  for (const char* name : {"input_name", "output_name"})
    reference::require(
        operator_tensors.at(parameters.at(name).get<std::string>())
                .value("dtype", "") == "int8x4",
        "Dynamic Q8 convolution requires native int8x4 input/output");
  Json result = validate_operator_dynamic_resize(spec, graph);
  result["stream_tiles"] = plan.num_tiles;
  result["resized_scratch_bytes"] = resized_bytes;
  return result;
}
REGISTER_CASE_CHECKS {
  REGISTER_CASE_CHECK(
      q8ta_unsigned_dot_selection, check_q8ta_unsigned_dot_selection);
  REGISTER_CASE_CHECK(
      q8ta_streaming_dynamic_shrink, check_q8ta_streaming_dynamic_shrink);
}
} // namespace vkcompute::shader_profiling
