// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/Q8taConv2dPW.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/im2col/Q8taConv2dIm2Col.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/CaseCheckRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/OperatorBenchmark.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>
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
  Json operator_spec = parse_configs(parameters.at("operator_case")).front();
  const std::string output_name = parameters.at("output_name");
  Json input_spec = operator_spec;
  input_spec["tensors"].erase(output_name);
  Json output_spec = operator_spec;
  output_spec["tensors"] = {
      {output_name, operator_spec.at("tensors").at(output_name)}};
  auto host_data = prepare_tensor_data(input_spec);
  const auto input_tensors = create_tensors(input_spec, graph, host_data, true);
  auto output_data = prepare_tensor_data(output_spec);
  const auto output_tensors =
      create_tensors(output_spec, graph, output_data, false);
  host_data.merge(output_data);
  auto tensors = input_tensors;
  tensors.insert(output_tensors.begin(), output_tensors.end());
  for (const auto& input : graph.inputs())
    reference::require(
        input.value != tensors.at(output_name),
        "Dynamic validation output must not be staged as an input");
  reference::require(
      graph.dtype_of(tensors.at(
          parameters.at("input_name").get<std::string>())) == vkapi::kInt8x4 &&
          graph.dtype_of(tensors.at(output_name)) == vkapi::kInt8x4,
      "Dynamic Q8 convolution requires native int8x4 input/output");
  const auto args = build_operator_args(operator_spec, graph, tensors);
  build_operator_route(operator_spec, graph, args);
  Json accesses = Json::object();
  for (const auto& tensor : tensors)
    accesses[tensor.first] = tensor.first == output_name ? "write" : "read";
  const auto staging =
      add_validation_outputs(operator_spec, graph, tensors, accesses);
  graph.prepare();
  graph.prepack();
  Json validations = Json::array();
  for (const Json& step : spec.at("validation_steps")) {
    Json resized = operator_spec;
    auto data = host_data;
    const Json& sizes = step.at("tensor_sizes");
    reference::require(
        !sizes.contains(output_name),
        "Dynamic validation steps must not resize the output manually");
    for (const auto& item : sizes.items()) {
      const auto shape = item.value().get<std::vector<int64_t>>();
      const ValueRef ref = input_tensors.at(item.key());
      bool found = false;
      for (size_t i = 0; i < graph.inputs().size(); ++i)
        if (graph.inputs().at(i).value == ref) {
          graph.resize_input(i, shape);
          found = true;
          break;
        }
      reference::require(
          found, "Dynamic validation size target is not a staged input");
      resized["tensors"][item.key()]["sizes"] = shape;
      const size_t bytes = tensor_numel(item.value());
      reference::require(
          bytes <= data.at(item.key()).size(),
          "Dynamic validation shape exceeds upper input allocation");
      data.at(item.key()).resize(bytes);
    }
    resized["tensors"][output_name]["sizes"] = step.at("expected_output_sizes");
    graph.propagate_resize();
    reference::require(
        graph.sizes_of(tensors.at(output_name)) ==
            step.at("expected_output_sizes").get<std::vector<int64_t>>(),
        "Streaming im2col output did not resize");
    const size_t output_bytes = tensor_numel(step.at("expected_output_sizes"));
    reference::require(
        output_bytes <= data.at(output_name).size(),
        "Dynamic validation output exceeds upper allocation");
    data.at(output_name).resize(output_bytes);
    copy_validation_inputs(resized, graph, input_tensors, data);
    graph.execute();
    if (step.value("validate_output", true)) {
      const Json validation =
          validate_outputs(resized, graph, tensors, staging, data);
      reference::require(
          validation.at("status") == "PASS",
          "Streaming im2col resize validation failed: " + validation.dump());
      validations.push_back(validation);
    }
  }
  return {
      {"status", "PASS"},
      {"validation_steps", spec.at("validation_steps").size()},
      {"validated_outputs", validations.size()},
      {"stream_tiles", plan.num_tiles},
      {"resized_scratch_bytes", resized_bytes}};
}
REGISTER_CASE_CHECKS {
  REGISTER_CASE_CHECK(
      q8ta_unsigned_dot_selection, check_q8ta_unsigned_dot_selection);
  REGISTER_CASE_CHECK(
      q8ta_streaming_dynamic_shrink, check_q8ta_streaming_dynamic_shrink);
}
} // namespace vkcompute::shader_profiling
