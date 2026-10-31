// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/GemmThroughputCommon.h>

#include <stdexcept>

namespace vkcompute::shader_profiling {
namespace {

double linear_flops(const Json& spec) {
  return throughput::gemm_flops(spec, 1, 2);
}

double linear_shader_flops(const Json& spec) {
  const Json* input = nullptr;
  for (const auto& binding : spec.at("bindings").items()) {
    const Json& tensor = binding.value();
    if (tensor.contains("sizes") && tensor.value("access", "") == "read" &&
        (!input ||
         tensor.value("binding_index", SIZE_MAX) <
             input->value("binding_index", SIZE_MAX))) {
      input = &tensor;
    }
  }
  if (!input) {
    throw std::invalid_argument(
        "Linear shader work count requires first input binding");
  }
  const Json& output = throughput::output_tensor(spec);
  const double output_elements = throughput::tensor_elements(output);
  const double input_elements = throughput::tensor_elements(*input);
  const auto input_sizes = input->at("sizes").get<std::vector<int64_t>>();
  const auto output_sizes = output.at("sizes").get<std::vector<int64_t>>();
  if (input_elements / input_sizes.back() !=
      output_elements / output_sizes.back()) {
    throw std::invalid_argument(
        "Linear shader input/output row counts disagree");
  }
  return output_elements * 2 * double(input_sizes.back());
}

double linear_bias_shader_flops(const Json& spec) {
  return linear_shader_flops(spec) +
      throughput::tensor_elements(throughput::output_tensor(spec));
}

} // namespace

REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      aten.linear.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(aten.linear.default, linear_flops);
  REGISTER_SHADER_MEMORY_TRAFFIC(linear_vec, throughput::tensor_traffic);
  REGISTER_SHADER_COMPUTE_FLOPS(linear_vec, linear_shader_flops);
  REGISTER_SHADER_COMPUTE_FLOPS(linear_vec_bias, linear_bias_shader_flops);
}

} // namespace vkcompute::shader_profiling
