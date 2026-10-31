// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/GemmThroughputCommon.h>

#include <stdexcept>

namespace vkcompute::shader_profiling {
namespace {

double q4gsw_linear_flops(const Json& spec) {
  const double flops = throughput::gemm_flops(spec, 2, 4);
  const auto input_sizes = throughput::tensor_argument(spec, 0)
                               .at("sizes")
                               .get<std::vector<int64_t>>();
  const auto output_sizes =
      throughput::output_tensor(spec).at("sizes").get<std::vector<int64_t>>();
  const int64_t k = input_sizes.back();
  const int64_t n = output_sizes.back();

  const int64_t group_size = spec.at("args").at(3).at("int");
  const Json& scales = throughput::tensor_argument(spec, 2);
  const auto scale_sizes = scales.at("sizes").get<std::vector<int64_t>>();
  if (group_size <= 0 || k % group_size != 0 ||
      scale_sizes != std::vector<int64_t>({k / group_size, n})) {
    throw std::invalid_argument(
        "Q4 work count requires scales [K/group_size, N]");
  }
  return flops;
}

double q4_shader_traffic(const Json& spec, size_t inactive_input) {
  Json filtered = spec;
  Json& bindings = filtered.at("bindings");
  unsigned seen_indices = 0;
  for (auto binding = bindings.begin(); binding != bindings.end();) {
    const Json& tensor = binding.value();
    if (!tensor.contains("sizes")) {
      ++binding;
      continue;
    }
    const size_t index = tensor.at("binding_index").get<size_t>();
    if (index >= 6 || (seen_indices & (1u << index))) {
      throw std::invalid_argument(
          "Q4 shader memory count requires unique tensor binding indices 0..5");
    }
    seen_indices |= 1u << index;
    if (index == inactive_input || (index == 5 && tensor.at("sizes").empty())) {
      binding = bindings.erase(binding);
    } else {
      ++binding;
    }
  }
  if (seen_indices != 63) {
    throw std::invalid_argument(
        "Q4 shader memory count requires all six tensor binding slots");
  }
  return throughput::tensor_traffic(filtered);
}

double q4_gemm_shader_traffic(const Json& spec) {
  return q4_shader_traffic(spec, 2);
}

double q4_tin_shader_traffic(const Json& spec) {
  return q4_shader_traffic(spec, 1);
}

} // namespace

REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.q4gsw_linear.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(
      et_vk.q4gsw_linear.default, q4gsw_linear_flops);
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.linear_q4gsw.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(
      et_vk.linear_q4gsw.default, q4gsw_linear_flops);
  REGISTER_SHADER_MEMORY(
      transpose_cast_contig_to_vectorized_half_buffer_half_buffer,
      throughput::tensor_traffic);
  REGISTER_SHADER_MEMORY_TRAFFIC(q4gsw_linear_gemm__, q4_gemm_shader_traffic);
  REGISTER_SHADER_MEMORY_TRAFFIC(
      q4gsw_linear_gemm__tin__, q4_tin_shader_traffic);
}

} // namespace vkcompute::shader_profiling
