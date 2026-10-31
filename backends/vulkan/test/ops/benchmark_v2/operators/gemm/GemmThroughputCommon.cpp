// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/GemmThroughputCommon.h>

#include <stdexcept>

namespace vkcompute::shader_profiling::throughput {

double gemm_flops(const Json& spec, size_t weight_packing, size_t bias_index) {
  if (weight_packing == 0)
    throw std::invalid_argument("GEMM weight packing must be positive");
  const Json& input = throughput::tensor_argument(spec, 0);
  const Json& weight = throughput::tensor_argument(spec, 1);
  const Json& output = throughput::output_tensor(spec);
  const auto input_sizes = input.at("sizes").get<std::vector<int64_t>>();
  const auto weight_sizes = weight.at("sizes").get<std::vector<int64_t>>();
  const auto output_sizes = output.at("sizes").get<std::vector<int64_t>>();
  const double input_elements = throughput::tensor_elements(input);
  const double output_elements = throughput::tensor_elements(output);
  throughput::tensor_elements(weight);
  if (weight_sizes.size() != 2 || input_sizes.empty() || output_sizes.empty()) {
    throw std::invalid_argument(
        "GEMM work count requires input/output ranks >= 1 and weight rank 2");
  }
  const int64_t k = input_sizes.back();
  const int64_t n = output_sizes.back();
  if (weight_sizes[0] != n || weight_sizes[1] != k / weight_packing ||
      input_elements / k != output_elements / n || k % weight_packing != 0) {
    throw std::invalid_argument("GEMM work count tensor dimensions disagree");
  }
  const bool bias = !spec.at("args").at(bias_index).is_null();
  if (bias &&
      throughput::tensor_elements(
          throughput::tensor_argument(spec, bias_index)) != n) {
    throw std::invalid_argument("GEMM work count bias must have N elements");
  }
  return output_elements * (2 * double(k) + (bias ? 1 : 0));
}

} // namespace vkcompute::shader_profiling::throughput
