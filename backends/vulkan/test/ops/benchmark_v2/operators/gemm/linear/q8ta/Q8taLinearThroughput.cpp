// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

namespace vkcompute::shader_profiling {
namespace {
double q8ta_linear_flops(const Json& spec) {
  const auto& input = throughput::tensor_argument(spec, 0);
  const double k = input.at("sizes").back().get<double>();
  return 2 * k * throughput::tensor_elements(throughput::output_tensor(spec));
}
REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.q8ta_linear.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.q8ta_linear_gemv.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(et_vk.q8ta_linear.default, q8ta_linear_flops);
  REGISTER_OPERATOR_COMPUTE_FLOPS(
      et_vk.q8ta_linear_gemv.default, q8ta_linear_flops);
}
} // namespace
} // namespace vkcompute::shader_profiling
