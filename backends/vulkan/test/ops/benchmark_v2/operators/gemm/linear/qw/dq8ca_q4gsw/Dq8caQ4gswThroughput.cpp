// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

namespace vkcompute::shader_profiling {
namespace {
double quantized_linear_legacy_flops(const Json& spec) {
  const auto sizes = throughput::tensor_argument(spec, 0)
                         .at("sizes")
                         .get<std::vector<int64_t>>();
  return throughput::tensor_elements(throughput::output_tensor(spec)) *
      (3.0 * sizes.back() + 1);
}
REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.linear_dq8ca_q4gsw.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(
      et_vk.linear_dq8ca_q4gsw.default, quantized_linear_legacy_flops);
}

} // namespace
} // namespace vkcompute::shader_profiling
