// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

namespace vkcompute::shader_profiling {
namespace {
double matmul_flops(const Json& spec) {
  const size_t input_index =
      spec.at("operator") == "aten.addmm.default" ? 1 : 0;
  const auto sizes = throughput::tensor_argument(spec, input_index)
                         .at("sizes")
                         .get<std::vector<int64_t>>();
  return 2.0 * sizes.back() *
      throughput::tensor_elements(throughput::output_tensor(spec));
}
REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(aten.mm.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(aten.mm.default, matmul_flops);
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      aten.bmm.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(aten.bmm.default, matmul_flops);
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      aten.addmm.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(aten.addmm.default, matmul_flops);
}

} // namespace
} // namespace vkcompute::shader_profiling
