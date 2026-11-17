// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

namespace vkcompute::shader_profiling {
namespace {
double choose_qparams_flops(const Json& spec) {
  const auto sizes = throughput::tensor_argument(spec, 0)
                         .at("sizes")
                         .get<std::vector<int64_t>>();
  return double(sizes.at(0)) * (2.0 * sizes.at(1) + 15);
}
REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      etvk.choose_qparams_per_row.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(
      etvk.choose_qparams_per_row.default, choose_qparams_flops);
}

} // namespace
} // namespace vkcompute::shader_profiling
