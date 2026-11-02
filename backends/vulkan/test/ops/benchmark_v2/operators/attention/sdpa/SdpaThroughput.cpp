// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

namespace vkcompute::shader_profiling {
namespace {
double sdpa_flops(const Json& spec) {
  const auto q = throughput::tensor_argument(spec, 0)
                     .at("sizes")
                     .get<std::vector<int64_t>>();
  const bool fused = spec.value("operator_name", spec.value("operator", "")) ==
      "sdpa_with_kv_cache.default";
  const auto k = throughput::tensor_argument(spec, fused ? 3 : 1)
                     .at("sizes")
                     .get<std::vector<int64_t>>();
  const int64_t pos = spec.at("args").at(fused ? 5 : 3).at("int");
  const int64_t context = pos + q.at(1);
  if (pos < 0 || context > k.at(1))
    throw std::invalid_argument(
        "SDPA work count context exceeds cache capacity");
  return 4.0 * q.at(1) * q.at(2) * q.at(3) * context;
}
REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      llama.custom_sdpa.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(llama.custom_sdpa.default, sdpa_flops);
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      sdpa_with_kv_cache.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(sdpa_with_kv_cache.default, sdpa_flops);
}

} // namespace
} // namespace vkcompute::shader_profiling
