// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

#include <algorithm>
#include <cmath>

namespace vkcompute::shader_profiling {
namespace {
double pixel_shuffle_flops(const Json& spec) {
  const float in_scale = spec.at("args").at(1).at("float").get<float>();
  const float out_scale = 1.0f / spec.at("args").at(3).at("float").get<float>();
  const bool passthrough = std::abs(in_scale - out_scale) <=
          1e-7f * std::max(std::abs(in_scale), 1e-7f) &&
      spec.at("args").at(2).at("int") == spec.at("args").at(4).at("int");
  return (passthrough ? 0 : 3) *
      throughput::tensor_elements(throughput::output_tensor(spec));
}
REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.q8ta_pixel_shuffle.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(
      et_vk.q8ta_pixel_shuffle.default, pixel_shuffle_flops);
}
} // namespace
} // namespace vkcompute::shader_profiling
