// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

namespace vkcompute::shader_profiling {
namespace {
double relu_flops(const Json& spec) {
  return 3 * throughput::tensor_elements(throughput::output_tensor(spec));
}
REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      et_vk.q8ta_relu.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(et_vk.q8ta_relu.default, relu_flops);
}
} // namespace
} // namespace vkcompute::shader_profiling
