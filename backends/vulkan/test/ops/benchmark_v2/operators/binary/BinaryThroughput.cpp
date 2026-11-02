// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

namespace vkcompute::shader_profiling {
namespace {

double copy_flops(const Json&) {
  return 0;
}

double add_flops(const Json& spec) {
  return 2 * throughput::tensor_elements(throughput::output_tensor(spec));
}

} // namespace

REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_SHADER_MEMORY_TRAFFIC(buffer_to_buffer, throughput::tensor_traffic);
  REGISTER_SHADER_COMPUTE_FLOPS(buffer_to_buffer, copy_flops);
  REGISTER_SHADER_MEMORY_TRAFFIC(binary_add, throughput::tensor_traffic);
  REGISTER_SHADER_COMPUTE_FLOPS(binary_add, add_flops);
}

} // namespace vkcompute::shader_profiling
