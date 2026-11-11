// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

namespace vkcompute::shader_profiling {
namespace {
double add_tensor_flops(const Json& spec) {
  return 2 * throughput::tensor_elements(throughput::output_tensor(spec));
}
REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(aten.add.Tensor, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(aten.add.Tensor, add_tensor_flops);
}
} // namespace
} // namespace vkcompute::shader_profiling
