// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dThroughputCommon.h>

#include <stdexcept>

namespace vkcompute::shader_profiling {
namespace {

double conv2d_flops(const Json& spec) {
  const auto weight_sizes = throughput::tensor_argument(spec, 1)
                                .at("sizes")
                                .get<std::vector<int64_t>>();
  if (weight_sizes.size() != 4)
    throw std::invalid_argument(
        "Conv2d work count requires rank-4 input/output and supported weight shape");
  if (spec.at("args").at(6).at("bool").get<bool>())
    throw std::invalid_argument(
        "Conv2d work count does not support transposed convolution");
  return throughput::convolution_flops(
      spec,
      1,
      2,
      weight_sizes,
      spec.at("args").at(3).at("int_list").get<std::vector<int64_t>>(),
      spec.at("args").at(4).at("int_list").get<std::vector<int64_t>>(),
      spec.at("args").at(5).at("int_list").get<std::vector<int64_t>>(),
      spec.at("args").at(8).at("int"));
}
} // namespace

REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(
      aten.convolution.default, throughput::tensor_traffic);
  REGISTER_OPERATOR_COMPUTE_FLOPS(aten.convolution.default, conv2d_flops);
  REGISTER_SHADER_MEMORY_TRAFFIC(conv2d, throughput::tensor_traffic);
}

} // namespace vkcompute::shader_profiling
