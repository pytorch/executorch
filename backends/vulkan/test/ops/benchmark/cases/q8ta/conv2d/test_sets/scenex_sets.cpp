// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/conv2d/conv2d.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_conv2d {

namespace {

std::vector<TestCase> generate_scenex_test_cases(
    const std::vector<Conv2dConfig>& configs,
    const std::vector<std::string>& routes) {
  std::vector<TestCase> test_cases;
  for (const std::string& route : routes) {
    for (const Conv2dConfig& config : configs) {
      test_cases.push_back(create_scenex_test_case(config, route));
    }
  }
  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/conv2d", "scenex_regular") {
  const std::vector<Conv2dConfig> configs = {
      {OutInChannels(128, 64),
       InputSize2D(40, 51),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1,
       60},
      {OutInChannels(256, 128),
       InputSize2D(20, 26),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1,
       60},
  };
  const std::vector<std::string> routes = {"auto", "direct", "im2col"};
  return {
      generate_scenex_test_cases(configs, routes),
      scenex_direct_reference,
      quantized_conv2d_flop_calculator,
      /*warmup_runs=*/3,
      /*benchmark_runs=*/10};
}

REGISTER_TEST_CASE_SET("q8ta/conv2d", "scenex_grouped") {
  const std::vector<Conv2dConfig> configs = {
      {OutInChannels(64, 64),
       InputSize2D(128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       2,
       60},
      {OutInChannels(128, 128),
       InputSize2D(128, 128),
       KernelSize(5, 5),
       Stride(2, 2),
       Padding(2, 2),
       Dilation(1, 1),
       4,
       60},
      {OutInChannels(64, 64),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       2,
       60},
  };
  const std::vector<std::string> routes = {"auto", "direct", "im2col"};
  return {
      generate_scenex_test_cases(configs, routes),
      scenex_direct_reference,
      quantized_conv2d_flop_calculator,
      /*warmup_runs=*/3,
      /*benchmark_runs=*/10};
}

} // namespace q8ta_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
