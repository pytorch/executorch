// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/conv2d/conv2d.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace conv2d {

namespace {

std::vector<TestCase> generate_dw_test_cases(
    const std::vector<Conv2dDwConfig>& configs,
    const std::vector<vkapi::ScalarType>& dtypes,
    const std::vector<std::string>& impls) {
  std::vector<TestCase> test_cases;

  for (const auto& config : configs) {
    for (auto dtype : dtypes) {
      for (const auto& impl : impls) {
        test_cases.push_back(create_conv2d_dw_test_case(
            config, dtype, kStorageType, kMemoryLayout, impl));
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("conv2d", "dw_correctness") {
  // Accuracy shapes (small enough for float reference validation)
  std::vector<Conv2dDwConfig> accuracy_configs = {
      {InputDims(1, 8, 16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 8, 16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      {InputDims(1, 8, 16, 16),
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(1, 1),
       false},
      {InputDims(1, 8, 16, 16),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // Non-multiple-of-4 channels
      {InputDims(1, 11, 16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 3, 16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
  };

  // Accuracy test cases (float only, auto-selected implementation)
  return {
      generate_dw_test_cases(accuracy_configs, {vkapi::kFloat}, {""}),
      conv2d_dw_reference_impl,
      conv2d_dw_flop_calculator};
}

REGISTER_TEST_CASE_SET("conv2d", "dw_performance") {
  // EdgeTAM depthwise shapes (from profiling data)
  std::vector<Conv2dDwConfig> perf_configs = {
      // Backbone stem and early stages
      {InputDims(1, 24, 512, 512),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 48, 256, 256),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 48, 256, 256),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 96, 128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 96, 128, 128),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 192, 64, 64),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 192, 64, 64),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 384, 32, 32),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // 5x5 kernels
      {InputDims(1, 48, 256, 256),
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(1, 1),
       false},
      {InputDims(1, 96, 128, 128),
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(1, 1),
       false},
      // FPN/Neck
      {InputDims(1, 256, 256, 256),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 256, 128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
  };

  const std::vector<std::string> impls = {
      // Auto-selection (empty impl_selector)
      "",
      // Force b4x2 variant
      "b4x2",
      // Force b1x1 variant (only for 3x3 kernels; for 5x5 it falls back
      // to default, but we still generate it to test the fallback path)
      "b1x1",
  };

  // Performance test cases (float and half)
  return {
      generate_dw_test_cases(
          perf_configs, {vkapi::kFloat, vkapi::kHalf}, impls),
      conv2d_dw_reference_impl,
      conv2d_dw_flop_calculator};
}

} // namespace conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
