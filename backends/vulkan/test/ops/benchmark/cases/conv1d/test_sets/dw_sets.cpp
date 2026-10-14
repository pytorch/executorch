// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/conv1d/conv1d.h>

#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace conv1d {

namespace {

std::vector<TestCase> generate_dw_test_cases(
    const std::vector<Conv1dDWConfig>& configs,
    const std::vector<utils::StorageType>& storage_types,
    const std::vector<vkapi::ScalarType>& dtypes) {
  std::vector<TestCase> test_cases;

  for (const auto& cfg : configs) {
    for (auto st : storage_types) {
      for (auto dtype : dtypes) {
        test_cases.push_back(create_conv1d_dw_test_case(cfg, dtype, st));
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("conv1d", "dw_correctness") {
  // Accuracy shapes
  const std::vector<Conv1dDWConfig> configs = {
      // {N, C, L, K, stride, padding, dilation, has_bias}
      {1, 16, 64, 3, 1, 1, 1, false},
      {1, 32, 128, 5, 1, 2, 1, true},
      {1, 64, 32, 3, 2, 1, 1, false},
      {2, 16, 64, 3, 1, 1, 1, true},
      {1, 16, 64, 7, 1, 3, 2, false},
      // Non-aligned channel counts (not a multiple of 4)
      {1, 5, 64, 3, 1, 1, 1, false},
      {1, 5, 64, 3, 1, 1, 1, true},
      {1, 7, 32, 5, 1, 2, 1, false},
      {1, 13, 48, 3, 2, 1, 1, true},
      {2, 7, 64, 3, 1, 1, 1, false},
  };
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};
  const std::vector<vkapi::ScalarType> dtypes = {vkapi::kFloat};
  return {
      generate_dw_test_cases(configs, storage_types, dtypes),
      conv1d_dw_reference_impl,
      conv1d_dw_flop_calculator};
}

REGISTER_TEST_CASE_SET("conv1d", "dw_performance") {
  // Performance shapes (half + float)
  const std::vector<Conv1dDWConfig> configs = {
      {1, 256, 1024, 3, 1, 1, 1, false},
      {1, 512, 2048, 5, 1, 2, 1, true},
      {1, 128, 4096, 31, 1, 15, 1, false},
  };
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};
  const std::vector<vkapi::ScalarType> dtypes = {vkapi::kFloat, vkapi::kHalf};
  return {
      generate_dw_test_cases(configs, storage_types, dtypes),
      conv1d_dw_reference_impl,
      conv1d_dw_flop_calculator};
}

} // namespace conv1d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
