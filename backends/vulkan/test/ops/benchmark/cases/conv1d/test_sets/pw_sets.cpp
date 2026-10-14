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

std::vector<TestCase> generate_pw_test_cases(
    const std::vector<Conv1dPWConfig>& configs,
    const std::vector<utils::StorageType>& storage_types,
    const std::vector<vkapi::ScalarType>& dtypes) {
  std::vector<TestCase> test_cases;

  for (const auto& cfg : configs) {
    for (auto st : storage_types) {
      for (auto dtype : dtypes) {
        test_cases.push_back(create_conv1d_pw_test_case(cfg, dtype, st));
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("conv1d", "pw_correctness") {
  // Accuracy shapes (float, small)
  const std::vector<Conv1dPWConfig> configs = {
      {1, 16, 32, 64, false},
      {1, 16, 32, 64, true},
      {1, 32, 16, 128, false},
      {1, 32, 16, 128, true},
      {1, 64, 64, 32, false},
      {1, 128, 256, 16, true},
      {2, 16, 32, 64, false},
      {2, 16, 32, 64, true},
      // Non-aligned channel counts (not a multiple of 4)
      {1, 5, 7, 64, false},
      {1, 5, 7, 64, true},
      {1, 13, 17, 48, false},
      {1, 13, 17, 48, true},
      {1, 7, 5, 32, false},
      {2, 5, 13, 64, true},
  };
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};
  const std::vector<vkapi::ScalarType> dtypes = {vkapi::kFloat};
  return {
      generate_pw_test_cases(configs, storage_types, dtypes),
      conv1d_pw_reference_impl,
      conv1d_pw_flop_calculator};
}

REGISTER_TEST_CASE_SET("conv1d", "pw_performance") {
  // Performance shapes (half + float)
  const std::vector<Conv1dPWConfig> configs = {
      {1, 256, 512, 1024, false},
      {1, 256, 512, 1024, true},
      {1, 512, 256, 2048, false},
      {1, 128, 128, 4096, true},
  };
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};
  const std::vector<vkapi::ScalarType> dtypes = {vkapi::kFloat, vkapi::kHalf};
  return {
      generate_pw_test_cases(configs, storage_types, dtypes),
      conv1d_pw_reference_impl,
      conv1d_pw_flop_calculator};
}

} // namespace conv1d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
