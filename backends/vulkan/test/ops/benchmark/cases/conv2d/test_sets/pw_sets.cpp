// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/conv2d/conv2d.h>

#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace conv2d {

namespace {

struct Conv2dPwCaseGroup {
  std::vector<Conv2dPwConfig> configs;
  std::vector<vkapi::ScalarType> dtypes;
};

std::vector<TestCase> generate_pw_test_cases(
    const std::vector<Conv2dPwCaseGroup>& groups) {
  std::vector<TestCase> test_cases;

  for (const auto& group : groups) {
    for (const auto& config : group.configs) {
      for (auto dtype : group.dtypes) {
        test_cases.push_back(create_conv2d_pw_test_case(
            config, dtype, kStorageType, kMemoryLayout));
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("conv2d", "pw_correctness") {
  // Accuracy shapes (small enough for float reference validation)
  std::vector<Conv2dPwConfig> accuracy_configs = {
      {1, 16, 32, 8, 8, false},
      {1, 32, 16, 8, 8, false},
      {1, 16, 32, 8, 8, true},
      // Non-multiple-of-4 channels
      {1, 13, 27, 8, 8, false},
      {1, 33, 17, 8, 8, false},
  };

  const std::vector<Conv2dPwCaseGroup> groups = {
      // Accuracy test cases (float only)
      {accuracy_configs, {vkapi::kFloat}},
  };
  return {
      generate_pw_test_cases(groups),
      conv2d_pw_reference_impl,
      conv2d_pw_flop_calculator};
}

REGISTER_TEST_CASE_SET("conv2d", "pw_performance") {
  // Accuracy shapes (small enough for float reference validation) that
  // create_conv2d_pw_test_case labels PERF, since a channel count exceeds
  // kRefDimSizeLimit
  std::vector<Conv2dPwConfig> accuracy_configs = {
      {1, 48, 96, 16, 16, false},
      {1, 96, 48, 16, 16, false},
  };

  // EdgeTAM performance shapes
  std::vector<Conv2dPwConfig> perf_configs = {
      // EdgeTAM backbone stages
      {1, 48, 96, 256, 256, false},
      {1, 96, 48, 256, 256, false},
      {1, 96, 192, 128, 128, false},
      {1, 192, 96, 128, 128, false},
      {1, 192, 384, 64, 64, false},
      {1, 384, 192, 64, 64, false},
      {1, 384, 768, 32, 32, false},
      {1, 768, 384, 32, 32, false},
      // EdgeTAM FPN/Neck
      {1, 48, 256, 256, 256, false},
      {1, 256, 32, 256, 256, false},
      {1, 96, 256, 128, 128, false},
      {1, 256, 64, 128, 128, false},
  };

  const std::vector<Conv2dPwCaseGroup> groups = {
      // Accuracy test cases (float only)
      {accuracy_configs, {vkapi::kFloat}},
      // Performance test cases (float and half)
      {perf_configs, {vkapi::kFloat, vkapi::kHalf}},
  };
  return {
      generate_pw_test_cases(groups),
      conv2d_pw_reference_impl,
      conv2d_pw_flop_calculator};
}

} // namespace conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
