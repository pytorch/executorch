// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/linear/linear.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_linear {

namespace {

std::vector<TestCase> generate_test_cases(
    const std::vector<LinearConfig>& configs) {
  std::vector<TestCase> test_cases;

  for (auto config : configs) {
    bool is_performance = config.M >= kRefDimSizeLimit ||
        config.K >= kRefDimSizeLimit || config.N >= kRefDimSizeLimit;

    std::string prefix = is_performance ? "performance_" : "correctness_";
    std::string generated_test_case_name = prefix + std::to_string(config.M) +
        "_" + std::to_string(config.K) + "_" + std::to_string(config.N);
    if (!config.has_bias) {
      generated_test_case_name += "_no_bias";
    }

    config.test_case_name = generated_test_case_name;

    // Default (tiled) variant
    test_cases.push_back(create_test_case_from_config(config, vkapi::kFloat));

    // For batch size 1, also test the gemv variant
    if (config.M == 1) {
      test_cases.push_back(
          create_test_case_from_config(config, vkapi::kFloat, "gemv"));
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/linear", "correctness") {
  const std::vector<LinearConfig> configs = {
      // Batch size 1 cases (test both tiled and gemv)
      {1, 64, 32},
      {1, 128, 64},
      {1, 256, 128},
      {1, 128, 64, false},
      // Multi-batch cases
      {4, 64, 32},
      {4, 128, 64},
      {4, 256, 128},
      {32, 64, 32},
      {32, 128, 64},
      {32, 256, 128},
      // No bias tests
      {32, 128, 64, false},
      {32, 256, 128, false},
  };

  return {
      generate_test_cases(configs),
      reference_impl,
      q8ta_linear_flop_calculator};
}

REGISTER_TEST_CASE_SET("q8ta/linear", "performance") {
  const std::vector<LinearConfig> configs = {
      // Performance cases
      {1, 512, 512},
      {1, 2048, 2048},
      {1, 512, 9059},
      {256, 2048, 2048},
      {512, 2048, 2048},
      {1024, 2048, 2048},
  };

  return {
      generate_test_cases(configs),
      reference_impl,
      q8ta_linear_flop_calculator};
}

} // namespace q8ta_linear
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
