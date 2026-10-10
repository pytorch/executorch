// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8csw_linear/q8csw_linear.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8csw_linear {

namespace {

std::vector<TestCase> generate_test_cases(
    const std::vector<LinearConfig>& configs,
    const std::vector<utils::StorageType>& storage_types) {
  std::vector<TestCase> test_cases;

  for (auto config : configs) {
    std::string prefix =
        (config.M < kRefDimSizeLimit && config.K < kRefDimSizeLimit &&
         config.N < kRefDimSizeLimit)
        ? "correctness_"
        : "performance_";
    std::string generated_test_case_name = prefix + std::to_string(config.M) +
        "_" + std::to_string(config.K) + "_" + std::to_string(config.N);
    if (!config.has_bias) {
      generated_test_case_name += "_no_bias";
    }

    config.test_case_name = generated_test_case_name;

    for (const auto& storage_type : storage_types) {
      if (vkcompute::api::context()
              ->adapter_ptr()
              ->supports_int8_dot_product()) {
        // Test both activation+weight quantized and weight only quantized
        test_cases.push_back(
            create_test_case_from_config(config, storage_type, vkapi::kFloat));
      }

      LinearConfig wo_quant_config = config;
      wo_quant_config.op_name = "linear_q8csw";
      test_cases.push_back(create_test_case_from_config(
          wo_quant_config, storage_type, vkapi::kFloat));
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8csw_linear", "correctness") {
  const std::vector<LinearConfig> configs = {
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
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  return {
      generate_test_cases(configs, storage_types),
      reference_impl,
      quantized_linear_flop_calculator};
}

REGISTER_TEST_CASE_SET("q8csw_linear", "performance") {
  const std::vector<LinearConfig> configs = {
      {256, 2048, 2048},
      {512, 2048, 2048},
      {1024, 2048, 2048},
  };
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  return {
      generate_test_cases(configs, storage_types),
      reference_impl,
      quantized_linear_flop_calculator};
}

} // namespace q8csw_linear
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
