// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/choose_qparams_per_row/choose_qparams_per_row.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace choose_qparams_per_row {

namespace {

std::vector<TestCase> generate_test_cases(
    const std::vector<ChooseQParamsConfig>& configs,
    const std::vector<utils::StorageType>& storage_types) {
  std::vector<TestCase> test_cases;

  for (auto config : configs) {
    std::string prefix = (config.num_channels < kRefDimSizeLimit &&
                          config.channel_size < kRefDimSizeLimit)
        ? "correctness_"
        : "performance_";
    std::string generated_test_case_name = prefix +
        std::to_string(config.num_channels) + "_" +
        std::to_string(config.channel_size);

    config.test_case_name = generated_test_case_name;

    for (const auto& storage_type : storage_types) {
      test_cases.push_back(
          create_test_case_from_config(config, storage_type, vkapi::kFloat));
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("choose_qparams_per_row", "correctness") {
  const std::vector<ChooseQParamsConfig> configs = {
      {4, 16},
      {8, 32},
      {16, 64},
      {32, 128},
      {64, 256},
      {128, 512},
      {1, 512},
      {256, 1024},
      {512, 2048},
      {1, 2048},
  };
  // Test with different storage types
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};
  return {
      generate_test_cases(configs, storage_types),
      reference_impl,
      choose_qparams_per_channel_flop_calculator};
}

REGISTER_TEST_CASE_SET("choose_qparams_per_row", "debug") {
  const std::vector<ChooseQParamsConfig> configs = {
      // Single simple configuration for debugging
      {4, 8},
  };
  // Test with different storage types
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};
  return {
      generate_test_cases(configs, storage_types),
      reference_impl,
      choose_qparams_per_channel_flop_calculator};
}

REGISTER_TEST_CASE_SET("choose_qparams_per_row", "performance") {
  const std::vector<ChooseQParamsConfig> configs = {
      // Performance cases
      {1, 8096},
  };
  // Test with different storage types
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};
  return {
      generate_test_cases(configs, storage_types),
      reference_impl,
      choose_qparams_per_channel_flop_calculator};
}

} // namespace choose_qparams_per_row
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
