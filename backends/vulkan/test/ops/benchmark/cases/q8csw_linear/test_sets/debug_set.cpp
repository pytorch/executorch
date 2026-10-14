// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8csw_linear/q8csw_linear.h>

#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8csw_linear {

namespace {

std::vector<TestCase> generate_single_op_test_cases(
    const std::vector<LinearConfig>& configs,
    const std::vector<utils::StorageType>& storage_types,
    const std::vector<vkapi::ScalarType>& dtypes) {
  std::vector<TestCase> test_cases;

  for (const auto& config : configs) {
    for (const auto& storage_type : storage_types) {
      for (const auto& input_dtype : dtypes) {
        test_cases.push_back(
            create_test_case_from_config(config, storage_type, input_dtype));
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8csw_linear", "debug") {
  const std::vector<LinearConfig> configs = {
      // Single simple configuration for debugging
      {4, 4, 4, true, "simple"},
  };
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};
  const std::vector<vkapi::ScalarType> dtypes = {vkapi::kFloat};

  return {
      generate_single_op_test_cases(configs, storage_types, dtypes),
      reference_impl,
      quantized_linear_flop_calculator};
}

} // namespace q8csw_linear
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
