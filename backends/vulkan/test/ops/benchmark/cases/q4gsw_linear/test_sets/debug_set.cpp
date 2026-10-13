// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q4gsw_linear/q4gsw_linear.h>

#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q4gsw_linear {

namespace {

std::vector<TestCase> generate_debug_test_cases(
    const LinearConfig& config,
    const std::vector<utils::StorageType>& storage_types,
    const std::vector<vkapi::ScalarType>& input_dtypes) {
  std::vector<TestCase> test_cases;

  for (const auto& storage_type : storage_types) {
    for (const auto& input_dtype : input_dtypes) {
      test_cases.push_back(
          create_test_case_from_config(config, storage_type, input_dtype));
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q4gsw_linear", "debug") {
  // Single simple configuration for debugging
  return {
      generate_debug_test_cases(
          {
              8, // Batch size
              16, // Input features
              16, // Output features
              8, // Group size
              true, // has_bias
              "simple", // test_case_name
              "linear_dq8ca_q4gsw", // op_name
          },
          /*storage_types=*/{utils::kTexture3D, utils::kBuffer},
          /*input_dtypes=*/{vkapi::kFloat}),
      reference_impl,
      quantized_linear_flop_calculator};
}

} // namespace q4gsw_linear
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
