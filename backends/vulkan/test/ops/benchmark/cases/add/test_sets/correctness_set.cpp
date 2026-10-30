// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/add/add.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace add {

namespace {

std::vector<TestCase> generate_test_cases(
    const std::vector<std::vector<int64_t>>& size_configs,
    const std::vector<utils::StorageType>& storage_types,
    const std::vector<vkapi::ScalarType>& data_types) {
  std::vector<TestCase> test_cases;

  // Set the data generation type as a local variable
  DataGenType data_gen_type = DataGenType::ONES;

  // Generate test cases for each combination
  for (const auto& sizes : size_configs) {
    for (const auto& storage_type : storage_types) {
      for (const auto& data_type : data_types) {
        TestCase test_case;

        // Create a descriptive name for the test case
        std::string shape_str =
            shape_bracket(sizes) + "+" + shape_bracket(sizes);
        std::string storage_str = repr_str(storage_type, utils::kWidthPacked);
        std::string dtype_str = dtype_short(data_type);
        std::string test_name = make_test_label(
            "ACCU", dtype_str, dtype_str, shape_str, storage_str);
        test_case.set_name(test_name);

        // Set the operator name for the test case
        test_case.set_operator_name("etvk.add_prototype");

        // Add two input tensors with the same size, type, storage, and data
        // generation method
        ValueSpec input_a(
            sizes, data_type, storage_type, utils::kWidthPacked, data_gen_type);
        ValueSpec input_b(
            sizes, data_type, storage_type, utils::kWidthPacked, data_gen_type);

        // Add output tensor with the same size, type, and storage as inputs
        // (output uses ZEROS by default)
        ValueSpec output(
            sizes,
            data_type,
            storage_type,
            utils::kWidthPacked,
            DataGenType::ZEROS);

        test_case.add_input_spec(input_a);
        test_case.add_input_spec(input_b);
        test_case.add_output_spec(output);

        test_cases.push_back(test_case);
      }
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("add", "correctness") {
  // Define different input size configurations
  const std::vector<std::vector<int64_t>> size_configs = {
      {1, 64, 64}, // Small square
      {1, 128, 128}, // Medium square
      {1, 256, 256}, // Large square
      {1, 512, 512}, // Very large square
      {1, 1, 1024}, // Wide tensor
      {1, 1024, 1}, // Tall tensor
      {32, 32, 32}, // 3D cube
      {16, 128, 64}, // 3D rectangular
  };

  // Storage types to test
  const std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  // Data types to test
  const std::vector<vkapi::ScalarType> data_types = {
      vkapi::kFloat, vkapi::kHalf};

  return {
      generate_test_cases(size_configs, storage_types, data_types),
      add_reference_compute,
      add_flop_calculator};
}

} // namespace add
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
