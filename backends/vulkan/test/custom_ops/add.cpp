// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <iostream>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;

// Generate test cases for add operation
std::vector<TestCase> generate_add_test_cases() {
  std::vector<TestCase> test_cases;

  // Set the data generation type as a local variable
  DataGenType data_gen_type = DataGenType::ONES;

  // Define different input size configurations
  std::vector<std::vector<int64_t>> size_configs = {
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
  std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  // Data types to test
  std::vector<vkapi::ScalarType> data_types = {vkapi::kFloat, vkapi::kHalf};

  // Generate test cases for each combination
  for (const auto& sizes : size_configs) {
    for (const auto& storage_type : storage_types) {
      for (const auto& data_type : data_types) {
        TestCase test_case;

        // Create a descriptive name for the test case
        std::string size_str = "";
        for (size_t i = 0; i < sizes.size(); ++i) {
          size_str += std::to_string(sizes[i]);
          if (i < sizes.size() - 1)
            size_str += "x";
        }

        std::string storage_str =
            (storage_type == utils::kTexture3D) ? "Texture3D" : "Buffer";
        std::string dtype_str = (data_type == vkapi::kFloat) ? "Float" : "Half";

        // Add data generation type to the name for clarity
        std::string test_name =
            "Add_" + size_str + "_" + storage_str + "_" + dtype_str;
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

// Custom FLOP calculator for add operation
// Add operation performs 1 FLOP (addition) per element
int64_t add_flop_calculator(const TestCase& test_case) {
  // Calculate total elements from the first input tensor
  int64_t total_elements = 1;
  if (!test_case.empty() && test_case.num_inputs() > 0 &&
      test_case.inputs()[0].is_tensor()) {
    const auto& sizes = test_case.inputs()[0].get_tensor_sizes();
    for (int64_t size : sizes) {
      total_elements *= size;
    }
  }

  // Add operation: 1 FLOP per element (one addition)
  return total_elements;
}

// Reference implementation for add operator
void add_reference_compute(TestCase& test_case) {
  const ValueSpec& input_a = test_case.inputs().at(0);
  const ValueSpec& input_b = test_case.inputs().at(1);

  ValueSpec& output = test_case.outputs().at(0);

  if (input_a.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  // Calculate number of elements
  int64_t num_elements = input_a.numel();

  auto& input_a_data = input_a.get_float_data();
  auto& input_b_data = input_b.get_float_data();

  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    ref_data[i] = input_a_data[i] + input_b_data[i];
  }
}

int main(int argc, char* argv[]) {
  set_print_output(false); // Disable output tensor printing
  set_print_latencies(false); // Enable latency timing printing
  set_use_gpu_timestamps(true); // Enable GPU timestamps

  print_performance_header();
  std::cout << "Add Operation Prototyping Framework" << std::endl;
  print_separator();

  // Initialize Vulkan context
  try {
    api::context()->initialize_querypool();
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize Vulkan context: " << e.what()
              << std::endl;
    return 1;
  }

  // Execute test cases using the new framework with custom FLOP calculator and
  // reference compute
  auto results = execute_test_cases(
      generate_add_test_cases,
      add_flop_calculator,
      "Add",
      3,
      10,
      add_reference_compute);

  return 0;
}
