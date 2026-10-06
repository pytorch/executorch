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

// Generate test cases for mul.Scalar operation
std::vector<TestCase> generate_mul_scalar_test_cases() {
  std::vector<TestCase> test_cases;

  // Set the data generation type as a local variable
  DataGenType data_gen_type = DataGenType::ONES;

  // Define different input size configurations
  std::vector<std::vector<int64_t>> size_configs = {
      {1, 64, 64}, // Small square
      {1, 128, 128}, // Medium square
      {1, 256, 256}, // Large square
      {32, 32, 32}, // 3D cube
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
        std::string shape_str = shape_bracket(sizes) + "*Scalar";
        std::string storage_str = repr_str(storage_type, utils::kWidthPacked);
        std::string dtype_str = dtype_short(data_type);
        std::string test_name = make_test_label(
            "MULS", dtype_str, dtype_str, shape_str, storage_str);
        test_case.set_name(test_name);

        // Set the operator name for the test case
        test_case.set_operator_name("aten.mul.Scalar");

        // Add tensor input
        ValueSpec input_a(
            sizes, data_type, storage_type, utils::kWidthPacked, data_gen_type);
            
        // Add scalar input (e.g. 3.0f)
        ValueSpec input_b(3.0f);

        // Add output tensor with the same size, type, and storage as input_a
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

// Custom FLOP calculator for mul.Scalar operation
int64_t mul_scalar_flop_calculator(const TestCase& test_case) {
  int64_t total_elements = 1;
  if (!test_case.empty() && test_case.num_inputs() > 0 &&
      test_case.inputs()[0].is_tensor()) {
    const auto& sizes = test_case.inputs()[0].get_tensor_sizes();
    for (int64_t size : sizes) {
      total_elements *= size;
    }
  }
  return total_elements;
}

// Reference implementation for mul.Scalar operator
void mul_scalar_reference_compute(TestCase& test_case) {
  const ValueSpec& input_a = test_case.inputs().at(0);
  const ValueSpec& input_b = test_case.inputs().at(1);
  ValueSpec& output = test_case.outputs().at(0);

  if (input_a.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  int64_t num_elements = input_a.numel();
  auto& input_a_data = input_a.get_float_data();
  float scalar_val = input_b.get_float_data()[0];

  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    ref_data[i] = input_a_data[i] * scalar_val;
  }
}

int main(int argc, char* argv[]) {
  set_print_output(false); // Disable output tensor printing
  set_print_latencies(false); // Enable latency timing printing
  set_use_gpu_timestamps(true); // Enable GPU timestamps

  print_performance_header();
  std::cout << "Mul.Scalar Operation Prototyping Framework" << std::endl;
  print_separator();

  // Initialize Vulkan context
  try {
    api::context()->initialize_querypool();
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize Vulkan context: " << e.what()
              << std::endl;
    return 1;
  }

  auto results = execute_test_cases(
      generate_mul_scalar_test_cases,
      mul_scalar_flop_calculator,
      "Mul.Scalar",
      /*warmup_runs = */ 1,
      /*benchmark_runs = */ 1,
      mul_scalar_reference_compute);

  return 0;
}
