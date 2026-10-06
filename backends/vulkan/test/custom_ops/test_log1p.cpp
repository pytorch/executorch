// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "utils.h"

using namespace executorch::vulkan::prototyping;

// Generate test cases for log1p operation
std::vector<TestCase> generate_log1p_test_cases() {
  std::vector<TestCase> test_cases;

  // Set the data generation type
  DataGenType data_gen_type = DataGenType::RANDOM;

  // Define different input size configurations
  std::vector<std::vector<int64_t>> size_configs = {
      {1, 64, 64}, 
      {1, 128, 128}, 
      {1, 256, 256}, 
      {32, 32, 32},
  };

  std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  std::vector<vkapi::ScalarType> data_types = {vkapi::kFloat, vkapi::kHalf};

  for (const auto& sizes : size_configs) {
    for (const auto& storage_type : storage_types) {
      for (const auto& data_type : data_types) {
        TestCase test_case;

        std::string shape_str = shape_bracket(sizes);
        std::string storage_str = repr_str(storage_type, utils::kWidthPacked);
        std::string dtype_str = dtype_short(data_type);
        std::string test_name = make_test_label(
            "LOG1P", dtype_str, dtype_str, shape_str, storage_str);
        test_case.set_name(test_name);

        test_case.set_operator_name("aten.log1p.default");

        ValueSpec input(
            sizes, data_type, storage_type, utils::kWidthPacked, data_gen_type);

        ValueSpec output(
            sizes,
            data_type,
            storage_type,
            utils::kWidthPacked,
            DataGenType::ZEROS);

        test_case.add_input_spec(input);
        test_case.add_output_spec(output);

        test_cases.push_back(test_case);
      }
    }
  }

  return test_cases;
}

int64_t log1p_flop_calculator(const TestCase& test_case) {
  int64_t total_elements = test_case.inputs()[0].numel();
  return total_elements; // 1 log1p per element
}

void log1p_reference_compute(TestCase& test_case) {
  const ValueSpec& input = test_case.inputs().at(0);
  ValueSpec& output = test_case.outputs().at(0);

  if (input.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype for reference compute");
  }

  int64_t num_elements = input.numel();

  auto& in_data = input.get_float_data();
  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(num_elements);
  
  for (int64_t i = 0; i < num_elements; ++i) {
    // ensure x > -1 to avoid NaN if random data is used
    float x = in_data[i];
    if (x <= -1.0f) {
      x = std::abs(x);
    }
    ref_data[i] = std::log1p(x);
  }
}

int main(int argc, char* argv[]) {
  set_print_output(false); 
  set_print_latencies(false); 
  set_use_gpu_timestamps(true); 

  print_performance_header();
  std::cout << "Log1p Operation Prototyping Framework" << std::endl;
  print_separator();

  try {
    api::context()->initialize_querypool();
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize Vulkan context: " << e.what() << std::endl;
    return 1;
  }

  auto results = execute_test_cases(
      generate_log1p_test_cases,
      log1p_flop_calculator,
      "Log1p",
      /*warmup_runs = */ 1,
      /*benchmark_runs = */ 1,
      log1p_reference_compute);

  return 0;
}
