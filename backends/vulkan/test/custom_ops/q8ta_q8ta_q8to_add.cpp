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

// Utility function to create a test case for quantized add operation
TestCase create_quantized_add_test_case(
    const std::vector<int64_t>& sizes,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype) {
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
  std::string dtype_str = (input_dtype == vkapi::kFloat) ? "Float" : "Half";

  std::string test_name =
      "QuantizedAdd_" + size_str + "_" + storage_str + "_" + dtype_str;
  test_case.set_name(test_name);

  // Set the operator name for the test case
  test_case.set_operator_name("et_vk.add_q8ta_q8ta_q8to.test");

  utils::GPUMemoryLayout io_memory_layout = storage_type == utils::kBuffer
      ? utils::kWidthPacked
      : utils::kChannelsPacked;

  // Input tensor A (float/half)
  ValueSpec input_a(
      sizes, input_dtype, storage_type, io_memory_layout, DataGenType::RANDOM);

  // Input tensor B (float/half)
  ValueSpec input_b(
      sizes, input_dtype, storage_type, io_memory_layout, DataGenType::RANDOM);

  // Quantization parameters for input A
  float input_a_scale_val = 0.007843; // 2/255 approximately
  ValueSpec input_a_scale(input_a_scale_val);

  int32_t input_a_zero_point_val = 3;
  ValueSpec input_a_zero_point(input_a_zero_point_val);

  // Quantization parameters for input B
  float input_b_scale_val = 0.009412; // 2.4/255 approximately
  ValueSpec input_b_scale(input_b_scale_val);

  int32_t input_b_zero_point_val = -2;
  ValueSpec input_b_zero_point(input_b_zero_point_val);

  // Output quantization parameters
  float output_scale_val = 0.015686; // 4/255 approximately
  ValueSpec output_scale(output_scale_val);

  int32_t output_zero_point_val = 1;
  ValueSpec output_zero_point(output_zero_point_val);

  // Alpha parameter
  float alpha_val = 1.0f;
  ValueSpec alpha(alpha_val);

  // Output tensor (float/half)
  ValueSpec output(
      sizes, input_dtype, storage_type, io_memory_layout, DataGenType::ZEROS);

  // Add all specs to test case for q8ta_q8ta_q8to add operation
  test_case.add_input_spec(input_a);
  test_case.add_input_spec(input_b);
  test_case.add_input_spec(input_a_scale);
  test_case.add_input_spec(input_a_zero_point);
  test_case.add_input_spec(input_b_scale);
  test_case.add_input_spec(input_b_zero_point);
  test_case.add_input_spec(output_scale);
  test_case.add_input_spec(output_zero_point);
  test_case.add_input_spec(alpha);

  test_case.add_output_spec(output);

  test_case.set_abs_tolerance(output_scale_val + 1e-4f);

  return test_case;
}

// Generate test cases for quantized add operation
std::vector<TestCase> generate_quantized_add_test_cases() {
  std::vector<TestCase> test_cases;

  // Define different input size configurations
  std::vector<std::vector<int64_t>> size_configs = {
      {3, 32, 32}, // Small square
      {8, 64, 64}, // Medium square
      {16, 16, 16}, // 3D cube
      {8, 32, 16}, // 3D rectangular
      {7, 7, 13}, // Irregular sizes
  };

  // Storage types to test
  std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  // Data types to test
  std::vector<vkapi::ScalarType> data_types = {vkapi::kFloat};

  // Generate test cases for each combination
  for (const auto& sizes : size_configs) {
    for (const auto& storage_type : storage_types) {
      for (const auto& data_type : data_types) {
        test_cases.push_back(
            create_quantized_add_test_case(sizes, storage_type, data_type));
      }
    }
  }

  return test_cases;
}

// Reference implementation for quantized add operation
void add_q8ta_q8ta_q8to_reference_impl(TestCase& test_case) {
  // Extract input specifications
  int32_t idx = 0;
  const ValueSpec& input_a_spec = test_case.inputs()[idx++];
  const ValueSpec& input_b_spec = test_case.inputs()[idx++];
  const ValueSpec& input_a_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& input_a_zero_point_spec = test_case.inputs()[idx++];
  const ValueSpec& input_b_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& input_b_zero_point_spec = test_case.inputs()[idx++];
  const ValueSpec& output_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& output_zero_point_spec = test_case.inputs()[idx++];
  const ValueSpec& alpha_spec = test_case.inputs()[idx++];

  // Extract output specification (mutable reference)
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions
  auto input_sizes = input_a_spec.get_tensor_sizes();
  int64_t num_elements = input_a_spec.numel();

  if (input_a_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  // Get raw data pointers
  auto& input_a_data = input_a_spec.get_float_data();
  auto& input_b_data = input_b_spec.get_float_data();

  const float input_a_scale = input_a_scale_spec.get_float_value();
  const int32_t input_a_zero_point = input_a_zero_point_spec.get_int_value();
  const float input_b_scale = input_b_scale_spec.get_float_value();
  const int32_t input_b_zero_point = input_b_zero_point_spec.get_int_value();
  const float output_scale = output_scale_spec.get_float_value();
  const int32_t output_zero_point = output_zero_point_spec.get_int_value();
  const float alpha = alpha_spec.get_float_value();

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_elements);

  // Perform quantized add operation
  for (int64_t i = 0; i < num_elements; ++i) {
    // Quantize input A to int8
    float quant_a_f =
        std::round(input_a_data[i] / input_a_scale) + input_a_zero_point;
    quant_a_f = std::min(std::max(quant_a_f, -128.0f), 127.0f);
    int8_t quantized_a = static_cast<int8_t>(quant_a_f);

    // Quantize input B to int8
    float quant_b_f =
        std::round(input_b_data[i] / input_b_scale) + input_b_zero_point;
    quant_b_f = std::min(std::max(quant_b_f, -128.0f), 127.0f);
    int8_t quantized_b = static_cast<int8_t>(quant_b_f);

    // Dequantize both inputs to a common scale for addition
    float dequant_a =
        (static_cast<float>(quantized_a) - input_a_zero_point) * input_a_scale;
    float dequant_b =
        (static_cast<float>(quantized_b) - input_b_zero_point) * input_b_scale;

    // Perform addition in float space with alpha
    float float_result = dequant_a + alpha * dequant_b;

    // Quantize the result to int8
    float quant_output_f =
        std::round(float_result / output_scale) + output_zero_point;
    quant_output_f = std::min(std::max(quant_output_f, -128.0f), 127.0f);
    int8_t quantized_output = static_cast<int8_t>(quant_output_f);

    // Dequantize back to float for comparison
    float dequant_output =
        (static_cast<float>(quantized_output) - output_zero_point) *
        output_scale;

    ref_data[i] = dequant_output;
  }
}

void reference_impl(TestCase& test_case) {
  add_q8ta_q8ta_q8to_reference_impl(test_case);
}

// Custom FLOP calculator for quantized add operation
int64_t quantized_add_flop_calculator(const TestCase& test_case) {
  // Calculate total elements from the first input tensor
  int64_t total_elements = 1;
  if (!test_case.empty() && test_case.num_inputs() > 0 &&
      test_case.inputs()[0].is_tensor()) {
    const auto& sizes = test_case.inputs()[0].get_tensor_sizes();
    for (int64_t size : sizes) {
      total_elements *= size;
    }
  }

  // Quantized add operation includes:
  // - 2 quantizations (float to int8)
  // - 2 dequantizations (int8 to float)
  // - 1 addition
  // For simplicity, we count this as 1 FLOP per element (the addition)
  return total_elements;
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Quantized Add Operation (q8ta_q8ta_q8to) Prototyping Framework"
            << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  // Execute test cases using the new framework with custom FLOP calculator
  auto results = execute_test_cases(
      generate_quantized_add_test_cases,
      quantized_add_flop_calculator,
      "QuantizedAddQ8taQ8taQ8to",
      0,
      1,
      ref_fn);

  return 0;
}
