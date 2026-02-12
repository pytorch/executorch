// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;

static constexpr int64_t kRefDimSizeLimit = 512;

// Configuration struct for q8ta binary testing
struct Q8taBinaryConfig {
  std::vector<int64_t> shape; // Tensor shape (can be any dimensionality)
  std::string test_case_name = "placeholder";
  std::string op_name = "q8ta_add";
};

// Utility function to create a test case from a Q8taBinaryConfig
TestCase create_test_case_from_config(
    const Q8taBinaryConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype,
    utils::GPUMemoryLayout fp_memory_layout,
    utils::GPUMemoryLayout quant_layout,
    bool const_b = false) {
  TestCase test_case;

  // Create a descriptive name for the test case
  std::string shape_str = shape_string(config.shape);
  std::string test_name = config.test_case_name + "  I=" + shape_str + "  " +
      repr_str(utils::kBuffer, quant_layout);
  if (const_b) {
    test_name += "  const_b";
  }
  test_case.set_name(test_name);

  // Set the operator name for the test case
  std::string operator_name = "et_vk." + config.op_name + ".test";
  test_case.set_operator_name(operator_name);

  // Input tensor A (float/half)
  ValueSpec input_a(
      config.shape,
      input_dtype,
      storage_type,
      fp_memory_layout,
      DataGenType::RANDOM);

  // Input tensor B (float/half, or pre-quantized int8 for const_b)
  ValueSpec input_b(
      config.shape,
      const_b ? vkapi::kChar : input_dtype,
      storage_type,
      fp_memory_layout,
      const_b ? DataGenType::RANDINT8 : DataGenType::RANDOM);
  if (const_b) {
    input_b.set_constant(true);
  }

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

  // Quantized layout as integer
  int32_t quant_layout_int = static_cast<int32_t>(quant_layout);
  ValueSpec quant_layout_spec(quant_layout_int);

  // Output tensor (float/half)
  ValueSpec output(
      config.shape,
      input_dtype,
      storage_type,
      fp_memory_layout,
      DataGenType::ZEROS);

  // Add all specs to test case for q8ta add operation
  test_case.add_input_spec(input_a);
  test_case.add_input_spec(input_b);
  test_case.add_input_spec(input_a_scale);
  test_case.add_input_spec(input_a_zero_point);
  test_case.add_input_spec(input_b_scale);
  test_case.add_input_spec(input_b_zero_point);
  test_case.add_input_spec(output_scale);
  test_case.add_input_spec(output_zero_point);
  test_case.add_input_spec(alpha);
  test_case.add_input_spec(quant_layout_spec);

  test_case.add_output_spec(output);

  test_case.set_abs_tolerance(output_scale_val + 1e-4f);

  // Use layout-only filter to focus on the binary operation
  test_case.set_shader_filter({
      "nchw_to",
      "to_nchw",
      "q8ta_quantize",
      "q8ta_dequantize",
  });

  return test_case;
}

// Generate easy test cases for q8ta_add operation (for debugging)
std::vector<TestCase> generate_q8ta_add_easy_cases() {
  std::vector<TestCase> test_cases;

  // Single simple configuration for debugging
  Q8taBinaryConfig config = {
      {1, 16, 16, 16}, // shape: [N, C, H, W]
      "ACCU", // test_case_name
  };

  // Quantized memory layouts to test
  std::vector<utils::GPUMemoryLayout> quant_layouts = {
      utils::kPackedInt8_4W,
      utils::kPackedInt8_4C,
      utils::kPackedInt8_4W4C,
      utils::kPackedInt8_4H4W,
      utils::kPackedInt8_4C1W,
  };

  for (const auto& quant_layout : quant_layouts) {
    test_cases.push_back(create_test_case_from_config(
        config,
        /*fp_storage_type=*/utils::kBuffer,
        /*input_dtype=*/vkapi::kFloat,
        /*fp_layout=*/utils::kWidthPacked,
        quant_layout));
    test_cases.push_back(create_test_case_from_config(
        config,
        /*fp_storage_type=*/utils::kBuffer,
        /*input_dtype=*/vkapi::kFloat,
        /*fp_layout=*/utils::kWidthPacked,
        quant_layout,
        /*const_b=*/true));
  }

  return test_cases;
}

// Generate test cases for q8ta_add operation
std::vector<TestCase> generate_q8ta_add_test_cases() {
  std::vector<TestCase> test_cases;

  // Shapes to test
  std::vector<std::vector<int64_t>> shapes = {
      // Small test cases for correctness
      {1, 3, 16, 16},
      {1, 8, 32, 32},
      {1, 16, 24, 24},
      {1, 32, 12, 12},
      {1, 1, 64, 64},
      {1, 3, 64, 64},
      {1, 4, 16, 16},

      // Different tensor sizes
      {1, 8, 20, 20},
      {1, 16, 14, 14},
      {1, 8, 28, 28},

      // Odd tensor sizes
      {1, 3, 15, 15},
      {1, 13, 31, 31},
      {1, 17, 23, 23},

      // Performance test cases (larger tensors)
      {1, 64, 128, 128},
      {1, 32, 64, 64},
      {1, 128, 56, 56},
      {1, 128, 128, 128},
  };

  // Quantized memory layouts to test
  std::vector<utils::GPUMemoryLayout> quant_layouts = {
      utils::kPackedInt8_4W,
      utils::kPackedInt8_4C,
      utils::kPackedInt8_4W4C,
      utils::kPackedInt8_4H4W,
      utils::kPackedInt8_4C1W,
  };

  // Generate all combinations
  for (const auto& shape : shapes) {
    // Generate test case name prefix from shape dimensions
    std::string prefix = "ACCU";
    for (const auto& dim : shape) {
      if (dim > kRefDimSizeLimit) {
        prefix = "PERF";
        break;
      }
    }

    Q8taBinaryConfig config;
    config.shape = shape;
    config.test_case_name = prefix;
    for (const auto& quant_layout : quant_layouts) {
      test_cases.push_back(create_test_case_from_config(
          config,
          /*fp_storage_type=*/utils::kBuffer,
          /*fp_input_dtype=*/vkapi::kFloat,
          /*fp_layout=*/utils::kWidthPacked,
          quant_layout));
      test_cases.push_back(create_test_case_from_config(
          config,
          /*fp_storage_type=*/utils::kBuffer,
          /*fp_input_dtype=*/vkapi::kFloat,
          /*fp_layout=*/utils::kWidthPacked,
          quant_layout,
          /*const_b=*/true));
    }
  }

  return test_cases;
}

// Reference implementation for quantized add operation
void q8ta_add_reference_impl(TestCase& test_case) {
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
  const ValueSpec& quant_layout_spec = test_case.inputs()[idx++];
  (void)quant_layout_spec; // Not used in reference implementation

  // Extract output specification
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions
  auto input_sizes = input_a_spec.get_tensor_sizes();

  // Calculate total number of elements
  int64_t num_elements = 1;
  for (const auto& dim : input_sizes) {
    num_elements *= dim;
  }

  // Skip for large tensors since computation time will be extremely slow
  for (const auto& dim : input_sizes) {
    if (dim > kRefDimSizeLimit) {
      throw std::invalid_argument(
          "One or more dimensions exceed the allowed limit for reference "
          "implementation.");
    }
  }

  if (input_a_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  bool input_b_is_int8 = (input_b_spec.dtype == vkapi::kChar);

  // Get raw data pointers
  auto& input_a_data = input_a_spec.get_float_data();

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

    // Get quantized input B (either from pre-quantized int8 or by quantizing)
    int8_t quantized_b;
    if (input_b_is_int8) {
      quantized_b = input_b_spec.get_int8_data()[i];
    } else {
      float quant_b_f =
          std::round(input_b_spec.get_float_data()[i] / input_b_scale) +
          input_b_zero_point;
      quant_b_f = std::min(std::max(quant_b_f, -128.0f), 127.0f);
      quantized_b = static_cast<int8_t>(quant_b_f);
    }

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

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
#ifdef DEBUG_MODE
  set_print_latencies(false);
#else
  set_print_latencies(false);
#endif
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Q8TA Binary Add Operation Prototyping Framework" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = q8ta_add_reference_impl;

  auto results = execute_test_cases(
#ifdef DEBUG_MODE
      generate_q8ta_add_easy_cases,
#else
      generate_q8ta_add_test_cases,
#endif
      "Q8taBinaryAdd",
#ifdef DEBUG_MODE
      0,
      1,
#else
      3,
      10,
#endif
      ref_fn);

  return 0;
}
