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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 512;

// QDQ8TA Conv2D configuration struct for 4D tensor quantize-dequantize testing
struct QDQ8TAConv2DConfig {
  int64_t batch_size; // N dimension
  int64_t in_channels; // C dimension
  int64_t height; // H dimension
  int64_t width; // W dimension
  std::string test_case_name = "placeholder";
  std::string op_name = "qdq8ta_conv2d_input";
};

// Utility function to create a test case from a QDQ8TAConv2DConfig
TestCase create_test_case_from_config(
    const QDQ8TAConv2DConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype) {
  TestCase test_case;

  // Create a descriptive name for the test case
  std::string storage_str =
      (storage_type == utils::kTexture3D) ? "Texture3D" : "Buffer";
  std::string dtype_str = (input_dtype == vkapi::kFloat) ? "Float" : "Half";

  std::string test_name =
      config.test_case_name + "_" + storage_str + "_" + dtype_str;
  test_case.set_name(test_name);

  // Set the operator name for the test case
  std::string operator_name = "etvk." + config.op_name + ".default";
  test_case.set_operator_name(operator_name);

  // Input tensor (float) - [N, C, H, W]
  std::vector<int64_t> input_size = {
      config.batch_size, config.in_channels, config.height, config.width};
  ValueSpec input_tensor(
      input_size,
      input_dtype,
      storage_type,
      utils::kChannelsPacked, // Use channels packed for conv2d tensors
      DataGenType::RANDOM);

  float scale_val = 0.007112;
  ValueSpec scale(scale_val);

  // Generate random zero point within quantization range
  int32_t zero_point_val = -2;
  ValueSpec zero_point(zero_point_val);

  // Output tensor (float) - same shape as input [N, C, H, W]
  ValueSpec output_tensor(
      input_size,
      input_dtype,
      storage_type,
      utils::kChannelsPacked,
      DataGenType::ZEROS);

  // Add all specs to test case
  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(scale);
  test_case.add_input_spec(zero_point);
  test_case.add_output_spec(output_tensor);

  test_case.set_abs_tolerance(scale_val + 1e-4);

  return test_case;
}

// Generate easy test cases for qdq8ta_conv2d operation (for debugging)
std::vector<TestCase> generate_qdq8ta_conv2d_easy_cases() {
  std::vector<TestCase> test_cases;

  // Single simple configuration for debugging
  QDQ8TAConv2DConfig config = {
      1, // batch_size
      3, // in_channels
      4, // height
      4, // width
      "simple", // test_case_name
  };

  // Test with both storage types
  std::vector<utils::StorageType> storage_types = {utils::kTexture3D};
  std::vector<vkapi::ScalarType> float_types = {vkapi::kFloat};

  // Generate test cases for each combination
  for (const auto& storage_type : storage_types) {
    for (const auto& input_dtype : float_types) {
      test_cases.push_back(
          create_test_case_from_config(config, storage_type, input_dtype));
    }
  }

  return test_cases;
}

// Generate test cases for qdq8ta_conv2d operation
std::vector<TestCase> generate_qdq8ta_conv2d_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<QDQ8TAConv2DConfig> configs = {
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
  };

  // Test with different storage types
  std::vector<utils::StorageType> storage_types = {utils::kTexture3D};

  for (auto config : configs) {
    std::string prefix =
        (config.batch_size < kRefDimSizeLimit &&
         config.in_channels < kRefDimSizeLimit &&
         config.height < kRefDimSizeLimit && config.width < kRefDimSizeLimit)
        ? "correctness_"
        : "performance_";
    std::string generated_test_case_name = prefix +
        std::to_string(config.batch_size) + "_" +
        std::to_string(config.in_channels) + "_" +
        std::to_string(config.height) + "_" + std::to_string(config.width);

    config.test_case_name = generated_test_case_name;

    for (const auto& storage_type : storage_types) {
      test_cases.push_back(
          create_test_case_from_config(config, storage_type, vkapi::kFloat));
    }
  }

  return test_cases;
}

// Reference implementation for qdq8ta_conv2d operation
void qdq8ta_conv2d_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& scale_spec = test_case.inputs()[idx++];
  const ValueSpec& zero_point_spec = test_case.inputs()[idx++];

  // Extract output specification
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions
  auto input_sizes = input_spec.get_tensor_sizes(); // [N, C, H, W]
  int64_t N = input_sizes[0];
  int64_t C = input_sizes[1];
  int64_t H = input_sizes[2];
  int64_t W = input_sizes[3];

  // Skip for large tensors since computation time will be extremely slow
  if (N > kRefDimSizeLimit || C > kRefDimSizeLimit || H > kRefDimSizeLimit ||
      W > kRefDimSizeLimit) {
    throw std::invalid_argument(
        "One or more dimensions (N, C, H, W) exceed the allowed limit for reference implementation.");
  }

  if (input_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  // Get raw data pointers
  auto& input_data = input_spec.get_float_data();

  // Extract the randomized scale and zero point values (following
  // q8csw_conv2d.cpp pattern)
  float scale = scale_spec.get_float_value();
  int32_t zero_point = zero_point_spec.get_int_value();
  int32_t quant_min = -128;
  int32_t quant_max = 127;

  // Prepare output data
  auto& ref_data = output_spec.get_ref_float_data();
  int64_t num_elements = N * C * H * W;
  ref_data.resize(num_elements);

  // Perform quantize-dequantize operation on each element
  for (int64_t i = 0; i < num_elements; ++i) {
    float input_val = input_data[i];

    // Quantize: quantized = round(input / scale + zero_point)
    float quantized_float = std::round(input_val / scale) + zero_point;

    // Clamp to quantization range
    quantized_float = std::max(quantized_float, static_cast<float>(quant_min));
    quantized_float = std::min(quantized_float, static_cast<float>(quant_max));

    int32_t quantized_int = static_cast<int32_t>(quantized_float);

    // Dequantize: output = (quantized - zero_point) * scale
    float dequantized = (quantized_int - zero_point) * scale;

    ref_data[i] = dequantized;
  }
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "QDQ8TA Conv2D Operation Prototyping Framework" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = qdq8ta_conv2d_reference_impl;

  auto results = execute_test_cases(
      generate_qdq8ta_conv2d_test_cases, "QDQ8TAConv2D", 0, 1, ref_fn);

  return 0;
}
