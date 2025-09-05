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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

using namespace executorch::vulkan::prototyping;

using namespace vkcompute;

// Linear configuration struct
struct LinearConfig {
  int64_t M; // Batch size / number of rows in input
  int64_t K; // Input features / columns in input, rows in weight
  int64_t N; // Output features / columns in weight
  int64_t group_size; // Number of input channels per quantization group
  std::string name_suffix;
  std::string shader_variant_name = "default";
};

// Utility function to create a test case from a LinearConfig
TestCase create_test_case_from_config(
    const LinearConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype) {
  TestCase test_case;

  // Create a descriptive name for the test case
  std::string storage_str =
      (storage_type == utils::kTexture3D) ? "Texture3D" : "Buffer";
  std::string dtype_str = (input_dtype == vkapi::kFloat) ? "Float" : "Half";

  std::string test_name = "QuantizedLinearInt4_" + config.name_suffix + "_" +
      storage_str + "_" + dtype_str;
  test_case.set_name(test_name);

  // Set the operator name for the test case
  std::string operator_name = "et_vk.linear_weight_int4.default";
  test_case.set_operator_name(operator_name);

  // Derive sizes from M, K, N
  std::vector<int64_t> input_size = {config.M, config.K};
  std::vector<int64_t> weight_size = {
      config.N, config.K / 2}; // Packed 4-bit weights

  // Input tensor (float/half) - [M, K]
  ValueSpec input_tensor(
      input_size,
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ONES);

  if (debugging()) {
    print_valuespec_data(input_tensor, "input_tensor");
  }

  // Quantized weight tensor (int8, packed 4-bit) - [N, K/2]
  ValueSpec quantized_weight(
      weight_size,
      vkapi::kChar, // int8 for packed 4-bit quantized weights
      storage_type,
      utils::kWidthPacked,
      DataGenType::ONES);
  quantized_weight.set_constant(true);
  quantized_weight.set_int4(true);

  if (debugging()) {
    print_valuespec_data(quantized_weight, "weight_tensor");
  }

  // Group size parameter
  ValueSpec group_size_spec(static_cast<int32_t>(config.group_size));

  // Weight quantization scales and zeros (float/half, per-group) -
  // [K/group_size, N, 2]
  std::vector<int64_t> scales_and_zeros_size = {
      config.K / config.group_size, config.N, 2};
  ValueSpec scales_and_zeros(
      scales_and_zeros_size,
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ONES);
  scales_and_zeros.set_constant(true);

  if (debugging()) {
    print_valuespec_data(scales_and_zeros, "scales_and_zeros");
  }

  // Output tensor (float/half) - [M, N]
  ValueSpec output(
      {config.M, config.N},
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);

  // Add all specs to test case
  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(quantized_weight);
  test_case.add_input_spec(group_size_spec);
  test_case.add_input_spec(scales_and_zeros);
  // Add dummy value for inner_k_tiles (unused but required by operator
  // signature)
  ValueSpec dummy_inner_k_tiles(static_cast<int32_t>(8));
  test_case.add_input_spec(dummy_inner_k_tiles);

  test_case.add_output_spec(output);

  return test_case;
}

// Generate easy test cases for quantized linear operation (for debugging)
std::vector<TestCase> generate_quantized_linear_easy_cases() {
  std::vector<TestCase> test_cases;

  // Single simple configuration for debugging
  int M = 8;
  int K = 16;
  int N = 16;
  int group_size = 8;

  LinearConfig config = {
      M, // Batch size
      K, // Input features
      N, // Output features
      group_size, // Group size
      "simple", // descriptive name
      "default" // shader variant name
  };

  // Test with both storage types and data types for completeness
  std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};
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

// Generate test cases for quantized linear operation
std::vector<TestCase> generate_quantized_linear_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<LinearConfig> configs = {
      {8, 64, 32, 8, "correctness_8_64_32_g8"},
      {8, 128, 64, 16, "correctness_8_128_64_g16"},
      {8, 256, 128, 32, "correctness_8_256_128_g32"},
      {32, 64, 32, 8, "correctness_32_64_32_g8"},
      {32, 128, 64, 16, "correctness_32_128_64_g16"},
      {32, 256, 128, 32, "correctness_32_256_128_g32"},
      {1, 256, 128, 32, "correctness_32_256_128_g32"},
      // Performance test cases
      {1, 2048, 2048, 128, "performance_128_2048_2048_g128"},
      {128, 2048, 2048, 128, "performance_128_2048_2048_g128"},
      {248, 2048, 2048, 128, "performance_128_2048_2048_g128"},
      {1024, 2048, 2048, 128, "performance_128_2048_2048_g128"},
      // {16384, 576, 128, 32, "performance_16384_576_128_g32"}
  };

  // Test with different storage types and data types
  std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  // Generate test cases for each combination
  for (const auto& config : configs) {
    for (const auto& storage_type : storage_types) {
      test_cases.push_back(
          create_test_case_from_config(config, storage_type, vkapi::kFloat));
    }
  }

  return test_cases;
}

// Helper function to unpack 4-bit values from int8
std::pair<int8_t, int8_t> unpack_4bit(int8_t packed) {
  // Extract lower 4 bits and upper 4 bits
  int8_t lower = packed & 0x0F;
  int8_t upper = (packed >> 4) & 0x0F;

  // Sign extend from 4-bit to 8-bit
  if (lower & 0x08)
    lower |= 0xF0;
  if (upper & 0x08)
    upper |= 0xF0;

  return std::make_pair(lower, upper);
}

// Reference implementation for quantized linear operation
void quantized_linear_reference_impl(TestCase& test_case) {
  static constexpr int64_t kRefDimSizeLimit = 300;
  // Extract input specifications
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_spec = test_case.inputs()[idx++];
  const ValueSpec& group_size_spec = test_case.inputs()[idx++];
  const ValueSpec& scales_and_zeros_spec = test_case.inputs()[idx++];
  // Skip dummy inner_k_tiles
  idx++;

  // Extract output specification (mutable reference)
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions
  auto input_sizes = input_spec.get_tensor_sizes(); // [batch_size, in_features]
  auto weight_sizes =
      weight_spec.get_tensor_sizes(); // [out_features, in_features/2]
  auto output_sizes =
      output_spec.get_tensor_sizes(); // [batch_size, out_features]

  int64_t batch_size = input_sizes[0];
  int64_t in_features = input_sizes[1];
  int64_t out_features = output_sizes[1];
  int64_t group_size = group_size_spec.get_int_value();

  // Skip for large tensors since computation time will be extremely slow
  if (batch_size > kRefDimSizeLimit || in_features > kRefDimSizeLimit ||
      out_features > kRefDimSizeLimit) {
    throw std::invalid_argument(
        "One or more dimensions (batch_size, in_features, out_features) exceed the allowed limit for reference implementation.");
  }

  if (input_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  // Get raw data pointers
  auto& input_data = input_spec.get_float_data();
  auto& weight_data = weight_spec.get_int8_data();
  auto& scales_and_zeros_data = scales_and_zeros_spec.get_float_data();

  // Calculate number of output elements
  int64_t num_output_elements = batch_size * out_features;

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_output_elements);

  // Perform quantized linear transformation (matrix multiplication)
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t out_f = 0; out_f < out_features; ++out_f) {
      float sum = 0.0f;

      bool should_print = b == 0 && out_f == 0;
      should_print = false;

      if (should_print) {
        std::cout << "Weights seen: ";
      }

      // Matrix multiplication: output[b][out_f] = sum(input[b][in_f] *
      // weight[out_f][in_f])
      for (int64_t in_f = 0; in_f < in_features; ++in_f) {
        // Get input value
        int64_t input_idx = b * in_features + in_f;
        float input_val = input_data[input_idx];

        // Get weight value and dequantize (4-bit group affine quantization)
        int64_t group_idx = in_f / group_size;
        int64_t scales_and_zeros_idx = group_idx * out_features * 2 + out_f * 2;

        // Get packed weight value
        int64_t weight_idx = out_f * (in_features / 2) + (in_f / 2);
        int8_t packed_weight = weight_data[weight_idx];

        // Unpack 4-bit weight
        auto unpacked = unpack_4bit(packed_weight);
        int8_t weight_4bit = (in_f % 2 == 0) ? unpacked.first : unpacked.second;

        // Dequantize weight using group affine quantization
        float weight_scale = scales_and_zeros_data[scales_and_zeros_idx];
        float weight_zero = scales_and_zeros_data[scales_and_zeros_idx + 1];
        float dequant_weight =
            (static_cast<float>(weight_4bit) - 8.0f) * weight_scale +
            weight_zero;

        if (should_print) {
          std::cout << int(weight_4bit) << ", ";
        }

        sum += input_val * dequant_weight;
      }

      if (should_print) {
        std::cout << std::endl;
      }

      // Store result
      int64_t output_idx = b * out_features + out_f;
      ref_data[output_idx] = sum;
    }
  }
}

// Custom FLOP calculator for quantized linear operation
int64_t quantized_linear_flop_calculator(const TestCase& test_case) {
  if (test_case.num_inputs() < 4 || test_case.num_outputs() < 1) {
    return 0;
  }

  // Get input and weight dimensions
  const auto& input_sizes = test_case.inputs()[0].get_tensor_sizes();
  const auto& output_sizes = test_case.outputs()[0].get_tensor_sizes();

  int64_t batch_size = input_sizes[0];
  int64_t in_features = input_sizes[1];
  int64_t out_features = output_sizes[1];

  // Calculate FLOPs for quantized linear operation
  // Each output element requires:
  // - in_features multiply-accumulate operations
  // - Additional operations for quantization/dequantization
  int64_t output_elements = batch_size * out_features;
  int64_t ops_per_output = in_features;

  // Add quantization overhead (approximate)
  // - Dequantize weight: 2 ops per weight element used (unpack + dequantize)
  int64_t quantization_ops = ops_per_output * 2; // Simplified estimate

  int64_t flop = output_elements * (ops_per_output + quantization_ops);

  return flop;
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Quantized 4-bit Int4 Linear Operation Prototyping Framework"
            << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = quantized_linear_reference_impl;

  // Execute easy test cases using the new framework with custom FLOP
  // calculator
  auto results = execute_test_cases(
      generate_quantized_linear_test_cases,
      quantized_linear_flop_calculator,
      "QuantizedLinearInt4",
      0,
      10,
      ref_fn);

  return 0;
}
