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

static constexpr int64_t kRefDimSizeLimit = 300;

// Linear configuration struct
struct LinearConfig {
  int64_t M; // Batch size / number of rows in input
  int64_t K; // Input features / columns in input, rows in weight
  int64_t N; // Output features / columns in weight
  int64_t group_size; // Number of input channels per quantization group
  bool has_bias = false;
  std::string test_case_name = "placeholder";
  std::string op_name = "linear_dq8ca_q4gsw";
};

// Helper function to unpack 4-bit values from uint8
std::pair<int8_t, int8_t> unpack_4bit(uint8_t packed) {
  // Extract lower 4 bits and upper 4 bits
  int8_t lower = packed & 0x0F;
  int8_t upper = (packed >> 4) & 0x0F;

  // Subtract 8 from unpacked 4-bit values
  lower -= 8;
  upper -= 8;

  return std::make_pair(lower, upper);
}

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

  std::string test_name =
      config.test_case_name + "_" + storage_str + "_" + dtype_str;
  test_case.set_name(test_name);

  // Set the operator name for the test case
  std::string operator_name = "et_vk." + config.op_name + ".default";
  test_case.set_operator_name(operator_name);

  // Derive sizes from M, K, N
  std::vector<int64_t> input_size = {config.M, config.K};
  // Input tensor (float/half) - [M, K]
  ValueSpec input_tensor(
      input_size,
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDINT);

  if (debugging()) {
    print_valuespec_data(input_tensor, "input_tensor");
  }

  // For activation+weight quantized linear (linear_dq8ca_q4gsw)
  // Input scale and zero point as per-input channel tensors
  ValueSpec input_scale(
      {1, config.M}, // Per-input channel tensor
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  input_scale.set_constant(true);

  ValueSpec input_zero_point(
      {1, config.M}, // Per-input channel tensor
      vkapi::kChar,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDINT);
  input_zero_point.set_constant(true);

  // For 4-bit weights, packed size is [N, K/2] since 2 weights per byte
  std::vector<int64_t> weight_size = {config.N, config.K / 2};
  // Quantized weight tensor (uint8, packed 4-bit) - [N, K/2]
  ValueSpec quantized_weight(
      weight_size,
      vkapi::kByte, // uint8 for packed 4-bit quantized weights
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDINT4);
  quantized_weight.set_constant(true);
  quantized_weight.set_int4(true);

  if (debugging()) {
    print_valuespec_data(quantized_weight, "weight_tensor");
  }

  // Weight quantization scales (float/half, per-group)
  // For group symmetric quantization: [K/group_size, N]
  // Each group of input features has scales for all output features
  std::vector<int64_t> weight_scales_size = {
      config.K / config.group_size, config.N};
  ValueSpec weight_scales(
      weight_scales_size,
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  weight_scales.set_constant(true);

  // Pre-computed per-group weight sums for zero point adjustment
  // This is needed for activation+weight quantized operations
  // Size: [K/group_size, N] - one sum per group per output feature
  ValueSpec weight_sums(
      weight_scales_size, // Same size as weight_scales
      vkapi::kInt,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  weight_sums.set_constant(true);

  // Compute weight_sums data based on quantized weights
  int64_t num_groups = config.K / config.group_size;
  compute_weight_sums_4bit_grouped(
      weight_sums, quantized_weight, num_groups, config.N, config.group_size);

  // Group size parameter
  ValueSpec group_size_spec(static_cast<int32_t>(config.group_size));

  // Bias (optional, float/half) - [N]
  ValueSpec bias(
      {config.N}, // Per output feature
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  bias.set_constant(true);
  if (!config.has_bias) {
    bias.set_none(true);
  }

  // Output tensor (float/half) - [M, N]
  ValueSpec output(
      {config.M, config.N},
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);

  // Add all specs to test case based on operator type
  if (config.op_name.find("dq8ca") != std::string::npos) {
    // For activation+weight quantized linear (linear_dq8ca_q4gsw)
    test_case.add_input_spec(input_tensor);
    test_case.add_input_spec(input_scale);
    test_case.add_input_spec(input_zero_point);
    test_case.add_input_spec(quantized_weight);
    test_case.add_input_spec(weight_sums);
    test_case.add_input_spec(weight_scales);
    test_case.add_input_spec(group_size_spec);
    test_case.add_input_spec(bias);
    test_case.add_output_spec(output);
  } else {
    // For weight-only quantized linear (linear_q4gsw)
    test_case.add_input_spec(input_tensor);
    test_case.add_input_spec(quantized_weight);
    test_case.add_input_spec(weight_scales);
    test_case.add_input_spec(group_size_spec);
    test_case.add_input_spec(bias);
    test_case.add_output_spec(output);
  }

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
      true, // has_bias
      "simple", // test_case_name
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
      // // Gemv test cases
      // {1, 128, 64, 32},
      // {1, 256, 128, 64},
      // Gemm
      {4, 64, 32, 16},
      {4, 128, 64, 32},
      {4, 256, 128, 64},
      {32, 64, 32, 16},
      {32, 128, 64, 32},
      {32, 256, 128, 64},
      // No bias tests
      {32, 128, 64, 32, false},
      {32, 256, 128, 64, false},
      // Performance test cases
      {1, 2048, 2048, 128},
      {128, 2048, 2048, 256},
      {256, 2048, 2048, 256},
      {1024, 2048, 2048, 256},
  };

  // Test with different storage types and data types
  std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  for (auto config : configs) {
    std::string prefix =
        (config.M < kRefDimSizeLimit && config.K < kRefDimSizeLimit &&
         config.N < kRefDimSizeLimit)
        ? "correctness_"
        : "performance_";
    std::string generated_test_case_name = prefix + std::to_string(config.M) +
        "_" + std::to_string(config.K) + "_" + std::to_string(config.N) + "_g" +
        std::to_string(config.group_size);
    if (!config.has_bias) {
      generated_test_case_name += "_no_bias";
    }

    config.test_case_name = generated_test_case_name;

    for (const auto& storage_type : storage_types) {
      // Test both activation+weight quantized and weight only quantized, but
      // only if the current device supports int8 dot product
      if (vkcompute::api::context()
              ->adapter_ptr()
              ->supports_int8_dot_product()) {
        test_cases.push_back(
            create_test_case_from_config(config, storage_type, vkapi::kFloat));
      }

      LinearConfig wo_quant_config = config;
      wo_quant_config.op_name = "linear_q4gsw";
      test_cases.push_back(create_test_case_from_config(
          wo_quant_config, storage_type, vkapi::kFloat));
    }
  }

  return test_cases;
}

// Reference implementation for 4-bit group symmetric weight quantized linear
void linear_q4gsw_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_scales_spec = test_case.inputs()[idx++];
  const ValueSpec& group_size_spec = test_case.inputs()[idx++];
  const ValueSpec& bias_spec = test_case.inputs()[idx++];

  // Extract output specification (mutable reference)
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions
  auto input_sizes = input_spec.get_tensor_sizes(); // [batch_size, in_features]
  auto weight_sizes =
      weight_spec.get_tensor_sizes(); // [in_features, out_features/2]
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
  auto& weight_data = weight_spec.get_uint8_data();
  auto& weight_scales_data = weight_scales_spec.get_float_data();
  auto& bias_data = bias_spec.get_float_data();

  // Calculate number of output elements
  int64_t num_output_elements = batch_size * out_features;

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_output_elements);

  // Perform quantized linear transformation (matrix multiplication)
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t out_f = 0; out_f < out_features; ++out_f) {
      float sum = 0.0f;

      // Matrix multiplication: output[b][out_f] = sum(input[b][in_f] *
      // weight[out_f][in_f])
      for (int64_t in_f = 0; in_f < in_features; ++in_f) {
        // Get input value
        int64_t input_idx = b * in_features + in_f;
        float input_val = input_data[input_idx];

        // Get weight value and dequantize (4-bit group symmetric quantization)
        int64_t group_idx = in_f / group_size;
        int64_t scales_idx = group_idx * out_features + out_f;

        // Get packed weight value - weight matrix is [N, K/2]
        int64_t weight_idx = (out_f) * (in_features / 2) + (in_f / 2);
        uint8_t packed_weight = weight_data[weight_idx];

        // Unpack 4-bit weight
        auto unpacked = unpack_4bit(packed_weight);
        int8_t weight_4bit = (in_f % 2 == 0) ? unpacked.first : unpacked.second;

        // Dequantize weight using group symmetric quantization (no zero point)
        float weight_scale = weight_scales_data[scales_idx];
        float dequant_weight = static_cast<float>(weight_4bit) * weight_scale;

        sum += input_val * dequant_weight;
      }

      // Add bias and store result
      if (!bias_spec.is_none()) {
        sum += bias_data[out_f];
      }
      int64_t output_idx = b * out_features + out_f;
      ref_data[output_idx] = sum;
    }
  }
}

// Reference implementation for activation+weight quantized linear (dq8ca_q4gsw)
void linear_dq8ca_q4gsw_reference_impl(TestCase& test_case) {
  // Extract input specifications
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& input_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& input_zeros_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_sums_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_scales_spec = test_case.inputs()[idx++];
  const ValueSpec& group_size_spec = test_case.inputs()[idx++];
  const ValueSpec& bias_spec = test_case.inputs()[idx++];

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
  auto& input_scale_data =
      input_scale_spec.get_float_data(); // Per-input channel tensor
  auto& input_zero_point_data =
      input_zeros_spec.get_int8_data(); // Per-input channel tensor

  auto& weight_data = weight_spec.get_uint8_data();
  auto& weight_sums_data = weight_sums_spec.get_int32_data();
  (void)weight_sums_data; // Unused for now
  auto& weight_scales_data = weight_scales_spec.get_float_data();
  auto& bias_data = bias_spec.get_float_data();

  // Calculate number of output elements
  int64_t num_output_elements = batch_size * out_features;

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_output_elements);

  // Perform quantized linear transformation (matrix multiplication) with
  // integer accumulation
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t out_f = 0; out_f < out_features; ++out_f) {
      int32_t int_sum = 0;
      (void)int_sum;
      int32_t weight_sum = 0; // Track weight sum on the fly for each group
      (void)weight_sum;

      // For group symmetric quantization, compute with proper grouping for
      // accurate reference
      float float_result = 0.0f;

      for (int64_t in_f = 0; in_f < in_features; ++in_f) {
        // Get input value and quantize to int8 using per-input channel
        // parameters
        int64_t input_idx = b * in_features + in_f;

        // Use per-input channel scale and zero point - index by batch dimension
        float input_scale = input_scale_data[b]; // {1, M} -> index by batch
        int8_t input_zero_point =
            input_zero_point_data[b]; // {1, M} -> index by batch

        float quant_input_f =
            std::round(input_data[input_idx] / input_scale) + input_zero_point;
        quant_input_f = std::min(std::max(quant_input_f, -128.0f), 127.0f);
        int8_t quantized_input = static_cast<int8_t>(quant_input_f);

        // Get quantized weight and its scale
        int64_t weight_idx = out_f * (in_features / 2) + (in_f / 2);
        uint8_t packed_weight = weight_data[weight_idx];
        auto unpacked = unpack_4bit(packed_weight);
        int8_t quantized_weight =
            (in_f % 2 == 0) ? unpacked.first : unpacked.second;

        // Get the appropriate scale for this group
        int64_t group_idx = in_f / group_size;
        int64_t scales_idx = group_idx * out_features + out_f;
        float weight_scale = weight_scales_data[scales_idx];

        // Compute the contribution with proper scaling
        float contribution =
            static_cast<float>(quantized_input - input_zero_point) *
            static_cast<float>(quantized_weight) * input_scale * weight_scale;

        float_result += contribution;
      }

      // Add bias and store result
      if (!bias_spec.is_none()) {
        float_result += bias_data[out_f];
      }
      int64_t output_idx = b * out_features + out_f;
      ref_data[output_idx] = float_result;
    }
  }
}

void reference_impl(TestCase& test_case) {
  if (test_case.operator_name().find("dq8ca") != std::string::npos) {
    linear_dq8ca_q4gsw_reference_impl(test_case);
  } else {
    linear_q4gsw_reference_impl(test_case);
  }
}

int64_t quantized_linear_flop_calculator(const TestCase& test_case) {
  int input_idx = 0;
  int weight_idx = 1;
  if (test_case.operator_name().find("dq8ca") != std::string::npos) {
    input_idx = 0;
    weight_idx = 3; // Weight comes after input, input_scale, input_zero_point
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
  // - Unpack 4-bit weight: 1 op per weight element used
  // - Dequantize weight: 1 op per weight element used
  // - Add bias: 1 op per output element
  // - For activation+weight quantized: add input quantization ops
  int64_t quantization_ops = ops_per_output * 2 + 1; // Simplified estimate

  int64_t flop = output_elements * (ops_per_output + quantization_ops);

  return flop;
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout
      << "4-bit Group Symmetric Weight Quantized Linear Operation Prototyping Framework"
      << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  // Execute easy test cases using the new framework with custom FLOP calculator
  auto results = execute_test_cases(
      generate_quantized_linear_test_cases,
      quantized_linear_flop_calculator,
      "QuantizedLinearQ4GSW",
      10,
      10,
      ref_fn);

  return 0;
}
