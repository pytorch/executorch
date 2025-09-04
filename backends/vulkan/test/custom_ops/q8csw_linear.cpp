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
  bool has_bias = true;
  std::string test_case_name = "placeholder";
  std::string op_name = "linear_q8ta_q8csw";
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

  std::string test_name =
      config.test_case_name + "_" + storage_str + "_" + dtype_str;
  test_case.set_name(test_name);

  // Set the operator name for the test case
  std::string operator_name = "et_vk." + config.op_name + ".default";
  test_case.set_operator_name(operator_name);

  // Derive sizes from M, K, N
  std::vector<int64_t> input_size = {config.M, config.K};
  std::vector<int64_t> weight_size = {config.N, config.K};

  // Input tensor (float/half) - [M, K]
  ValueSpec input_tensor(
      input_size,
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM);

  if (debugging()) {
    print_valuespec_data(input_tensor, "input_tensor");
  }

  float input_scale_val = 0.008f;
  ValueSpec input_scale(input_scale_val);

  int32_t input_zero_point_val = -2;
  ValueSpec input_zero_point(input_zero_point_val);

  // Quantized weight tensor (int8) - [K, N]
  ValueSpec quantized_weight(
      weight_size,
      vkapi::kChar, // int8 for quantized weights
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDINT8);
  quantized_weight.set_constant(true);

  if (debugging()) {
    print_valuespec_data(quantized_weight, "weight_tensor");
  }

  // Weight quantization scales (float/half, per-channel)
  ValueSpec weight_scales(
      {config.N}, // Per output feature
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  weight_scales.set_constant(true);

  ValueSpec weight_sums(
      {config.N}, // Per output features
      vkapi::kFloat,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  weight_sums.set_constant(true);

  // Compute weight_sums data based on quantized weights
  int64_t in_features = config.K;
  int64_t out_features = config.N;
  compute_weight_sums(weight_sums, quantized_weight, out_features, in_features);

  // Bias (optional, float/half) - [N]
  ValueSpec bias(
      {config.N}, // Per output feature
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM);
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

  // Add all specs to test case
  if (config.op_name.find("q8ta") != std::string::npos) {
    test_case.add_input_spec(input_tensor);
    test_case.add_input_spec(input_scale);
    test_case.add_input_spec(input_zero_point);
    test_case.add_input_spec(quantized_weight);
    test_case.add_input_spec(weight_sums);
    test_case.add_input_spec(weight_scales);
    test_case.add_input_spec(bias);
    test_case.add_output_spec(output);
  } else {
    test_case.add_input_spec(input_tensor);
    test_case.add_input_spec(quantized_weight);
    test_case.add_input_spec(weight_scales);
    test_case.add_input_spec(bias);
    test_case.add_output_spec(output);
  }

  return test_case;
}

// Generate easy test cases for quantized linear operation (for debugging)
std::vector<TestCase> generate_quantized_linear_easy_cases() {
  std::vector<TestCase> test_cases;

  // Single simple configuration for debugging
  int M = 4;
  int K = 4;
  int N = 4;

  LinearConfig config = {
      M, // Batch size
      K, // Input features
      N, // Output features
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
      {4, 64, 32},
      {4, 128, 64},
      {4, 256, 128},
      {32, 64, 32},
      {32, 128, 64},
      {32, 256, 128},
      // No bias tests
      {32, 128, 64, false},
      {32, 256, 128, false},
      {256, 2048, 2048},
      {512, 2048, 2048},
      {1024, 2048, 2048},
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
        "_" + std::to_string(config.K) + "_" + std::to_string(config.N);
    if (!config.has_bias) {
      generated_test_case_name += "_no_bias";
    }

    config.test_case_name = generated_test_case_name;

    for (const auto& storage_type : storage_types) {
      if (vkcompute::api::context()
              ->adapter_ptr()
              ->supports_int8_dot_product()) {
        // Test both activation+weight quantized and weight only quantized
        test_cases.push_back(
            create_test_case_from_config(config, storage_type, vkapi::kFloat));
      }

      LinearConfig wo_quant_config = config;
      wo_quant_config.op_name = "linear_q8csw";
      test_cases.push_back(create_test_case_from_config(
          wo_quant_config, storage_type, vkapi::kFloat));
    }
  }

  return test_cases;
}

// Reference implementation for weight only quantized linear
void linear_q8csw_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_scales_spec = test_case.inputs()[idx++];
  const ValueSpec& bias_spec = test_case.inputs()[idx++];

  // Extract output specification (mutable reference)
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions
  auto input_sizes = input_spec.get_tensor_sizes(); // [batch_size, in_features]
  auto weight_sizes =
      weight_spec.get_tensor_sizes(); // [out_features, in_features]
  auto output_sizes =
      output_spec.get_tensor_sizes(); // [batch_size, out_features]

  int64_t batch_size = input_sizes[0];
  int64_t in_features = input_sizes[1];
  int64_t out_features = weight_sizes[0];

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
        // Get input value and dequantize
        int64_t input_idx = b * in_features + in_f;
        float input_val = input_data[input_idx];

        // Get weight value and dequantize
        int64_t weight_idx = out_f * in_features + in_f;
        float dequant_weight = (static_cast<float>(weight_data[weight_idx])) *
            weight_scales_data[out_f];

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

void linear_q8ta_q8csw_reference_impl(TestCase& test_case) {
  // Extract input specifications
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& input_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& input_zeros_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_sums_spec = test_case.inputs()[idx++];
  (void)weight_sums_spec;
  const ValueSpec& weight_scales_spec = test_case.inputs()[idx++];
  const ValueSpec& bias_spec = test_case.inputs()[idx++];

  // Extract output specification (mutable reference)
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions
  auto input_sizes = input_spec.get_tensor_sizes(); // [batch_size, in_features]
  auto weight_sizes =
      weight_spec.get_tensor_sizes(); // [out_features, in_features]
  auto output_sizes =
      output_spec.get_tensor_sizes(); // [batch_size, out_features]

  int64_t batch_size = input_sizes[0];
  int64_t in_features = input_sizes[1];
  int64_t out_features = weight_sizes[0];

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
  const float input_scale = input_scale_spec.get_float_value();
  const int32_t input_zero_point = input_zeros_spec.get_int_value();

  auto& weight_data = weight_spec.get_int8_data();
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
      int32_t weight_sum = 0; // Track weight sum on the fly

      // Matrix multiplication with integer accumulation:
      // int_sum = sum(quantized_input[b][in_f] * quantized_weight[out_f][in_f])
      for (int64_t in_f = 0; in_f < in_features; ++in_f) {
        // Get input value and quantize to int8
        int64_t input_idx = b * in_features + in_f;

        float quant_input_f =
            std::round(input_data[input_idx] / input_scale) + input_zero_point;
        quant_input_f = std::min(std::max(quant_input_f, -128.0f), 127.0f);
        int8_t quantized_input = static_cast<int8_t>(quant_input_f);

        // Get quantized weight (already int8)
        int64_t weight_idx = out_f * in_features + in_f;
        int8_t quantized_weight = weight_data[weight_idx];

        // Integer multiplication and accumulation
        int_sum += static_cast<int32_t>(quantized_input) *
            static_cast<int32_t>(quantized_weight);

        // Track weight sum for this output channel on the fly
        weight_sum += static_cast<int32_t>(quantized_weight);
      }

      // Convert accumulated integer result to float and apply scales
      // Final result = (int_sum - zero_point_correction) * input_scale *
      // weight_scale + bias zero_point_correction = input_zero_point *
      // sum_of_weights_for_this_output_channel
      int32_t zero_point_correction = input_zero_point * weight_sum;
      float float_result = (static_cast<float>(int_sum) -
                            static_cast<float>(zero_point_correction)) *
          input_scale * weight_scales_data[out_f];

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
  if (test_case.operator_name().find("q8ta") != std::string::npos) {
    linear_q8ta_q8csw_reference_impl(test_case);
  } else {
    linear_q8csw_reference_impl(test_case);
  }
}

int64_t quantized_linear_flop_calculator(const TestCase& test_case) {
  int input_idx = 0;
  int weight_idx = 1;
  if (test_case.operator_name().find("q8ta") != std::string::npos) {
    input_idx = 0;
    weight_idx = 3;
  }

  // Get input and weight dimensions
  const auto& input_sizes = test_case.inputs()[input_idx].get_tensor_sizes();
  const auto& weight_sizes = test_case.inputs()[weight_idx].get_tensor_sizes();

  int64_t batch_size = input_sizes[0];
  int64_t in_features = input_sizes[1];
  int64_t out_features = weight_sizes[0];

  // Calculate FLOPs for quantized linear operation
  // Each output element requires:
  // - in_features multiply-accumulate operations
  // - Additional operations for quantization/dequantization
  int64_t output_elements = batch_size * out_features;
  int64_t ops_per_output = in_features;

  // Add quantization overhead (approximate)
  // - Dequantize input: 1 op per input element used
  // - Dequantize weight: 1 op per weight element used
  // - Add bias: 1 op per output element
  int64_t quantization_ops = ops_per_output + 1; // Simplified estimate

  int64_t flop = output_elements * (ops_per_output + quantization_ops);

  return flop;
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Quantized Linear Operation Prototyping Framework" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  auto results = execute_test_cases(
      generate_quantized_linear_test_cases,
      quantized_linear_flop_calculator,
      "QuantizedLinear",
      0,
      10,
      ref_fn);

  return 0;
}
