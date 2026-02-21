// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <vector>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include "utils.h"

using namespace executorch::vulkan::prototyping;

using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 300;

struct LinearConfig {
  int64_t M;
  int64_t K;
  int64_t N;
  bool has_bias = true;
  std::string test_case_name = "placeholder";
};

static TestCase create_test_case_from_config(
    const LinearConfig& config,
    vkapi::ScalarType input_dtype) {
  TestCase test_case;

  std::string dtype_str = (input_dtype == vkapi::kFloat) ? "Float" : "Half";

  std::string test_name = config.test_case_name + "_Buffer_" + dtype_str;
  test_case.set_name(test_name);

  test_case.set_operator_name("test_etvk.test_q8ta_linear.default");

  std::vector<int64_t> input_size = {config.M, config.K};
  std::vector<int64_t> weight_size = {config.N, config.K};

  // Input tensor (float) - [M, K]
  ValueSpec input_tensor(
      input_size,
      input_dtype,
      utils::kBuffer,
      utils::kWidthPacked,
      DataGenType::RANDOM);

  if (debugging()) {
    print_valuespec_data(input_tensor, "input_tensor");
  }

  float input_scale_val = 0.008f;
  ValueSpec input_scale(input_scale_val);

  int32_t input_zero_point_val = -2;
  ValueSpec input_zero_point(input_zero_point_val);

  // Quantized weight tensor (int8) - [N, K]
  ValueSpec quantized_weight(
      weight_size,
      vkapi::kChar,
      utils::kBuffer,
      utils::kWidthPacked,
      DataGenType::RANDINT8);
  quantized_weight.set_constant(true);

  if (debugging()) {
    print_valuespec_data(quantized_weight, "weight_tensor");
  }

  // Weight quantization scales (float, per-channel)
  ValueSpec weight_scales(
      {config.N},
      input_dtype,
      utils::kBuffer,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  weight_scales.set_constant(true);

  ValueSpec weight_sums(
      {config.N},
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  weight_sums.set_constant(true);

  // Compute weight_sums data based on quantized weights
  compute_weight_sums(weight_sums, quantized_weight, config.N, config.K);

  // Output quantization parameters
  float output_scale_val = 0.05314f;
  ValueSpec output_scale(output_scale_val);

  int32_t output_zero_point_val = -1;
  ValueSpec output_zero_point(output_zero_point_val);

  // Bias (optional, float) - [N]
  ValueSpec bias(
      {config.N},
      input_dtype,
      utils::kBuffer,
      utils::kWidthPacked,
      DataGenType::RANDOM);
  bias.set_constant(true);
  if (!config.has_bias) {
    bias.set_none(true);
  }

  // Output tensor (float) - [M, N]
  ValueSpec output(
      {config.M, config.N},
      input_dtype,
      utils::kBuffer,
      utils::kWidthPacked,
      DataGenType::ZEROS);

  // Add all specs to test case
  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(input_scale);
  test_case.add_input_spec(input_zero_point);
  test_case.add_input_spec(quantized_weight);
  test_case.add_input_spec(weight_sums);
  test_case.add_input_spec(weight_scales);
  test_case.add_input_spec(output_scale);
  test_case.add_input_spec(output_zero_point);
  test_case.add_input_spec(bias);

  // Activation (none = no activation)
  ValueSpec activation = ValueSpec::make_string("none");
  test_case.add_input_spec(activation);

  test_case.add_output_spec(output);

  test_case.set_abs_tolerance(output_scale_val + 1e-4f);

  // Filter out quantize/dequantize shaders from timing measurements
  test_case.set_shader_filter({
      "nchw_to",
      "to_nchw",
      "q8ta_quantize",
      "q8ta_dequantize",
  });

  return test_case;
}

// Generate test cases for q8ta_linear operation
static std::vector<TestCase> generate_q8ta_linear_test_cases() {
  std::vector<TestCase> test_cases;
  if (!vkcompute::api::context()->adapter_ptr()->supports_int8_dot_product()) {
    return test_cases;
  }

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
      // Performance cases
      {256, 2048, 2048},
      {512, 2048, 2048},
      {1024, 2048, 2048},
  };

  for (auto config : configs) {
    bool is_performance = config.M >= kRefDimSizeLimit ||
        config.K >= kRefDimSizeLimit || config.N >= kRefDimSizeLimit;

    std::string prefix = is_performance ? "performance_" : "correctness_";
    std::string generated_test_case_name = prefix + std::to_string(config.M) +
        "_" + std::to_string(config.K) + "_" + std::to_string(config.N);
    if (!config.has_bias) {
      generated_test_case_name += "_no_bias";
    }

    config.test_case_name = generated_test_case_name;

    test_cases.push_back(create_test_case_from_config(config, vkapi::kFloat));
  }

  return test_cases;
}

// Reference implementation for q8ta_linear (activation+weight+output quantized)
static void q8ta_linear_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& input_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& input_zeros_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_sums_spec = test_case.inputs()[idx++];
  (void)weight_sums_spec;
  const ValueSpec& weight_scales_spec = test_case.inputs()[idx++];
  const ValueSpec& output_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& output_zeros_spec = test_case.inputs()[idx++];
  const ValueSpec& bias_spec = test_case.inputs()[idx++];

  ValueSpec& output_spec = test_case.outputs()[0];

  auto input_sizes = input_spec.get_tensor_sizes();
  auto weight_sizes = weight_spec.get_tensor_sizes();

  int64_t batch_size = input_sizes[0];
  int64_t in_features = input_sizes[1];
  int64_t out_features = weight_sizes[0];

  if (batch_size > kRefDimSizeLimit || in_features > kRefDimSizeLimit ||
      out_features > kRefDimSizeLimit) {
    throw std::invalid_argument(
        "One or more dimensions exceed the allowed limit for reference implementation.");
  }

  if (input_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  auto& input_data = input_spec.get_float_data();
  const float input_scale = input_scale_spec.get_float_value();
  const int32_t input_zero_point = input_zeros_spec.get_int_value();

  auto& weight_data = weight_spec.get_int8_data();
  auto& weight_scales_data = weight_scales_spec.get_float_data();
  auto& bias_data = bias_spec.get_float_data();

  const float output_scale = output_scale_spec.get_float_value();
  const int32_t output_zero_point = output_zeros_spec.get_int_value();

  int64_t num_output_elements = batch_size * out_features;

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_output_elements);

  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t out_f = 0; out_f < out_features; ++out_f) {
      int32_t int_sum = 0;
      int32_t weight_sum = 0;

      for (int64_t in_f = 0; in_f < in_features; ++in_f) {
        int64_t input_idx = b * in_features + in_f;

        float quant_input_f =
            std::round(input_data[input_idx] / input_scale) + input_zero_point;
        quant_input_f = std::min(std::max(quant_input_f, -128.0f), 127.0f);
        int8_t quantized_input = static_cast<int8_t>(quant_input_f);

        int64_t weight_idx = out_f * in_features + in_f;
        int8_t quantized_weight = weight_data[weight_idx];

        int_sum += static_cast<int32_t>(quantized_input) *
            static_cast<int32_t>(quantized_weight);

        weight_sum += static_cast<int32_t>(quantized_weight);
      }

      int32_t zero_point_correction = input_zero_point * weight_sum;
      int32_t accum_adjusted = int_sum - zero_point_correction;

      float float_result =
          accum_adjusted * input_scale * weight_scales_data[out_f];

      if (!bias_spec.is_none()) {
        float_result += bias_data[out_f];
      }

      // Quantize the output to int8
      float quant_output_f =
          std::round(float_result / output_scale) + output_zero_point;
      quant_output_f = std::min(std::max(quant_output_f, -128.0f), 127.0f);
      int8_t quantized_output = static_cast<int8_t>(quant_output_f);

      // Dequantize back to float (this is what the test wrapper does)
      float dequant_output =
          (static_cast<float>(quantized_output) - output_zero_point) *
          output_scale;

      int64_t output_idx = b * out_features + out_f;
      ref_data[output_idx] = dequant_output;
    }
  }
}

static void reference_impl(TestCase& test_case) {
  q8ta_linear_reference_impl(test_case);
}

static int64_t q8ta_linear_flop_calculator(const TestCase& test_case) {
  const auto& input_sizes = test_case.inputs()[0].get_tensor_sizes();
  const auto& weight_sizes = test_case.inputs()[3].get_tensor_sizes();

  int64_t batch_size = input_sizes[0];
  int64_t in_features = input_sizes[1];
  int64_t out_features = weight_sizes[0];

  int64_t output_elements = batch_size * out_features;
  int64_t ops_per_output = in_features;

  int64_t flop = output_elements * ops_per_output;

  return flop;
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Q8ta Linear Operation Prototyping Framework" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  auto results = execute_test_cases(
      generate_q8ta_linear_test_cases,
      q8ta_linear_flop_calculator,
      "Q8taLinear",
      3,
      10,
      ref_fn);

  return 0;
}
