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

static constexpr int64_t kRefDimSizeLimit = 2050;
static constexpr float SMALL_SCALE_THRESHOLD = 6.1e-5f;

// ChooseQParams configuration struct
struct ChooseQParamsConfig {
  int64_t num_channels; // Height dimension (number of channels)
  int64_t channel_size; // Width dimension (size per channel)
  int32_t quant_min = -128;
  int32_t quant_max = 127;
  std::string test_case_name = "placeholder";
  std::string op_name = "choose_qparams_per_row";
};

// Utility function to create a test case from a ChooseQParamsConfig
TestCase create_test_case_from_config(
    const ChooseQParamsConfig& config,
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

  // Input tensor (float) - [num_channels, channel_size]
  std::vector<int64_t> input_size = {config.num_channels, config.channel_size};
  ValueSpec input_tensor(
      input_size,
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM);

  if (debugging()) {
    print_valuespec_data(input_tensor, "input_tensor");
  }

  // Quantization parameters
  ValueSpec quant_min(config.quant_min);
  ValueSpec quant_max(config.quant_max);

  // Output scale tensor (float) - [num_channels]
  ValueSpec scale_out(
      {config.num_channels},
      vkapi::kFloat,
      utils::kTexture3D, // Always buffer as per requirement
      utils::kWidthPacked,
      DataGenType::ZEROS);

  // Output zero_point tensor (int8) - [num_channels]
  ValueSpec zero_point_out(
      {config.num_channels},
      vkapi::kChar, // int8 for quantized zero point
      utils::kTexture3D, // Always buffer as per requirement
      utils::kWidthPacked,
      DataGenType::ZEROS);

  // Add all specs to test case
  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(quant_min);
  test_case.add_input_spec(quant_max);
  test_case.add_output_spec(scale_out);
  test_case.add_output_spec(zero_point_out);

  return test_case;
}

// CPU reference implementation matching the behavior from op_choose_qparams.cpp
void calculate_scale_and_zero_point_reference(
    float min_val,
    float max_val,
    int32_t qmin,
    int32_t qmax,
    float& scale,
    int32_t& zero_point) {
  // Extend the [min, max] interval to ensure that it contains 0
  min_val = std::min(min_val, 0.0f);
  max_val = std::max(max_val, 0.0f);

  // Use double precision for intermediate computation but use single precision
  // in final number to reflect the actual number used during quantization.
  double scale_double =
      (static_cast<double>(max_val) - min_val) / (qmax - qmin);

  // If scale is 0 or too small so its reciprocal is infinity, we arbitrary
  // adjust the scale to 0.1 . We want to avoid scale's reciprocal being
  // infinity because some of fbgemm code pre-computes scale's reciprocal to do
  // multiplication instead of division in the time critical part of code.
  if (static_cast<float>(scale_double) == 0.0f ||
      std::isinf(1.0f / static_cast<float>(scale_double))) {
    scale_double = 0.1;
  }

  // Cut off small scale
  if (scale_double < SMALL_SCALE_THRESHOLD) {
    float org_scale = static_cast<float>(scale_double);
    scale_double = SMALL_SCALE_THRESHOLD;
    // Adjust the min and max based on the new scale
    if (min_val == 0.0f) {
      max_val = SMALL_SCALE_THRESHOLD * (qmax - qmin);
    } else if (max_val == 0.0f) {
      min_val = -SMALL_SCALE_THRESHOLD * (qmax - qmin);
    } else {
      float amplifier = SMALL_SCALE_THRESHOLD / org_scale;
      min_val *= amplifier;
      max_val *= amplifier;
    }
  }

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  double zero_point_from_min = qmin - min_val / scale_double;
  double zero_point_from_max = qmax - max_val / scale_double;
  double zero_point_from_min_error =
      std::abs(qmin) - std::abs(min_val / scale_double);
  double zero_point_from_max_error =
      std::abs(qmax) - std::abs(max_val / scale_double);
  double initial_zero_point =
      zero_point_from_min_error < zero_point_from_max_error
      ? zero_point_from_min
      : zero_point_from_max;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with zero
  // padding).
  int32_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point =
        static_cast<int32_t>(nearbyint(static_cast<float>(initial_zero_point)));
  }

  scale = static_cast<float>(scale_double);
  zero_point = nudged_zero_point;
}

// Generate easy test cases for choose_qparams_per_channel operation (for
// debugging)
std::vector<TestCase> generate_choose_qparams_per_channel_easy_cases() {
  std::vector<TestCase> test_cases;

  // Single simple configuration for debugging
  int num_channels = 4;
  int channel_size = 8;

  ChooseQParamsConfig config = {
      num_channels, // num_channels
      channel_size, // channel_size
      -128, // quant_min
      127, // quant_max
      "simple", // test_case_name
  };

  // Test with both storage types
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

// Generate test cases for choose_qparams_per_channel operation
std::vector<TestCase> generate_choose_qparams_per_channel_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<ChooseQParamsConfig> configs = {
      {4, 16},
      {8, 32},
      {16, 64},
      {32, 128},
      {64, 256},
      {128, 512},
      {1, 512},
      // Performance cases
      {256, 1024},
      {512, 2048},
      {1, 2048},
      {1, 8096},
  };

  // Test with different storage types
  std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  for (auto config : configs) {
    std::string prefix = (config.num_channels < kRefDimSizeLimit &&
                          config.channel_size < kRefDimSizeLimit)
        ? "correctness_"
        : "performance_";
    std::string generated_test_case_name = prefix +
        std::to_string(config.num_channels) + "_" +
        std::to_string(config.channel_size);

    config.test_case_name = generated_test_case_name;

    for (const auto& storage_type : storage_types) {
      test_cases.push_back(
          create_test_case_from_config(config, storage_type, vkapi::kFloat));
    }
  }

  return test_cases;
}

// Reference implementation for choose_qparams_per_channel
void choose_qparams_per_channel_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& quant_min_spec = test_case.inputs()[idx++];
  const ValueSpec& quant_max_spec = test_case.inputs()[idx++];
  const ValueSpec& eps_spec = test_case.inputs()[idx++];
  const ValueSpec& dtype_spec = test_case.inputs()[idx++];
  (void)eps_spec; // Unused in reference implementation
  (void)dtype_spec; // Unused in reference implementation

  // Extract output specifications
  ValueSpec& scale_out_spec = test_case.outputs()[0];
  ValueSpec& zero_point_out_spec = test_case.outputs()[1];

  // Get tensor dimensions
  auto input_sizes =
      input_spec.get_tensor_sizes(); // [num_channels, channel_size]
  int64_t num_channels = input_sizes[0];
  int64_t channel_size = input_sizes[1];

  // Skip for large tensors since computation time will be extremely slow
  if (num_channels > kRefDimSizeLimit || channel_size > kRefDimSizeLimit) {
    throw std::invalid_argument(
        "One or more dimensions (num_channels, channel_size) exceed the allowed limit for reference implementation.");
  }

  if (input_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  // Get raw data pointers
  auto& input_data = input_spec.get_float_data();
  int32_t quant_min = quant_min_spec.get_int_value();
  int32_t quant_max = quant_max_spec.get_int_value();

  // Prepare output data
  auto& scale_ref_data = scale_out_spec.get_ref_float_data();
  auto& zero_point_ref_data = zero_point_out_spec.get_ref_int8_data();
  scale_ref_data.resize(num_channels);
  zero_point_ref_data.resize(num_channels);

  // Process each channel
  for (int64_t channel = 0; channel < num_channels; ++channel) {
    // Find min and max for this channel
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    for (int64_t i = 0; i < channel_size; ++i) {
      int64_t input_idx = channel * channel_size + i;
      float val = input_data[input_idx];
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
    }

    // Calculate scale and zero point for this channel
    float scale;
    int32_t zero_point;
    calculate_scale_and_zero_point_reference(
        min_val, max_val, quant_min, quant_max, scale, zero_point);

    // Store results (cast zero_point to int8)
    scale_ref_data[channel] = scale;
    zero_point_ref_data[channel] = static_cast<int8_t>(zero_point);
  }
}

void reference_impl(TestCase& test_case) {
  choose_qparams_per_channel_reference_impl(test_case);
}

int64_t choose_qparams_per_channel_flop_calculator(const TestCase& test_case) {
  // Get input dimensions
  const auto& input_sizes = test_case.inputs()[0].get_tensor_sizes();
  int64_t num_channels = input_sizes[0];
  int64_t channel_size = input_sizes[1];

  // Calculate FLOPs for choose_qparams_per_channel operation
  // Each channel requires:
  // - Min/max finding: approximately 2 * channel_size comparisons
  // - Scale calculation: ~5 operations (division, min/max operations)
  // - Zero point calculation: ~10 operations (multiple arithmetic operations)
  int64_t ops_per_channel = 2 * channel_size + 15; // Simplified estimate

  int64_t flop = num_channels * ops_per_channel;

  return flop;
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Choose QParams Per Channel Operation Prototyping Framework"
            << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  auto results = execute_test_cases(
      generate_choose_qparams_per_channel_test_cases,
      choose_qparams_per_channel_flop_calculator,
      "ChooseQParamsPerChannel",
      0,
      10,
      ref_fn);

  return 0;
}
