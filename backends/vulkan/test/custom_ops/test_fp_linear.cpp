// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <vector>

#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include "nv_utils.h"
#include "utils.h"

#define DEBUG_MODE

using namespace executorch::vulkan::prototyping;

using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 2048;

// Configuration for linear layer test cases
struct LinearConfig {
  int64_t batch_size;
  int64_t in_features;
  int64_t out_features;
  bool has_bias;
  std::string test_case_name;
};

// Utility function to create a test case from a LinearConfig
TestCase create_test_case_from_config(
    const LinearConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    int32_t impl_selector = 0) {
  TestCase test_case;

  utils::GPUMemoryLayout memory_layout = storage_type == utils::kBuffer
      ? utils::kWidthPacked
      : utils::kChannelsPacked;

  // Create test case name
  // Format: ACCU/PERF  B=batch  I=in_features  O=out_features  Tex/Buf
  std::string prefix = config.test_case_name.substr(0, 4); // "ACCU" or "PERF"
  std::string storage_str =
      storage_type == utils::kBuffer ? "Buf" : "Tex";
  std::string dtype_str = dtype == vkapi::kFloat ? "fp32" : "fp16";
  std::string bias_str = config.has_bias ? "+bias" : "";

  std::string test_name = prefix + "  " + "B=" + std::to_string(config.batch_size) +
      "  I=" + std::to_string(config.in_features) +
      "  O=" + std::to_string(config.out_features) + "  " + storage_str + "  " +
      dtype_str + bias_str;
  if (impl_selector == 1) {
    test_name += " Experimental"; // Legacy/alternative implementation
  }
  test_case.set_name(test_name);

  // Set the operator name for the test case - use the test operator
  std::string operator_name = "test_etvk.test_fp_linear.default";
  test_case.set_operator_name(operator_name);

  // Input tensor - [batch_size, in_features]
  std::vector<int64_t> input_size = {config.batch_size, config.in_features};
  ValueSpec input_tensor(
      input_size, dtype, storage_type, memory_layout, DataGenType::RANDOM);

  if (debugging()) {
    print_valuespec_data(input_tensor, "input_tensor", false, 64);
  }

  // Weight tensor - [out_features, in_features]
  std::vector<int64_t> weight_size = {config.out_features, config.in_features};
  ValueSpec weight_tensor(
      weight_size, dtype, storage_type, memory_layout, DataGenType::RANDOM);
  weight_tensor.set_constant(true);

  if (debugging()) {
    print_valuespec_data(weight_tensor, "weight_tensor", false, 64);
  }

  // Bias tensor (optional) - [out_features]
  ValueSpec bias_tensor;
  if (config.has_bias) {
    std::vector<int64_t> bias_size = {config.out_features};
    bias_tensor = ValueSpec(
        bias_size, dtype, storage_type, utils::kWidthPacked, DataGenType::RANDOM);
    bias_tensor.set_constant(true);

    if (debugging()) {
      print_valuespec_data(bias_tensor, "bias_tensor", false, 64);
    }
  } else {
    bias_tensor = ValueSpec();
    bias_tensor.set_none(true);
  }

  // Output tensor - [batch_size, out_features]
  std::vector<int64_t> output_size = {config.batch_size, config.out_features};
  ValueSpec output_tensor(
      output_size, dtype, storage_type, memory_layout, DataGenType::ZEROS);

  // Add impl_selector parameter
  ValueSpec impl_selector_spec(impl_selector);

  // Add all specs to test case
  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(weight_tensor);
  test_case.add_input_spec(bias_tensor);
  test_case.add_input_spec(impl_selector_spec);
  test_case.add_output_spec(output_tensor);

  // Set tolerance based on dtype
  if (dtype == vkapi::kFloat) {
    test_case.set_abs_tolerance(1e-4f);
  } else {
    // FP16 cooperative matrix operations have slightly more numerical variance
    test_case.set_abs_tolerance(2e-1f);
  }

  return test_case;
}

// Generate easy test cases for debugging
std::vector<TestCase> generate_linear_easy_cases() {
  std::vector<TestCase> test_cases;

  // Test with multiple row tiles only (2 row tiles, 1 column tile)
  // Using M=32, N=16 to get 2x1 workgroups (blocks_m=2, blocks_n=1)
  LinearConfig config = {
      64,   // batch_size (2 tiles in M dimension)
      256,   // in_features
      256,   // out_features (1 tile in N dimension)
      true, // has_bias
      "PERF",
  };

  std::vector<utils::StorageType> storage_types = {utils::kBuffer};
  // Use FP16 for cooperative matrix shader (RTX 4080 requires float16 A/B/C)
  std::vector<vkapi::ScalarType> dtypes = {vkapi::kHalf};

  for (const utils::StorageType storage_type : storage_types) {
    for (const vkapi::ScalarType dtype : dtypes) {
      config.test_case_name = "PERF";
      // Test with impl_selector = 0 (default)
      test_cases.push_back(
          create_test_case_from_config(config, dtype, storage_type, 0));
      // // Test with impl_selector = 1 (alternative)
      test_cases.push_back(
          create_test_case_from_config(config, dtype, storage_type, 1));
    }
  }

  return test_cases;
}

// Generate comprehensive test cases for linear layer
std::vector<TestCase> generate_linear_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<LinearConfig> configs = {
      // Small accuracy test cases
      {1, 64, 64, true, "ACCU"},
      {4, 128, 64, true, "ACCU"},
      {8, 256, 128, true, "ACCU"},
      {16, 64, 256, true, "ACCU"},
      {1, 512, 512, true, "ACCU"},
      // Without bias
      {4, 128, 64, false, "ACCU"},
      {8, 256, 128, false, "ACCU"},

      // Performance test cases - BERT/ViT style (hidden_dim=768, ffn=3072)
      {1, 768, 768, true, "PERF"},
      {32, 768, 768, true, "PERF"},
      {64, 768, 768, true, "PERF"},
  };

  std::vector<utils::StorageType> storage_types = {utils::kBuffer};
  std::vector<vkapi::ScalarType> dtypes = {vkapi::kHalf};

  for (auto& config : configs) {
    bool is_performance = config.batch_size > kRefDimSizeLimit ||
        config.in_features > kRefDimSizeLimit ||
        config.out_features > kRefDimSizeLimit;

    for (const utils::StorageType storage_type : storage_types) {
      for (const vkapi::ScalarType dtype : dtypes) {
        config.test_case_name = is_performance ? "PERF" : "ACCU";
        // Test with impl_selector = 0 (default)
        test_cases.push_back(
            create_test_case_from_config(config, dtype, storage_type, 0));
        // Test with impl_selector = 1 (alternative)
        test_cases.push_back(
            create_test_case_from_config(config, dtype, storage_type, 1));
      }
    }
  }

  return test_cases;
}

// Reference implementation for fp32/fp16 linear layer
void linear_reference_impl(TestCase& test_case) {
  // Extract input specifications
  const ValueSpec& input_spec = test_case.inputs()[0];
  const ValueSpec& weight_spec = test_case.inputs()[1];
  const ValueSpec& bias_spec = test_case.inputs()[2];
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions
  auto input_sizes = input_spec.get_tensor_sizes();   // [batch, in_features]
  auto weight_sizes = weight_spec.get_tensor_sizes(); // [out_features, in_features]
  auto output_sizes = output_spec.get_tensor_sizes(); // [batch, out_features]

  int64_t batch_size = input_sizes[0];
  int64_t in_features = input_sizes[1];
  int64_t out_features = weight_sizes[0];

  // Skip for large tensors since computation time will be extremely slow
  if (batch_size > kRefDimSizeLimit || in_features > kRefDimSizeLimit ||
      out_features > kRefDimSizeLimit) {
    throw std::invalid_argument(
        "One or more dimensions exceed the allowed limit for reference implementation.");
  }

  bool has_bias = !bias_spec.is_none();

  // Get raw data pointers based on dtype
  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(batch_size * out_features);

  if (input_spec.dtype == vkapi::kFloat) {
    auto& input_data = input_spec.get_float_data();
    auto& weight_data = weight_spec.get_float_data();

    // Perform linear operation: output = input @ weight^T + bias
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t o = 0; o < out_features; ++o) {
        float sum = 0.0f;
        for (int64_t i = 0; i < in_features; ++i) {
          // input[b, i] * weight[o, i]
          int64_t input_idx = b * in_features + i;
          int64_t weight_idx = o * in_features + i;
          sum += input_data[input_idx] * weight_data[weight_idx];
        }

        // Add bias if present
        if (has_bias) {
          auto& bias_data = bias_spec.get_float_data();
          sum += bias_data[o];
        }

        int64_t output_idx = b * out_features + o;
        ref_data[output_idx] = sum;
      }
    }
  } else if (input_spec.dtype == vkapi::kHalf) {
    auto& input_data = input_spec.get_half_data();
    auto& weight_data = weight_spec.get_half_data();

    // IEEE 754 FP16 to float conversion helper
    auto half_to_float = [](uint16_t h) -> float {
      uint32_t sign = (h >> 15) & 0x1;
      uint32_t exponent = (h >> 10) & 0x1F;
      uint32_t mantissa = h & 0x3FF;

      uint32_t f_sign = sign << 31;
      uint32_t f_exp;
      uint32_t f_mant;

      if (exponent == 0) {
        if (mantissa == 0) {
          f_exp = 0;
          f_mant = 0;
        } else {
          // Denormalized
          uint32_t exp_adj = 1;
          uint32_t mant_temp = mantissa;
          while ((mant_temp & 0x400) == 0) {
            mant_temp <<= 1;
            exp_adj--;
          }
          mant_temp &= 0x3FF;
          f_exp = (127 - 15 + exp_adj) << 23;
          f_mant = mant_temp << 13;
        }
      } else if (exponent == 31) {
        f_exp = 0xFF << 23;
        f_mant = mantissa << 13;
      } else {
        f_exp = (exponent + 127 - 15) << 23;
        f_mant = mantissa << 13;
      }

      uint32_t bits = f_sign | f_exp | f_mant;
      float result;
      std::memcpy(&result, &bits, sizeof(result));
      return result;
    };

    // Perform linear operation: output = input @ weight^T + bias
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t o = 0; o < out_features; ++o) {
        float sum = 0.0f;
        for (int64_t i = 0; i < in_features; ++i) {
          // input[b, i] * weight[o, i]
          // Convert from IEEE 754 FP16 to float
          int64_t input_idx = b * in_features + i;
          int64_t weight_idx = o * in_features + i;
          float input_val = half_to_float(input_data[input_idx]);
          float weight_val = half_to_float(weight_data[weight_idx]);
          sum += input_val * weight_val;
        }

        // Add bias if present
        if (has_bias) {
          auto& bias_data = bias_spec.get_half_data();
          sum += half_to_float(bias_data[o]);
        }

        int64_t output_idx = b * out_features + o;
        ref_data[output_idx] = sum;
      }
    }
  } else {
    throw std::invalid_argument("Unsupported dtype for linear reference impl");
  }
}

void reference_impl(TestCase& test_case) {
  linear_reference_impl(test_case);
}

// FLOP calculator for linear operation
int64_t linear_flop_calculator(const TestCase& test_case) {
  const auto& input_sizes = test_case.inputs()[0].get_tensor_sizes();
  const auto& weight_sizes = test_case.inputs()[1].get_tensor_sizes();

  int64_t batch_size = input_sizes[0];
  int64_t in_features = input_sizes[1];
  int64_t out_features = weight_sizes[0];

  // Each output element requires in_features multiply-accumulate operations
  // Plus one add for bias (if present)
  int64_t output_elements = batch_size * out_features;
  int64_t ops_per_output = in_features; // MACs

  int64_t flop = output_elements * ops_per_output;

  return flop;
}

int main(int /* argc */, char* /* argv */[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(true);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "FP32/FP16 Linear Layer Benchmark" << std::endl;
  print_separator();

  // Query cooperative matrix properties to understand what's supported
  queryCooperativeMatrixProperties();

  ReferenceComputeFunc ref_fn = reference_impl;

  // Execute test cases using the framework with custom FLOP calculator
  auto results = execute_test_cases(
#ifdef DEBUG_MODE
      generate_linear_easy_cases,
#else
      generate_linear_test_cases,
#endif
      linear_flop_calculator,
      "FPLinear",
#ifdef DEBUG_MODE
      0,
      1,
#else
      5,
      40,
#endif
      ref_fn);

  return 0;
}
