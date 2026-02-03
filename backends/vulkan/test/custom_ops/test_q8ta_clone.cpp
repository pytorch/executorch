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

// #define DEBUG_MODE

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 512;

// Configuration struct for q8ta clone testing
struct Q8taCloneConfig {
  std::vector<int64_t> shape; // Tensor shape (can be any dimensionality)
  std::string test_case_name = "placeholder";
  std::string op_name = "q8ta_clone_test";
};

// Utility function to create a test case from a Q8taCloneConfig
TestCase create_test_case_from_config(
    const Q8taCloneConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype,
    utils::GPUMemoryLayout fp_memory_layout,
    utils::GPUMemoryLayout inp_quant_layout,
    utils::GPUMemoryLayout outp_quant_layout) {
  TestCase test_case;

  // Create a descriptive name for the test case
  std::string shape_str = shape_string(config.shape);
  std::string test_name = config.test_case_name + "  I=" + shape_str + "  " +
      repr_str(storage_type, fp_memory_layout) + "->" +
      repr_str(utils::kBuffer, inp_quant_layout) + "->" +
      repr_str(utils::kBuffer, outp_quant_layout);
  test_case.set_name(test_name);

  // Set the operator name for the test case
  std::string operator_name = "test_etvk." + config.op_name + ".default";
  test_case.set_operator_name(operator_name);

  // Input tensor (float) - any dimensionality
  ValueSpec input_tensor(
      config.shape,
      input_dtype,
      storage_type,
      fp_memory_layout,
      DataGenType::RANDOM);

  float scale_val = 0.007112;
  ValueSpec scale(scale_val);

  // Zero point for quantization
  int32_t zero_point_val = 0;
  ValueSpec zero_point(zero_point_val);

  // Input and output quantized layouts as integers
  int32_t inp_layout_int = static_cast<int32_t>(inp_quant_layout);
  ValueSpec inp_layout_spec(inp_layout_int);

  int32_t outp_layout_int = static_cast<int32_t>(outp_quant_layout);
  ValueSpec outp_layout_spec(outp_layout_int);

  // Output tensor (float) - same shape as input
  ValueSpec output_tensor(
      config.shape,
      input_dtype,
      storage_type,
      fp_memory_layout,
      DataGenType::ZEROS);

  // Add all specs to test case
  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(scale);
  test_case.add_input_spec(zero_point);
  test_case.add_input_spec(inp_layout_spec);
  test_case.add_input_spec(outp_layout_spec);
  test_case.add_output_spec(output_tensor);

  test_case.set_abs_tolerance(scale_val + 1e-4);

  // Use layout-only filter for this test since clone IS the operation being
  // tested
  test_case.set_shader_filter({
      "nchw_to",
      "to_nchw",
      "q8ta_quantize",
      "q8ta_dequantize",
  });

  return test_case;
}

// Generate easy test cases for q8ta_clone operation (for debugging)
std::vector<TestCase> generate_q8ta_clone_easy_cases() {
  std::vector<TestCase> test_cases;

  // Single simple configuration for debugging
  Q8taCloneConfig config = {
      {1, 16, 16, 16}, // shape: [N, C, H, W]
      "ACCU", // test_case_name
  };

  // FP memory layouts to test
  std::vector<utils::GPUMemoryLayout> fp_layouts = {
      utils::kWidthPacked,
      utils::kChannelsPacked,
  };

  // Quantized memory layouts to test
  std::vector<utils::GPUMemoryLayout> quant_layouts = {
      utils::kPackedInt8_4W,
      utils::kPackedInt8_4C,
      utils::kPackedInt8_4W4C,
      utils::kPackedInt8_4H4W,
      utils::kPackedInt8_4C1W,
  };

  std::vector<utils::StorageType> storage_types = {utils::kBuffer};
  std::vector<vkapi::ScalarType> float_types = {vkapi::kFloat};

  // Generate test cases for each combination (same layout for input and output)
  for (const auto& fp_layout : fp_layouts) {
    for (const auto& quant_layout : quant_layouts) {
      for (const auto& storage_type : storage_types) {
        for (const auto& input_dtype : float_types) {
          // Same layout: should be a simple copy
          test_cases.push_back(create_test_case_from_config(
              config,
              storage_type,
              input_dtype,
              fp_layout,
              quant_layout,
              quant_layout));
        }
      }
    }
  }

  return test_cases;
}

// Generate test cases for q8ta_clone operation
std::vector<TestCase> generate_q8ta_clone_test_cases() {
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

  // FP memory layouts to test
  std::vector<utils::GPUMemoryLayout> fp_layouts = {
      utils::kWidthPacked,
      utils::kChannelsPacked,
  };

  // Quantized memory layouts to test
  std::vector<utils::GPUMemoryLayout> quant_layouts = {
      utils::kPackedInt8_4W,
      utils::kPackedInt8_4C,
      utils::kPackedInt8_4W4C,
      utils::kPackedInt8_4H4W,
      utils::kPackedInt8_4C1W,
  };

  // Test with buffer storage only
  std::vector<utils::StorageType> storage_types = {utils::kBuffer};

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

    for (const auto& fp_layout : fp_layouts) {
      for (const auto& inp_quant_layout : quant_layouts) {
        for (const auto& outp_quant_layout : quant_layouts) {
          for (const auto& storage_type : storage_types) {
            Q8taCloneConfig config;
            config.shape = shape;
            config.test_case_name = prefix;

            test_cases.push_back(create_test_case_from_config(
                config,
                storage_type,
                vkapi::kFloat,
                fp_layout,
                inp_quant_layout,
                outp_quant_layout));
          }
        }
      }
    }
  }

  return test_cases;
}

// Reference implementation for q8ta_clone operation
// Since clone just copies data, the result should be the same as
// quantize-dequantize
void q8ta_clone_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& scale_spec = test_case.inputs()[idx++];
  const ValueSpec& zero_point_spec = test_case.inputs()[idx++];
  const ValueSpec& inp_layout_spec = test_case.inputs()[idx++];
  const ValueSpec& outp_layout_spec = test_case.inputs()[idx++];
  (void)inp_layout_spec; // Not used in reference implementation
  (void)outp_layout_spec; // Not used in reference implementation

  // Extract output specification
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions (arbitrary dimensionality)
  auto input_sizes = input_spec.get_tensor_sizes();

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

  if (input_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  // Get raw data pointers
  auto& input_data = input_spec.get_float_data();

  // Extract the randomized scale and zero point values
  float scale = scale_spec.get_float_value();
  int32_t zero_point = zero_point_spec.get_int_value();
  int32_t quant_min = -128;
  int32_t quant_max = 127;

  // Prepare output data
  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_elements);

  // Perform quantize-clone-dequantize operation on each element
  // Clone preserves the quantized values, so result is same as Q-DQ
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
#ifdef DEBUG_MODE
  set_print_latencies(false);
#else
  set_print_latencies(false);
#endif
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Q8TA Clone Operation Prototyping Framework" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = q8ta_clone_reference_impl;

  auto results = execute_test_cases(
#ifdef DEBUG_MODE
      generate_q8ta_clone_easy_cases,
#else
      generate_q8ta_clone_test_cases,
#endif
      "Q8taClone",
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
