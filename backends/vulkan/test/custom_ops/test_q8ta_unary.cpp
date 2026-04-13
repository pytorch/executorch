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

struct Q8taUnaryConfig {
  std::vector<int64_t> shape;
  std::string test_case_name = "placeholder";
  std::string op_name = "q8ta_unary_test";
};

TestCase create_test_case_from_config(
    const Q8taUnaryConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype,
    utils::GPUMemoryLayout fp_memory_layout,
    utils::GPUMemoryLayout quant_layout) {
  TestCase test_case;

  std::string shape_str = shape_string(config.shape);
  std::string test_name = config.test_case_name + "  I=" + shape_str + "  " +
      repr_str(storage_type, fp_memory_layout) + "->" +
      repr_str(utils::kBuffer, quant_layout);
  test_case.set_name(test_name);

  std::string operator_name = "test_etvk." + config.op_name + ".default";
  test_case.set_operator_name(operator_name);

  // Input tensor (float)
  ValueSpec input_tensor(
      config.shape,
      input_dtype,
      storage_type,
      fp_memory_layout,
      DataGenType::RANDOM);

  float scale_val = 0.007112;
  ValueSpec input_scale(scale_val);

  int32_t zero_point_val = 0;
  ValueSpec input_zero_point(zero_point_val);

  // For relu, output scale and zero point can differ from input
  float output_scale_val = 0.007112;
  ValueSpec output_scale(output_scale_val);

  int32_t output_zp_val = 0;
  ValueSpec output_zero_point(output_zp_val);

  int32_t layout_int = static_cast<int32_t>(quant_layout);
  ValueSpec layout_spec(layout_int);

  // Output tensor (float) - same shape as input
  ValueSpec output_tensor(
      config.shape,
      input_dtype,
      storage_type,
      fp_memory_layout,
      DataGenType::ZEROS);

  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(input_scale);
  test_case.add_input_spec(input_zero_point);
  test_case.add_input_spec(output_scale);
  test_case.add_input_spec(output_zero_point);
  test_case.add_input_spec(layout_spec);
  test_case.add_output_spec(output_tensor);

  test_case.set_abs_tolerance(scale_val + 1e-4);

  test_case.set_shader_filter({
      "nchw_to",
      "to_nchw",
      "q8ta_quantize",
      "q8ta_dequantize",
  });

  return test_case;
}

std::vector<TestCase> generate_q8ta_unary_easy_cases() {
  std::vector<TestCase> test_cases;

  Q8taUnaryConfig config = {
      {1, 16, 16, 16},
      "ACCU",
  };

  std::vector<utils::GPUMemoryLayout> fp_layouts = {
      utils::kWidthPacked,
      utils::kChannelsPacked,
  };

  std::vector<utils::GPUMemoryLayout> quant_layouts = {
      utils::kPackedInt8_4W,
      utils::kPackedInt8_4C,
      utils::kPackedInt8_4W4C,
      utils::kPackedInt8_4H4W,
      utils::kPackedInt8_4C1W,
  };

  std::vector<utils::StorageType> storage_types = {utils::kBuffer};
  std::vector<vkapi::ScalarType> float_types = {vkapi::kFloat};

  for (const auto& fp_layout : fp_layouts) {
    for (const auto& quant_layout : quant_layouts) {
      for (const auto& storage_type : storage_types) {
        for (const auto& input_dtype : float_types) {
          test_cases.push_back(create_test_case_from_config(
              config, storage_type, input_dtype, fp_layout, quant_layout));
        }
      }
    }
  }

  return test_cases;
}

std::vector<TestCase> generate_q8ta_unary_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<std::vector<int64_t>> shapes = {
      {1, 3, 16, 16},
      {1, 8, 32, 32},
      {1, 16, 24, 24},
      {1, 32, 12, 12},
      {1, 1, 64, 64},
      {1, 3, 64, 64},
      {1, 4, 16, 16},

      {1, 8, 20, 20},
      {1, 16, 14, 14},
      {1, 8, 28, 28},

      // Odd tensor sizes
      {1, 3, 15, 15},
      {1, 13, 31, 31},
      {1, 17, 23, 23},

      // Larger tensors
      {1, 64, 128, 128},
      {1, 32, 64, 64},
      {1, 128, 56, 56},
      {1, 128, 128, 128},
  };

  std::vector<utils::GPUMemoryLayout> fp_layouts = {
      utils::kWidthPacked,
      utils::kChannelsPacked,
  };

  std::vector<utils::GPUMemoryLayout> quant_layouts = {
      utils::kPackedInt8_4W,
      utils::kPackedInt8_4C,
      utils::kPackedInt8_4W4C,
      utils::kPackedInt8_4H4W,
      utils::kPackedInt8_4C1W,
  };

  std::vector<utils::StorageType> storage_types = {utils::kBuffer};

  for (const auto& shape : shapes) {
    std::string prefix = "ACCU";
    for (const auto& dim : shape) {
      if (dim > kRefDimSizeLimit) {
        prefix = "PERF";
        break;
      }
    }

    for (const auto& fp_layout : fp_layouts) {
      for (const auto& quant_layout : quant_layouts) {
        for (const auto& storage_type : storage_types) {
          Q8taUnaryConfig config;
          config.shape = shape;
          config.test_case_name = prefix;

          test_cases.push_back(create_test_case_from_config(
              config, storage_type, vkapi::kFloat, fp_layout, quant_layout));
        }
      }
    }
  }

  return test_cases;
}

// Reference implementation: quantize -> relu -> dequantize
void q8ta_unary_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& input_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& input_zp_spec = test_case.inputs()[idx++];
  const ValueSpec& output_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& output_zp_spec = test_case.inputs()[idx++];
  const ValueSpec& layout_spec = test_case.inputs()[idx++];
  (void)layout_spec;

  ValueSpec& output_spec = test_case.outputs()[0];

  auto input_sizes = input_spec.get_tensor_sizes();

  int64_t num_elements = 1;
  for (const auto& dim : input_sizes) {
    num_elements *= dim;
  }

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

  auto& input_data = input_spec.get_float_data();

  float input_scale = input_scale_spec.get_float_value();
  int32_t input_zp = input_zp_spec.get_int_value();
  float output_scale = output_scale_spec.get_float_value();
  int32_t output_zp = output_zp_spec.get_int_value();
  int32_t quant_min = -128;
  int32_t quant_max = 127;

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_elements);

  for (int64_t i = 0; i < num_elements; ++i) {
    float input_val = input_data[i];

    // Quantize with input scale/zp
    float quantized_float = std::round(input_val / input_scale) + input_zp;
    quantized_float = std::max(quantized_float, static_cast<float>(quant_min));
    quantized_float = std::min(quantized_float, static_cast<float>(quant_max));
    int32_t quantized_int = static_cast<int32_t>(quantized_float);

    // Dequantize to float
    float dequantized = (quantized_int - input_zp) * input_scale;

    // Apply ReLU
    float activated = std::max(dequantized, 0.0f);

    // Requantize with output scale/zp
    float requantized_float = std::round(activated / output_scale) + output_zp;
    requantized_float =
        std::max(requantized_float, static_cast<float>(quant_min));
    requantized_float =
        std::min(requantized_float, static_cast<float>(quant_max));
    int32_t requantized_int = static_cast<int32_t>(requantized_float);

    // Dequantize back to float for comparison
    ref_data[i] = (requantized_int - output_zp) * output_scale;
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
  std::cout << "Q8TA Unary (ReLU) Operation Prototyping Framework" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = q8ta_unary_reference_impl;

  auto results = execute_test_cases(
#ifdef DEBUG_MODE
      generate_q8ta_unary_easy_cases,
#else
      generate_q8ta_unary_test_cases,
#endif
      "Q8taUnary",
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
