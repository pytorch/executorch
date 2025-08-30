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

// Component structs for better readability
struct KernelSize {
  int32_t h;
  int32_t w;

  KernelSize(int32_t height, int32_t width) : h(height), w(width) {}
};

struct Stride {
  int32_t h;
  int32_t w;

  Stride(int32_t height, int32_t width) : h(height), w(width) {}
};

struct Padding {
  int32_t h;
  int32_t w;

  Padding(int32_t height, int32_t width) : h(height), w(width) {}
};

struct Dilation {
  int32_t h;
  int32_t w;

  Dilation(int32_t height = 1, int32_t width = 1) : h(height), w(width) {}
};

struct OutInChannels {
  int32_t out;
  int32_t in;

  OutInChannels(int32_t out_channels, int32_t in_channels)
      : out(out_channels), in(in_channels) {}
};

struct InputSize2D {
  int32_t h;
  int32_t w;

  InputSize2D(int32_t height, int32_t width) : h(height), w(width) {}
};

// Conv2d configuration struct
struct Conv2dConfig {
  OutInChannels channels;
  InputSize2D input_size;
  KernelSize kernel;
  Stride stride;
  Padding padding;
  Dilation dilation;
  int32_t groups; // Number of groups for grouped convolution
  std::string name_suffix;
  std::string shader_variant_name = "default";

  // Calculate output dimensions
  int64_t get_output_height() const {
    return (input_size.h + 2 * padding.h - dilation.h * (kernel.h - 1) - 1) /
        stride.h +
        1;
  }

  int64_t get_output_width() const {
    return (input_size.w + 2 * padding.w - dilation.w * (kernel.w - 1) - 1) /
        stride.w +
        1;
  }
};

// Utility function to create a test case from a Conv2dConfig
TestCase create_test_case_from_config(
    const Conv2dConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype) {
  TestCase test_case;

  // Create a descriptive name for the test case
  std::string storage_str =
      (storage_type == utils::kTexture3D) ? "Texture3D" : "Buffer";
  std::string dtype_str = (input_dtype == vkapi::kFloat) ? "Float" : "Half";

  std::string test_name =
      "Conv2d_" + config.name_suffix + "_" + storage_str + "_" + dtype_str;
  test_case.set_name(test_name);

  // Set the operator name for the test case
  std::string operator_name = "aten.convolution.";
  operator_name += config.shader_variant_name;
  test_case.set_operator_name(operator_name);

  // Calculate output dimensions
  int64_t H_out = config.get_output_height();
  int64_t W_out = config.get_output_width();

  // Input tensor (float/half) - [1, C_in, H_in, W_in] (batch size always 1)
  std::vector<int64_t> input_size = {
      1, config.channels.in, config.input_size.h, config.input_size.w};

  ValueSpec input_tensor(
      input_size,
      input_dtype,
      storage_type,
      utils::kChannelsPacked,
      DataGenType::RANDINT);

  if (debugging()) {
    print_valuespec_data(input_tensor, "input_tensor");
  }

  // Weight tensor (float/half) - [C_out, C_in, K_h, K_w]
  std::vector<int64_t> weight_size = {
      config.channels.out,
      config.channels.in,
      config.kernel.h,
      config.kernel.w};
  ValueSpec weight(
      weight_size,
      input_dtype,
      storage_type,
      utils::kChannelsPacked,
      DataGenType::RANDOM);
  weight.set_constant(true);

  if (debugging()) {
    print_valuespec_data(weight, "weight_tensor");
  }

  // Bias (optional, float/half) - [C_out]
  ValueSpec bias(
      {config.channels.out}, // Per output channel
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  bias.set_constant(true);

  // Stride and padding parameters
  ValueSpec stride({config.stride.h, config.stride.w});
  ValueSpec padding({config.padding.h, config.padding.w});

  // Dilation and groups parameters
  ValueSpec dilation({config.dilation.h, config.dilation.w});
  ValueSpec transposed{false};
  ValueSpec output_padding({0, 0});
  ValueSpec groups(config.groups);
  ValueSpec out_min{-1000.0f};
  ValueSpec out_max{-1000.0f};

  // Output tensor (float/half) - [1, C_out, H_out, W_out] (batch size always 1)
  ValueSpec output(
      {1, config.channels.out, H_out, W_out},
      input_dtype,
      storage_type,
      utils::kChannelsPacked,
      DataGenType::ZEROS);

  // Add all specs to test case
  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(weight);
  test_case.add_input_spec(bias);
  test_case.add_input_spec(stride);
  test_case.add_input_spec(padding);
  test_case.add_input_spec(dilation);
  test_case.add_input_spec(transposed);
  test_case.add_input_spec(output_padding);
  test_case.add_input_spec(groups);
  test_case.add_input_spec(out_min);
  test_case.add_input_spec(out_max);

  test_case.add_output_spec(output);

  return test_case;
}

// Generate easy test cases for conv2d operation (for debugging)
std::vector<TestCase> generate_conv2d_easy_cases() {
  std::vector<TestCase> test_cases;

  // Single simple configuration for debugging
  Conv2dConfig config = {
      OutInChannels(32, 3), // channels (out, in)
      InputSize2D(64, 64), // input_size (h, w)
      KernelSize(3, 3), // kernel
      Stride(2, 2), // stride
      Padding(1, 1), // padding
      Dilation(1, 1), // dilation
      1, // groups
      "simple" // descriptive name
  };

  // Test with both storage types and data types for completeness
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

// Generate test cases for conv2d operation
std::vector<TestCase> generate_conv2d_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<Conv2dConfig> configs = {// Performance test cases
                                       {OutInChannels(128, 64),
                                        InputSize2D(128, 128),
                                        KernelSize(3, 3),
                                        Stride(1, 1),
                                        Padding(1, 1),
                                        Dilation(1, 1),
                                        1,
                                        "perf"},
                                       {OutInChannels(256, 128),
                                        InputSize2D(128, 128),
                                        KernelSize(1, 1),
                                        Stride(1, 1),
                                        Padding(1, 1),
                                        Dilation(1, 1),
                                        8,
                                        "pw_perf"}};

  // Test with different storage types and data types
  std::vector<utils::StorageType> storage_types = {utils::kTexture3D};

  // Generate test cases for each combination
  for (const auto& config : configs) {
    for (const auto& storage_type : storage_types) {
      test_cases.push_back(
          create_test_case_from_config(config, storage_type, vkapi::kFloat));
      test_cases.push_back(
          create_test_case_from_config(config, storage_type, vkapi::kHalf));
    }
  }

  return test_cases;
}

// Custom FLOP calculator for conv2d operation
int64_t conv2d_flop_calculator(const TestCase& test_case) {
  if (test_case.num_inputs() < 7 || test_case.num_outputs() < 1) {
    return 0;
  }

  // Get input and weight dimensions
  const auto& input_sizes = test_case.inputs()[0].get_tensor_sizes();
  const auto& weight_sizes = test_case.inputs()[1].get_tensor_sizes();
  const auto& output_sizes = test_case.outputs()[0].get_tensor_sizes();

  int64_t N = input_sizes[0];
  int64_t C_in = input_sizes[1];
  int64_t C_out = weight_sizes[0];
  int64_t K_h = weight_sizes[2];
  int64_t K_w = weight_sizes[3];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];

  // Calculate FLOPs for conv2d operation
  // Each output element requires:
  // - C_in * K_h * K_w multiply-accumulate operations
  // - 1 bias addition
  int64_t output_elements = N * C_out * H_out * W_out;
  int64_t ops_per_output = C_in * K_h * K_w;

  // Add bias operation
  int64_t bias_ops = 1;

  int64_t flop = output_elements * (ops_per_output + bias_ops);

  return flop;
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Conv2d Operation Prototyping Framework" << std::endl;
  print_separator();

  // No reference function needed since fp32 convolutions are tested elsewhere
  ReferenceComputeFunc ref_fn = nullptr;

  // Execute test cases using the new framework with custom FLOP calculator
  auto results = execute_test_cases(
      generate_conv2d_test_cases,
      conv2d_flop_calculator,
      "Conv2d",
      0,
      1,
      ref_fn);

  return 0;
}
