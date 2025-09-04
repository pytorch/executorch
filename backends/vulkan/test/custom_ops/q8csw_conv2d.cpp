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

static constexpr int64_t kRefDimSizeLimit = 100;

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
  std::string test_case_name = "placeholder";
  std::string op_name = "conv2d_q8ta_q8csw";

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
      config.test_case_name + "_" + storage_str + "_" + dtype_str;
  test_case.set_name(test_name);

  // Set the operator name for the test case
  std::string operator_name = "et_vk." + config.op_name + ".default";
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
      DataGenType::RANDOM);

  if (debugging()) {
    print_valuespec_data(input_tensor, "input_tensor");
  }

  float input_scale_val = 0.07f;
  ValueSpec input_scale(input_scale_val);

  int32_t input_zero_point_val = -3;
  ValueSpec input_zero_point(input_zero_point_val);

  // Quantized weight tensor (int8) - [C_out, C_in_per_group * K_h * K_w]
  // Memory layout: height, width, then channels - in_c is innermost (stride 1)
  // in the second dimension
  const int64_t in_channels_per_group = config.channels.in / config.groups;
  const int64_t in_features = utils::align_up_4(
      in_channels_per_group * config.kernel.h * config.kernel.w);
  std::vector<int64_t> weight_size = {config.channels.out, in_features};
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
      {config.channels.out}, // Per output channel
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  weight_scales.set_constant(true);

  ValueSpec weight_sums(
      {config.channels.out}, // Per output channel
      vkapi::kFloat,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  weight_sums.set_constant(true);

  // Compute weight_sums data based on quantized weights
  compute_weight_sums(
      weight_sums, quantized_weight, config.channels.out, in_features);

  // Bias (optional, float/half) - [C_out]
  ValueSpec bias(
      {config.channels.out}, // Per output channel
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM);
  bias.set_constant(true);

  // Stride and padding parameters
  ValueSpec stride({config.stride.h, config.stride.w});
  ValueSpec padding({config.padding.h, config.padding.w});

  // Dilation and groups parameters
  ValueSpec dilation({config.dilation.h, config.dilation.w});
  ValueSpec groups(config.groups);

  // Kernel size parameters
  ValueSpec kernel_size({config.kernel.h, config.kernel.w});

  // Output tensor (float/half) - [1, C_out, H_out, W_out] (batch size always 1)
  ValueSpec output(
      {1, config.channels.out, H_out, W_out},
      input_dtype,
      storage_type,
      utils::kChannelsPacked,
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
    test_case.add_input_spec(kernel_size);
    test_case.add_input_spec(stride);
    test_case.add_input_spec(padding);
    test_case.add_input_spec(dilation);
    test_case.add_input_spec(groups);
  } else {
    test_case.add_input_spec(input_tensor);
    test_case.add_input_spec(quantized_weight);
    test_case.add_input_spec(weight_scales);
    test_case.add_input_spec(bias);
    test_case.add_input_spec(kernel_size);
    test_case.add_input_spec(stride);
    test_case.add_input_spec(padding);
    test_case.add_input_spec(dilation);
    test_case.add_input_spec(groups);
  }

  test_case.add_output_spec(output);

  return test_case;
}

// Generate easy test cases for quantized conv2d operation (for debugging)
std::vector<TestCase> generate_quantized_conv2d_easy_cases() {
  std::vector<TestCase> test_cases;

  // Single simple configuration for debugging
  Conv2dConfig config = {
      OutInChannels(8, 3), // channels (out, in)
      InputSize2D(8, 8), // input_size (h, w)
      KernelSize(3, 3), // kernel
      Stride(1, 1), // stride
      Padding(0, 0), // padding
      Dilation(1, 1), // dilation
      1, // groups
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

// Generate test cases for quantized conv2d operation
std::vector<TestCase> generate_quantized_conv2d_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<Conv2dConfig> configs = {
      {OutInChannels(32, 3),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(32, 16),
       InputSize2D(32, 32),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 32),
       InputSize2D(16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      // One output channel case
      {OutInChannels(1, 32),
       InputSize2D(55, 55),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},

      // Stride 2 convolutions
      {OutInChannels(32, 3),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 32),
       InputSize2D(32, 32),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      // Different kernel sizes
      {OutInChannels(32, 16),
       InputSize2D(28, 28),
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 32),
       InputSize2D(14, 14),
       KernelSize(7, 7),
       Stride(1, 1),
       Padding(3, 3),
       Dilation(1, 1),
       1},

      // Dilated convolutions
      {OutInChannels(32, 16),
       InputSize2D(32, 32),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(2, 2),
       1},
      {OutInChannels(64, 32),
       InputSize2D(16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(3, 3),
       Dilation(3, 3),
       1},

      // Grouped convolutions
      {OutInChannels(32, 32),
       InputSize2D(32, 32),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       4},
      {OutInChannels(64, 64),
       InputSize2D(16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       8},
      // Performance test cases
      {OutInChannels(256, 128),
       InputSize2D(128, 128),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       8},
      {OutInChannels(128, 64),
       InputSize2D(128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1}};

  // Test with different storage types and data types
  std::vector<utils::StorageType> storage_types = {utils::kTexture3D};

  // Generate test cases for each combination
  for (auto& config : configs) {
    for (const auto& storage_type : storage_types) {
      // Generate test case name programmatically
      bool is_performance = config.channels.out > kRefDimSizeLimit ||
          config.channels.in > kRefDimSizeLimit ||
          config.input_size.h > kRefDimSizeLimit ||
          config.input_size.w > kRefDimSizeLimit;
      std::string prefix = is_performance ? "performance_" : "correctness_";
      std::string suffix = std::to_string(config.channels.out) + "/" +
          std::to_string(config.channels.in) + "_" +
          std::to_string(config.input_size.h) + "/" +
          std::to_string(config.input_size.w) + "_" +
          std::to_string(config.kernel.h) + "/" +
          std::to_string(config.kernel.w);

      config.test_case_name = prefix + suffix;
      // The default operator tested is activation + weight quantized conv2d;
      // however, only test this if the int8 dot product extension is supported
      if (vkcompute::api::context()
              ->adapter_ptr()
              ->supports_int8_dot_product()) {
        test_cases.push_back(
            create_test_case_from_config(config, storage_type, vkapi::kFloat));
      }

      Conv2dConfig wo_quant_config = config;
      wo_quant_config.op_name = "conv2d_q8csw";
      test_cases.push_back(create_test_case_from_config(
          wo_quant_config, storage_type, vkapi::kFloat));
    }
  }

  return test_cases;
}

// Reference implementation for weight only quantized conv2d (fp accumulation)
void conv2d_q8csw_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_scales_spec = test_case.inputs()[idx++];
  const ValueSpec& bias_spec = test_case.inputs()[idx++];
  const ValueSpec& kernel_size_spec = test_case.inputs()[idx++];
  const ValueSpec& stride_spec = test_case.inputs()[idx++];
  const ValueSpec& padding_spec = test_case.inputs()[idx++];
  const ValueSpec& dilation_spec = test_case.inputs()[idx++];
  const ValueSpec& groups_spec = test_case.inputs()[idx++];

  // Extract output specification (mutable reference)
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions
  auto input_sizes = input_spec.get_tensor_sizes(); // [N, C_in, H_in, W_in]
  auto weight_sizes =
      weight_spec.get_tensor_sizes(); // [C_out, C_in_per_group * K_h * K_w]
  auto output_sizes =
      output_spec.get_tensor_sizes(); // [N, C_out, H_out, W_out]

  int64_t N = input_sizes[0];
  int64_t C_in = input_sizes[1];
  int64_t H_in = input_sizes[2];
  int64_t W_in = input_sizes[3];
  int64_t C_out = output_sizes[1];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];

  // Get kernel dimensions from kernel_size ValueSpec
  auto kernel_size_data = kernel_size_spec.get_int32_data();
  int64_t K_h = kernel_size_data[0];
  int64_t K_w = kernel_size_data[1];

  // Get stride, padding, dilation, and groups
  auto stride_data = stride_spec.get_int32_data();
  auto padding_data = padding_spec.get_int32_data();
  auto dilation_data = dilation_spec.get_int32_data();
  int64_t stride_h = stride_data[0];
  int64_t stride_w = stride_data[1];
  int64_t pad_h = padding_data[0];
  int64_t pad_w = padding_data[1];
  int64_t dilation_h = dilation_data[0];
  int64_t dilation_w = dilation_data[1];
  int64_t groups = groups_spec.get_int_value();

  // Skip for large tensors since computation time will be extremely slow
  if (N > kRefDimSizeLimit || C_in > kRefDimSizeLimit ||
      H_in > kRefDimSizeLimit || W_in > kRefDimSizeLimit ||
      C_out > kRefDimSizeLimit) {
    throw std::invalid_argument(
        "One or more dimensions exceed the allowed limit for reference implementation.");
  }

  if (input_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  // Get raw data pointers
  auto& input_data = input_spec.get_float_data();
  auto& weight_data = weight_spec.get_int8_data();
  auto& weight_scales_data = weight_scales_spec.get_float_data();
  auto& bias_data = bias_spec.get_float_data();

  // Calculate channels per group for grouped convolution
  int64_t C_in_per_group = C_in / groups;
  int64_t C_out_per_group = C_out / groups;

  // Calculate number of output elements
  int64_t num_output_elements = N * C_out * H_out * W_out;

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_output_elements);

  const int in_features = utils::align_up_4(C_in_per_group * K_h * K_w);

  // Perform weight-only quantized conv2d operation (fp accumulation)
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t out_c = 0; out_c < C_out; ++out_c) {
      for (int64_t out_h = 0; out_h < H_out; ++out_h) {
        for (int64_t out_w = 0; out_w < W_out; ++out_w) {
          float sum = 0.0f;

          // Determine which group this output channel belongs to
          int64_t group_idx = out_c / C_out_per_group;
          int64_t in_c_start = group_idx * C_in_per_group;
          int64_t in_c_end = (group_idx + 1) * C_in_per_group;

          // Convolution operation with dilation support and grouped convolution
          for (int64_t in_c = in_c_start; in_c < in_c_end; ++in_c) {
            for (int64_t kh = 0; kh < K_h; ++kh) {
              for (int64_t kw = 0; kw < K_w; ++kw) {
                // Calculate input position with dilation
                int64_t in_h = out_h * stride_h - pad_h + kh * dilation_h;
                int64_t in_w = out_w * stride_w - pad_w + kw * dilation_w;

                // Check bounds (zero padding)
                if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
                  // Get input value (keep as float)
                  int64_t input_idx = n * (C_in * H_in * W_in) +
                      in_c * (H_in * W_in) + in_h * W_in + in_w;
                  float input_val = input_data[input_idx];

                  // Get weight value and dequantize
                  // Weight layout: [C_out, C_in_per_group * K_h * K_w]
                  int64_t weight_idx = out_c * in_features +
                      (kh * (K_w * C_in_per_group) + kw * C_in_per_group +
                       (in_c % C_in_per_group));
                  float dequant_weight =
                      (static_cast<float>(weight_data[weight_idx])) *
                      weight_scales_data[out_c];

                  sum += input_val * dequant_weight;
                }
              }
            }
          }

          // Add bias and store result
          sum += bias_data[out_c];
          int64_t output_idx = n * (C_out * H_out * W_out) +
              out_c * (H_out * W_out) + out_h * W_out + out_w;
          ref_data[output_idx] = sum;
        }
      }
    }
  }
}

// Reference implementation for activation and weight quantized conv2d (int
// accumulation)
void conv2d_q8ta_q8csw_reference_impl(TestCase& test_case) {
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
  const ValueSpec& kernel_size_spec = test_case.inputs()[idx++];
  const ValueSpec& stride_spec = test_case.inputs()[idx++];
  const ValueSpec& padding_spec = test_case.inputs()[idx++];
  const ValueSpec& dilation_spec = test_case.inputs()[idx++];
  const ValueSpec& groups_spec = test_case.inputs()[idx++];

  // Extract output specification (mutable reference)
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions
  auto input_sizes = input_spec.get_tensor_sizes(); // [N, C_in, H_in, W_in]
  auto weight_sizes =
      weight_spec.get_tensor_sizes(); // [C_out, C_in_per_group * K_h * K_w]
  auto output_sizes =
      output_spec.get_tensor_sizes(); // [N, C_out, H_out, W_out]

  int64_t N = input_sizes[0];
  int64_t C_in = input_sizes[1];
  int64_t H_in = input_sizes[2];
  int64_t W_in = input_sizes[3];
  int64_t C_out = output_sizes[1];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];

  // Get kernel dimensions from kernel_size ValueSpec
  auto kernel_size_data = kernel_size_spec.get_int32_data();
  int64_t K_h = kernel_size_data[0];
  int64_t K_w = kernel_size_data[1];

  // Get stride, padding, dilation, and groups
  auto stride_data = stride_spec.get_int32_data();
  auto padding_data = padding_spec.get_int32_data();
  auto dilation_data = dilation_spec.get_int32_data();
  int64_t stride_h = stride_data[0];
  int64_t stride_w = stride_data[1];
  int64_t pad_h = padding_data[0];
  int64_t pad_w = padding_data[1];
  int64_t dilation_h = dilation_data[0];
  int64_t dilation_w = dilation_data[1];
  int64_t groups = groups_spec.get_int_value();

  // Skip for large tensors since computation time will be extremely slow
  if (N > kRefDimSizeLimit || C_in > kRefDimSizeLimit ||
      H_in > kRefDimSizeLimit || W_in > kRefDimSizeLimit ||
      C_out > kRefDimSizeLimit) {
    throw std::invalid_argument(
        "One or more dimensions exceed the allowed limit for reference implementation.");
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

  // Calculate channels per group for grouped convolution
  int64_t C_in_per_group = C_in / groups;
  int64_t C_out_per_group = C_out / groups;

  // Calculate number of output elements
  int64_t num_output_elements = N * C_out * H_out * W_out;

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_output_elements);

  const int in_features = utils::align_up_4(C_in_per_group * K_h * K_w);

  // Perform activation and weight quantized conv2d operation (int accumulation)
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t out_c = 0; out_c < C_out; ++out_c) {
      for (int64_t out_h = 0; out_h < H_out; ++out_h) {
        for (int64_t out_w = 0; out_w < W_out; ++out_w) {
          int32_t int_sum = 0;
          int32_t weight_sum = 0; // Track weight sum on the fly

          // Determine which group this output channel belongs to
          int64_t group_idx = out_c / C_out_per_group;
          int64_t in_c_start = group_idx * C_in_per_group;
          int64_t in_c_end = (group_idx + 1) * C_in_per_group;

          // Convolution operation with integer accumulation
          for (int64_t in_c = in_c_start; in_c < in_c_end; ++in_c) {
            for (int64_t kh = 0; kh < K_h; ++kh) {
              for (int64_t kw = 0; kw < K_w; ++kw) {
                // Calculate input position with dilation
                int64_t in_h = out_h * stride_h - pad_h + kh * dilation_h;
                int64_t in_w = out_w * stride_w - pad_w + kw * dilation_w;

                // Check bounds (zero padding)
                if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
                  // Get input value and quantize to int8
                  int64_t input_idx = n * (C_in * H_in * W_in) +
                      in_c * (H_in * W_in) + in_h * W_in + in_w;

                  float quant_input_f =
                      std::round(input_data[input_idx] / input_scale) +
                      input_zero_point;
                  quant_input_f =
                      std::min(std::max(quant_input_f, -128.0f), 127.0f);
                  int8_t quantized_input = static_cast<int8_t>(quant_input_f);

                  // Get quantized weight (already int8)
                  // Weight layout: [C_out, C_in_per_group * K_h * K_w]
                  int64_t weight_idx = out_c * in_features +
                      (kh * (K_w * C_in_per_group) + kw * C_in_per_group +
                       (in_c % C_in_per_group));
                  int8_t quantized_weight = weight_data[weight_idx];

                  // Integer multiplication and accumulation
                  int_sum += static_cast<int32_t>(quantized_input) *
                      static_cast<int32_t>(quantized_weight);

                  // Track weight sum for this output channel on the fly
                  weight_sum += static_cast<int32_t>(quantized_weight);
                } else {
                  // For zero padding, we still need to account for the weight
                  // in weight_sum when input is effectively 0 (but quantized 0
                  // is input_zero_point)
                  int64_t weight_idx = out_c * in_features +
                      (kh * (K_w * C_in_per_group) + kw * C_in_per_group +
                       (in_c % C_in_per_group));
                  int8_t quantized_weight = weight_data[weight_idx];

                  // Add contribution from zero-padded input (quantized zero =
                  // input_zero_point)
                  int_sum += static_cast<int32_t>(input_zero_point) *
                      static_cast<int32_t>(quantized_weight);

                  // Track weight sum for this output channel on the fly
                  weight_sum += static_cast<int32_t>(quantized_weight);
                }
              }
            }
          }

          // Convert accumulated integer result to float and apply scales
          // Final result = (int_sum - zero_point_correction) * input_scale *
          // weight_scale + bias zero_point_correction = input_zero_point *
          // sum_of_weights_for_this_output_channel
          int32_t zero_point_correction = input_zero_point * weight_sum;
          float float_result = (static_cast<float>(int_sum) -
                                static_cast<float>(zero_point_correction)) *
              input_scale * weight_scales_data[out_c];

          // Add bias and store result
          float_result += bias_data[out_c];
          int64_t output_idx = n * (C_out * H_out * W_out) +
              out_c * (H_out * W_out) + out_h * W_out + out_w;
          ref_data[output_idx] = float_result;
        }
      }
    }
  }
}

void reference_impl(TestCase& test_case) {
  if (test_case.operator_name().find("q8ta") != std::string::npos) {
    conv2d_q8ta_q8csw_reference_impl(test_case);
  } else {
    conv2d_q8csw_reference_impl(test_case);
  }
}

// Custom FLOP calculator for quantized conv2d operation
int64_t quantized_conv2d_flop_calculator(const TestCase& test_case) {
  int kernel_idx = 4;
  if (test_case.operator_name().find("q8ta") != std::string::npos) {
    kernel_idx = 7;
  }
  // Get input and weight dimensions
  const auto& input_sizes = test_case.inputs()[0].get_tensor_sizes();
  const auto& output_sizes = test_case.outputs()[0].get_tensor_sizes();

  const auto& kernel_sizes = test_case.inputs()[kernel_idx].get_int32_data();

  int64_t N = input_sizes[0];
  int64_t C_in = input_sizes[1];
  int64_t C_out = output_sizes[1];
  int64_t K_h = kernel_sizes[0];
  int64_t K_w = kernel_sizes[1];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];

  // Calculate FLOPs for quantized conv2d operation
  // Each output element requires:
  // - C_in * K_h * K_w multiply-accumulate operations
  // - Additional operations for quantization/dequantization
  int64_t output_elements = N * C_out * H_out * W_out;
  int64_t ops_per_output = C_in * K_h * K_w;

  int64_t flop = output_elements * (ops_per_output);

  return flop;
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Quantized Conv2d Operation Prototyping Framework" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  // Execute test cases using the new framework with custom FLOP calculator
  auto results = execute_test_cases(
      generate_quantized_conv2d_test_cases,
      quantized_conv2d_flop_calculator,
      "QuantizedConv2d",
      0,
      10,
      ref_fn);

  return 0;
}
