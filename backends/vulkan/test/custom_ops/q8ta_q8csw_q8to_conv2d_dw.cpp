// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <iostream>
#include <vector>
#include "conv2d_utils.h"
#include "utils.h"

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

using namespace executorch::vulkan::prototyping;

using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 100;

// Utility function to create a test case from a Conv2dConfig for depthwise
// convolution
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
  std::string operator_name = "etvk." + config.op_name + ".test";
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
    print_valuespec_data(input_tensor, "input_tensor", false, 64);
  }

  float input_scale_val = 0.008123;
  ValueSpec input_scale(input_scale_val);

  int32_t input_zero_point_val = 2;
  ValueSpec input_zero_point(input_zero_point_val);

  // Quantized weight tensor (int8) for depthwise convolution
  // Memory layout: [K_h, K_w, OC]
  // For depthwise conv: groups = channels.out, in_channels_per_group = 1
  std::vector<int64_t> weight_size = {
      config.kernel.h, config.kernel.w, config.channels.out};
  ValueSpec quantized_weight(
      weight_size,
      vkapi::kChar, // int8 for quantized weights
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDINT8);
  quantized_weight.set_constant(true);

  if (debugging()) {
    print_valuespec_data(quantized_weight, "weight_tensor", false, 64);
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
      vkapi::kInt,
      storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  weight_sums.set_constant(true);

  // Compute weight_sums data based on quantized weights for depthwise layout
  // For depthwise conv: each output channel has K_h * K_w weights
  // Custom computation for depthwise layout [K_h, K_w, OC]
  auto& weight_sums_data = weight_sums.get_int32_data();
  auto& quantized_weight_data = quantized_weight.get_int8_data();

  weight_sums_data.resize(config.channels.out);

  for (int64_t out_c = 0; out_c < config.channels.out; ++out_c) {
    int32_t sum = 0;
    for (int64_t kh = 0; kh < config.kernel.h; ++kh) {
      for (int64_t kw = 0; kw < config.kernel.w; ++kw) {
        // Weight indexing for depthwise layout [K_h, K_w, OC]
        int64_t weight_idx = kh * (config.kernel.w * config.channels.out) +
            kw * config.channels.out + out_c;
        sum += static_cast<int32_t>(quantized_weight_data[weight_idx]);
      }
    }
    weight_sums_data[out_c] = sum;
  }

  // Bias (optional, float/half) - [C_out]
  ValueSpec bias(
      {config.channels.out}, // Per output channel
      input_dtype,
      storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM);
  bias.set_constant(true);

  // Output quantization parameters
  float output_scale_val = 0.05314;
  ValueSpec output_scale(output_scale_val);

  int32_t output_zero_point_val = -1;
  ValueSpec output_zero_point(output_zero_point_val);

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

  // Add all specs to test case for q8ta_q8csw_q8to operation
  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(input_scale);
  test_case.add_input_spec(input_zero_point);
  test_case.add_input_spec(quantized_weight);
  test_case.add_input_spec(weight_sums);
  test_case.add_input_spec(weight_scales);
  test_case.add_input_spec(output_scale);
  test_case.add_input_spec(output_zero_point);
  test_case.add_input_spec(bias);
  test_case.add_input_spec(kernel_size);
  test_case.add_input_spec(stride);
  test_case.add_input_spec(padding);
  test_case.add_input_spec(dilation);
  test_case.add_input_spec(groups);

  test_case.add_output_spec(output);

  test_case.set_abs_tolerance(output_scale_val + 1e-4f);

  return test_case;
}

// Generate easy test cases for quantized depthwise conv2d operation (for
// debugging)
std::vector<TestCase> generate_quantized_conv2d_dw_easy_cases() {
  std::vector<TestCase> test_cases;

  // Single simple configuration for debugging - depthwise convolution
  Conv2dConfig config = {
      OutInChannels(8, 8), // channels (out, in) - equal for depthwise
      InputSize2D(8, 8), // input_size (h, w)
      KernelSize(3, 3), // kernel
      Stride(2, 2), // stride
      Padding(1, 1), // padding
      Dilation(1, 1), // dilation
      8, // groups = channels.out for depthwise
  };
  config.op_name = "conv2d_q8ta_q8csw_q8to";

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

// Generate test cases for quantized depthwise conv2d operation
std::vector<TestCase> generate_quantized_conv2d_dw_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<Conv2dConfig> configs = {
      // Depthwise convolutions: groups = channels.out, channels.in =
      // channels.out
      {OutInChannels(32, 32),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       32},
      {OutInChannels(64, 64),
       InputSize2D(32, 32),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(2, 2),
       Dilation(1, 1),
       64},
      {OutInChannels(64, 64),
       InputSize2D(32, 32),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       64},
      {OutInChannels(80, 80),
       InputSize2D(16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       80},
      {OutInChannels(16, 16),
       InputSize2D(57, 33),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       16},
      // Different kernel sizes for depthwise
      {OutInChannels(32, 32),
       InputSize2D(64, 64),
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(1, 1),
       32},
      {OutInChannels(96, 96),
       InputSize2D(64, 64),
       KernelSize(5, 5),
       Stride(2, 2),
       Padding(2, 2),
       Dilation(1, 1),
       96},
      // Performance cases
      {OutInChannels(128, 128),
       InputSize2D(128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       128},
      {OutInChannels(64, 64),
       InputSize2D(256, 256),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       64},
      {OutInChannels(288, 288),
       InputSize2D(16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       288},
      {OutInChannels(32, 32),
       InputSize2D(128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(1, 1),
       32}};

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
      std::string prefix =
          is_performance ? "performance_dw_" : "correctness_dw_";
      std::string suffix = std::to_string(config.channels.out) + "/" +
          std::to_string(config.channels.in) + "_" +
          std::to_string(config.input_size.h) + "/" +
          std::to_string(config.input_size.w) + "_" +
          std::to_string(config.kernel.h) + "/" +
          std::to_string(config.kernel.w);

      config.op_name = "conv2d_q8ta_q8csw_q8to";
      config.test_case_name = prefix + suffix;

      // Only test q8ta_q8csw_q8to if the int8 dot product extension is
      // supported
      if (vkcompute::api::context()
              ->adapter_ptr()
              ->supports_int8_dot_product()) {
        test_cases.push_back(
            create_test_case_from_config(config, storage_type, vkapi::kFloat));
      }
    }
  }

  return test_cases;
}

// Reference implementation for activation, weight, and output quantized
// depthwise conv2d
void conv2d_q8ta_q8csw_q8to_dw_reference_impl(TestCase& test_case) {
  // Extract input specifications
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
      weight_spec.get_tensor_sizes(); // [K_h, align_up_4(K_w), OC]
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

  // Verify this is a depthwise convolution
  if (groups != C_out || C_in != C_out) {
    throw std::invalid_argument(
        "This is not a depthwise convolution configuration");
  }

  // Get raw data pointers
  auto& input_data = input_spec.get_float_data();
  const float input_scale = input_scale_spec.get_float_value();
  const int32_t input_zero_point = input_zeros_spec.get_int_value();

  auto& weight_data = weight_spec.get_int8_data();
  auto& weight_scales_data = weight_scales_spec.get_float_data();
  auto& bias_data = bias_spec.get_float_data();

  const float output_scale = output_scale_spec.get_float_value();
  const int32_t output_zero_point = output_zeros_spec.get_int_value();

  // Calculate number of output elements
  int64_t num_output_elements = N * C_out * H_out * W_out;

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_output_elements);

  // Perform activation, weight, and output quantized depthwise conv2d operation
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t out_c = 0; out_c < C_out; ++out_c) {
      for (int64_t out_h = 0; out_h < H_out; ++out_h) {
        for (int64_t out_w = 0; out_w < W_out; ++out_w) {
          int32_t int_sum = 0;
          int32_t weight_sum = 0; // Track weight sum on the fly

          // For depthwise convolution, each output channel corresponds to one
          // input channel
          int64_t in_c = out_c;

          // Convolution operation with integer accumulation
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

                // Get quantized weight using depthwise layout [K_h, K_w, OC]
                int64_t weight_idx = kh * (K_w * C_out) + kw * C_out + out_c;
                int8_t quantized_weight = weight_data[weight_idx];

                if (false && in_w == 0 && in_h == 0 && out_c == 0) {
                  std::cout << "input: " << input_data[input_idx] << std::endl;
                  std::cout << "quantized_input: " << (int)quantized_input
                            << std::endl;
                  std::cout << "quantized_weight: " << (int)quantized_weight
                            << std::endl;
                }
                // Integer multiplication and accumulation
                int_sum += static_cast<int32_t>(quantized_input) *
                    static_cast<int32_t>(quantized_weight);

                // Track weight sum for this output channel on the fly
                weight_sum += static_cast<int32_t>(quantized_weight);
              } else {
                // For zero padding, we still need to account for the weight
                // in weight_sum when input is effectively 0 (but quantized 0
                // is input_zero_point)
                int64_t weight_idx = kh * (K_w * C_out) + kw * C_out + out_c;
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

          // Convert accumulated integer result to float and apply scales
          // Final result = (int_sum - zero_point_correction) * input_scale *
          // weight_scale + bias zero_point_correction = input_zero_point *
          // sum_of_weights_for_this_output_channel
          int32_t zero_point_correction = input_zero_point * weight_sum;
          int32_t accum_adjusted = int_sum - zero_point_correction;
          float float_result =
              accum_adjusted * input_scale * weight_scales_data[out_c];

          // Add bias and store result
          float_result += bias_data[out_c];

          // Quantize the output to int8
          float quant_output_f =
              std::round(float_result / output_scale) + output_zero_point;
          quant_output_f = std::min(std::max(quant_output_f, -128.0f), 127.0f);
          int8_t quantized_output = static_cast<int8_t>(quant_output_f);

          if (false && out_c < 4 && out_h < 1 && out_w < 4) {
            std::cout << "int_sum[" << out_c << ", " << out_h << ", " << out_w
                      << "] = " << int_sum << ", " << float_result << ", "
                      << output_scale << ", " << quant_output_f << std::endl;
          }

          // Dequantize back to float
          float dequant_output =
              (static_cast<float>(quantized_output) - output_zero_point) *
              output_scale;

          int64_t output_idx = n * (C_out * H_out * W_out) +
              out_c * (H_out * W_out) + out_h * W_out + out_w;
          ref_data[output_idx] = dequant_output;
        }
      }
    }
  }
}

void reference_impl(TestCase& test_case) {
  conv2d_q8ta_q8csw_q8to_dw_reference_impl(test_case);
}

// Custom FLOP calculator for quantized depthwise conv2d operation
int64_t quantized_conv2d_dw_flop_calculator(const TestCase& test_case) {
  int kernel_idx = 9; // kernel_size is at index 9 for q8ta_q8csw_q8to

  // Get input and weight dimensions
  const auto& input_sizes = test_case.inputs()[0].get_tensor_sizes();
  const auto& output_sizes = test_case.outputs()[0].get_tensor_sizes();

  const auto& kernel_sizes = test_case.inputs()[kernel_idx].get_int32_data();

  int64_t N = input_sizes[0];
  int64_t C_out = output_sizes[1];
  int64_t K_h = kernel_sizes[0];
  int64_t K_w = kernel_sizes[1];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];

  // Calculate FLOPs for quantized depthwise conv2d operation
  // Each output element requires:
  // - K_h * K_w multiply-accumulate operations (only one input channel per
  // output channel)
  // - Additional operations for quantization/dequantization
  int64_t output_elements = N * C_out * H_out * W_out;
  int64_t ops_per_output = K_h * K_w;

  int64_t flop = output_elements * ops_per_output;

  return flop;
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout
      << "Quantized Depthwise Conv2d Operation with Output Quantization Prototyping Framework"
      << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  // Execute test cases using the new framework with custom FLOP calculator
  auto results = execute_test_cases(
      generate_quantized_conv2d_dw_test_cases,
      quantized_conv2d_dw_flop_calculator,
      "QuantizedDepthwiseInt8Conv2d",
      3,
      10,
      ref_fn);

  return 0;
}
