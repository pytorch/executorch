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

#include "conv2d_utils.h"
#include "utils.h"

// #define DEBUG_MODE

using namespace executorch::vulkan::prototyping;

using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 100;

// Utility function to create a test case from a Conv2dConfig
static TestCase create_test_case_from_config(
    const Conv2dConfig& config,
    vkapi::ScalarType input_dtype,
    utils::StorageType fp_storage_type,
    utils::GPUMemoryLayout int8_memory_layout,
    const std::string& impl_selector = "") {
  TestCase test_case;

  // Calculate output dimensions
  int64_t H_out = config.get_output_height();
  int64_t W_out = config.get_output_width();

  // Input tensor (float/half) - [1, C_in, H_in, W_in] (batch size always 1)
  std::vector<int64_t> input_size = {
      1, config.channels.in, config.input_size.h, config.input_size.w};

  utils::GPUMemoryLayout fp_memory_layout = fp_storage_type == utils::kBuffer
      ? utils::kWidthPacked
      : utils::kChannelsPacked;

  // Create test case name
  // Format: ACCU/PERF  OC->IC  I=H,W  g=groups  k=kernel  Tex(CP)->Buf(4C1W)
  std::string prefix = config.test_case_name.substr(0, 4); // "ACCU" or "PERF"
  std::string test_name = prefix + "  " + std::to_string(config.channels.out) +
      "->" + std::to_string(config.channels.in) + "  " +
      "I=" + std::to_string(config.input_size.h) + "," +
      std::to_string(config.input_size.w) + "  " +
      "g=" + std::to_string(config.groups) + "  " +
      "k=" + std::to_string(config.kernel.h) + "  " +
      repr_str(fp_storage_type, fp_memory_layout) + "->" +
      repr_str(utils::kBuffer, int8_memory_layout);
  if (!impl_selector.empty()) {
    test_name += " [" + impl_selector + "]";
  }
  test_case.set_name(test_name);

  // Set the operator name for the test case - use the unified test operator
  std::string operator_name = "test_etvk.test_q8ta_conv2d.default";
  test_case.set_operator_name(operator_name);

  ValueSpec input_tensor(
      input_size,
      input_dtype,
      fp_storage_type,
      fp_memory_layout,
#ifdef DEBUG_MODE
      DataGenType::RANDOM
#else
      DataGenType::RANDOM
#endif
  );

  if (debugging()) {
    print_valuespec_data(input_tensor, "input_tensor");
  }

  float input_scale_val = 0.008123;
  ValueSpec input_scale(input_scale_val);

  int32_t input_zero_point_val = 2;
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
      fp_storage_type,
      utils::kWidthPacked,
      DataGenType::RANDINT8);
  quantized_weight.set_constant(true);

  if (debugging()) {
    print_valuespec_data(quantized_weight, "weight_tensor");
  }

  const int64_t aligned_out_channels = utils::align_up_4(config.channels.out);

  // Weight quantization scales (float/half, per-channel)
  ValueSpec weight_scales(
      {aligned_out_channels}, // Per output channel
      input_dtype,
      fp_storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  weight_scales.set_constant(true);

  ValueSpec weight_sums(
      {aligned_out_channels}, // Per output channel
      vkapi::kInt,
      fp_storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  weight_sums.set_constant(true);

  // Compute weight_sums data based on quantized weights
  compute_weight_sums(
      weight_sums, quantized_weight, config.channels.out, in_features);

  // Bias (optional, float/half) - [C_out]
  ValueSpec bias(
      {aligned_out_channels}, // Per output channel
      input_dtype,
      fp_storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
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
      fp_storage_type,
      fp_memory_layout,
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

  // Add memory layout parameter for the quantized tensors
  ValueSpec layout_int(static_cast<int32_t>(int8_memory_layout));
  test_case.add_input_spec(layout_int);

  // Add impl_selector string
  ValueSpec impl_selector_spec = ValueSpec::make_string(impl_selector);
  test_case.add_input_spec(impl_selector_spec);

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

// Generate test cases for quantized pointwise conv2d operation
static std::vector<TestCase> generate_quantized_conv2d_pw_test_cases() {
  std::vector<TestCase> test_cases;
  if (!vkcompute::api::context()->adapter_ptr()->supports_int8_dot_product()) {
    return test_cases;
  }

  std::vector<Conv2dConfig> configs = {
      // Pointwise convolutions: kernel size 1x1
      {OutInChannels(32, 3),
       InputSize2D(64, 64),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 32),
       InputSize2D(32, 32),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(96, 64),
       InputSize2D(16, 16),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(13, 7),
       InputSize2D(57, 33),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(80, 40),
       InputSize2D(64, 64),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      // Performance cases (pointwise - will use im2col)
      {OutInChannels(160, 480),
       InputSize2D(8, 8),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(22, 48),
       InputSize2D(256, 256),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(48, 48),
       InputSize2D(128, 128),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(128, 128),
       InputSize2D(128, 128),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
  };

  // Test with different storage types and memory layouts
  std::vector<utils::StorageType> fp_storage_types = {
      utils::kTexture3D, utils::kBuffer};

  // Memory layouts for int8 tensors - test both optimized (4W4C) and general
  // paths
  std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W, utils::kPackedInt8_4W4C, utils::kPackedInt8_4C};

  // Generate test cases for each combination
  for (auto& config : configs) {
    bool is_performance = config.channels.out > kRefDimSizeLimit ||
        config.channels.in > kRefDimSizeLimit ||
        config.input_size.h > kRefDimSizeLimit ||
        config.input_size.w > kRefDimSizeLimit;

    config.op_name = "conv2d_q8ta_q8csw_q8to";

    for (const utils::StorageType fp_storage_type : fp_storage_types) {
      for (const utils::GPUMemoryLayout int8_memory_layout :
           int8_memory_layouts) {
        config.test_case_name = make_test_case_name(
            config, is_performance, fp_storage_type, utils::kBuffer);
        test_cases.push_back(create_test_case_from_config(
            config, vkapi::kFloat, fp_storage_type, int8_memory_layout));

        // For 4W4C layout, also test the legacy implementation
        if (int8_memory_layout == utils::kPackedInt8_4W4C) {
          test_cases.push_back(create_test_case_from_config(
              config,
              vkapi::kFloat,
              fp_storage_type,
              int8_memory_layout,
              /*impl_selector=*/"legacy_4w4c"));
        }
      }
    }
  }

  return test_cases;
}

// Reference implementation for activation, weight, and output quantized conv2d
static void conv2d_q8ta_q8csw_q8to_reference_impl(TestCase& test_case) {
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
  const ValueSpec& layout_spec = test_case.inputs()[idx++];
  (void)layout_spec; // Not used in reference implementation
  const ValueSpec& impl_selector_spec = test_case.inputs()[idx++];
  (void)impl_selector_spec; // Not used in reference implementation

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
    std::cout
        << "Reference implementation: computation may take some time for large tensors..."
        << std::endl;
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

  const float output_scale = output_scale_spec.get_float_value();
  const int32_t output_zero_point = output_zeros_spec.get_int_value();

  // Calculate channels per group for grouped convolution
  int64_t C_in_per_group = C_in / groups;
  int64_t C_out_per_group = C_out / groups;

  // Calculate number of output elements
  int64_t num_output_elements = N * C_out * H_out * W_out;

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_output_elements);

  const int in_features = utils::align_up_4(C_in_per_group * K_h * K_w);

  // Perform activation, weight, and output quantized conv2d operation
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

static void reference_impl(TestCase& test_case) {
  conv2d_q8ta_q8csw_q8to_reference_impl(test_case);
}

// Custom FLOP calculator for quantized conv2d operation
static int64_t quantized_conv2d_flop_calculator(const TestCase& test_case) {
  int kernel_idx = 9; // kernel_size is at index 9 for q8ta_q8csw_q8to

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
#ifdef DEBUG_MODE
  set_print_latencies(true);
#else
  set_print_latencies(false);
#endif
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout
      << "Quantized Pointwise Conv2d (1x1) Operation Prototyping Framework"
      << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  // Execute test cases using the new framework with custom FLOP calculator
  auto results = execute_test_cases(
#ifdef DEBUG_MODE
      generate_quantized_conv2d_pw_test_cases,
#else
      generate_quantized_conv2d_pw_test_cases,
#endif
      quantized_conv2d_flop_calculator,
      "QuantizedConv2dPW",
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
