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

// Transposed convolution output size formula:
// H_out = (H_in - 1) * stride_h - 2 * pad_h + dilation_h * (K_h - 1)
//         + output_pad_h + 1
static int64_t get_transpose_output_height(
    const Conv2dConfig& config,
    int32_t output_pad_h) {
  return (config.input_size.h - 1) * config.stride.h - 2 * config.padding.h +
      config.dilation.h * (config.kernel.h - 1) + output_pad_h + 1;
}

static int64_t get_transpose_output_width(
    const Conv2dConfig& config,
    int32_t output_pad_w) {
  return (config.input_size.w - 1) * config.stride.w - 2 * config.padding.w +
      config.dilation.w * (config.kernel.w - 1) + output_pad_w + 1;
}

// Utility function to create a test case from a Conv2dConfig for transposed
// convolution
static TestCase create_test_case_from_config(
    const Conv2dConfig& config,
    int32_t output_pad_h,
    int32_t output_pad_w,
    vkapi::ScalarType input_dtype,
    utils::StorageType fp_storage_type,
    utils::GPUMemoryLayout int8_memory_layout) {
  TestCase test_case;

  int64_t H_out = get_transpose_output_height(config, output_pad_h);
  int64_t W_out = get_transpose_output_width(config, output_pad_w);

  // Input tensor (float/half) - [1, C_in, H_in, W_in] (batch size always 1)
  // For transposed conv, C_in is typically larger (downsampled channels)
  std::vector<int64_t> input_size = {
      1, config.channels.in, config.input_size.h, config.input_size.w};

  utils::GPUMemoryLayout fp_memory_layout = fp_storage_type == utils::kBuffer
      ? utils::kWidthPacked
      : utils::kChannelsPacked;

  // Create test case name
  std::string prefix = config.test_case_name.substr(0, 4);
  std::string test_name = prefix + "  " + std::to_string(config.channels.in) +
      "->" + std::to_string(config.channels.out) + "  " +
      "I=" + std::to_string(config.input_size.h) + "," +
      std::to_string(config.input_size.w) + "  " +
      "g=" + std::to_string(config.groups) + "  " +
      "k=" + std::to_string(config.kernel.h) + "  " +
      "op=" + std::to_string(output_pad_h) + "," +
      std::to_string(output_pad_w) + "  " +
      repr_str(utils::kBuffer, int8_memory_layout);
  test_case.set_name(test_name);

  test_case.set_operator_name("test_etvk.test_q8ta_conv2d_transposed.default");

  ValueSpec input_tensor(
      input_size,
      input_dtype,
      fp_storage_type,
      fp_memory_layout,
      DataGenType::RANDOM);

  if (debugging()) {
    print_valuespec_data(input_tensor, "input_tensor");
  }

  float input_scale_val = 0.008123;
  ValueSpec input_scale(input_scale_val);

  int32_t input_zero_point_val = 2;
  ValueSpec input_zero_point(input_zero_point_val);

  // Quantized weight tensor (int8) - [C_out, align_up_4(C_in_per_group * K_h *
  // K_w)] After the pattern matcher reshapes, the transposed conv weight has
  // the same layout as regular conv2d
  const int64_t in_channels_per_group = config.channels.in / config.groups;
  const int64_t in_features = utils::align_up_4(
      in_channels_per_group * config.kernel.h * config.kernel.w);
  std::vector<int64_t> weight_size = {config.channels.out, in_features};
  ValueSpec quantized_weight(
      weight_size,
      vkapi::kChar,
      fp_storage_type,
      utils::kWidthPacked,
      DataGenType::RANDINT8);
  quantized_weight.set_constant(true);

  if (debugging()) {
    print_valuespec_data(quantized_weight, "weight_tensor");
  }

  const int64_t aligned_out_channels = utils::align_up_4(config.channels.out);

  ValueSpec weight_scales(
      {aligned_out_channels},
      input_dtype,
      fp_storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  weight_scales.set_constant(true);

  ValueSpec weight_sums(
      {aligned_out_channels},
      vkapi::kInt,
      fp_storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  weight_sums.set_constant(true);

  compute_weight_sums(
      weight_sums, quantized_weight, config.channels.out, in_features);

  ValueSpec bias(
      {aligned_out_channels},
      input_dtype,
      fp_storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  bias.set_constant(true);

  float output_scale_val = 0.05314;
  ValueSpec output_scale(output_scale_val);

  int32_t output_zero_point_val = -1;
  ValueSpec output_zero_point(output_zero_point_val);

  ValueSpec stride({config.stride.h, config.stride.w});
  ValueSpec padding({config.padding.h, config.padding.w});
  ValueSpec output_padding({output_pad_h, output_pad_w});
  ValueSpec dilation({config.dilation.h, config.dilation.w});
  ValueSpec groups(config.groups);
  ValueSpec kernel_size({config.kernel.h, config.kernel.w});

  // Output tensor - [1, C_out, H_out, W_out]
  ValueSpec output(
      {1, config.channels.out, H_out, W_out},
      input_dtype,
      fp_storage_type,
      fp_memory_layout,
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
  test_case.add_input_spec(kernel_size);
  test_case.add_input_spec(stride);
  test_case.add_input_spec(padding);
  test_case.add_input_spec(output_padding);
  test_case.add_input_spec(dilation);
  test_case.add_input_spec(groups);

  ValueSpec activation = ValueSpec::make_string("none");
  test_case.add_input_spec(activation);

  ValueSpec layout_int(static_cast<int32_t>(int8_memory_layout));
  test_case.add_input_spec(layout_int);

  test_case.add_output_spec(output);

  test_case.set_abs_tolerance(output_scale_val + 1e-4f);

  test_case.set_shader_filter({
      "nchw_to",
      "to_nchw",
      "q8ta_quantize",
      "q8ta_dequantize",
  });

  return test_case;
}

// Generate easy test cases for debugging
std::vector<TestCase> generate_quantized_conv2d_transposed_easy_cases() {
  std::vector<TestCase> test_cases;

  Conv2dConfig config = {
      OutInChannels(16, 32),
      InputSize2D(8, 8),
      KernelSize(3, 3),
      Stride(2, 2),
      Padding(1, 1),
      Dilation(1, 1),
      1,
  };

  std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W, utils::kPackedInt8_4W4C, utils::kPackedInt8_4C};

  for (const utils::GPUMemoryLayout int8_memory_layout : int8_memory_layouts) {
    config.test_case_name =
        make_test_case_name(config, false, utils::kTexture3D, utils::kBuffer);
    test_cases.push_back(create_test_case_from_config(
        config,
        /*output_pad_h=*/1,
        /*output_pad_w=*/1,
        vkapi::kFloat,
        utils::kTexture3D,
        int8_memory_layout));
  }

  return test_cases;
}

// Generate test cases for quantized transposed conv2d
static std::vector<TestCase> generate_quantized_conv2d_transposed_test_cases() {
  std::vector<TestCase> test_cases;
  if (!vkcompute::api::context()->adapter_ptr()->supports_int8_dot_product()) {
    return test_cases;
  }

  // Each entry: {config, output_pad_h, output_pad_w}
  struct TransposedConvTestConfig {
    Conv2dConfig config;
    int32_t output_pad_h;
    int32_t output_pad_w;
  };

  std::vector<TransposedConvTestConfig> configs = {
      // Basic transposed conv (stride=2, common in decoder networks)
      {{OutInChannels(16, 32),
        InputSize2D(8, 8),
        KernelSize(3, 3),
        Stride(2, 2),
        Padding(1, 1),
        Dilation(1, 1),
        1},
       1,
       1},
      {{OutInChannels(32, 64),
        InputSize2D(4, 4),
        KernelSize(3, 3),
        Stride(2, 2),
        Padding(1, 1),
        Dilation(1, 1),
        1},
       1,
       1},
      // No output padding
      {{OutInChannels(16, 32),
        InputSize2D(8, 8),
        KernelSize(4, 4),
        Stride(2, 2),
        Padding(1, 1),
        Dilation(1, 1),
        1},
       0,
       0},
      // Stride=1 (degenerate case)
      {{OutInChannels(16, 16),
        InputSize2D(8, 8),
        KernelSize(3, 3),
        Stride(1, 1),
        Padding(1, 1),
        Dilation(1, 1),
        1},
       0,
       0},
      // Grouped transposed conv
      {{OutInChannels(32, 64),
        InputSize2D(8, 8),
        KernelSize(3, 3),
        Stride(2, 2),
        Padding(1, 1),
        Dilation(1, 1),
        2},
       1,
       1},
      // Larger spatial
      {{OutInChannels(64, 128),
        InputSize2D(16, 16),
        KernelSize(4, 4),
        Stride(2, 2),
        Padding(1, 1),
        Dilation(1, 1),
        1},
       0,
       0},
      // Performance cases
      {{OutInChannels(64, 128),
        InputSize2D(32, 32),
        KernelSize(3, 3),
        Stride(2, 2),
        Padding(1, 1),
        Dilation(1, 1),
        1},
       1,
       1},
      {{OutInChannels(128, 256),
        InputSize2D(16, 16),
        KernelSize(4, 4),
        Stride(2, 2),
        Padding(1, 1),
        Dilation(1, 1),
        1},
       0,
       0},
  };

  std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W, utils::kPackedInt8_4W4C, utils::kPackedInt8_4C};

  for (auto& tc : configs) {
    auto& config = tc.config;
    bool is_performance = config.channels.out > kRefDimSizeLimit ||
        config.channels.in > kRefDimSizeLimit ||
        config.input_size.h > kRefDimSizeLimit ||
        config.input_size.w > kRefDimSizeLimit;

    for (const utils::GPUMemoryLayout int8_memory_layout :
         int8_memory_layouts) {
      config.test_case_name = make_test_case_name(
          config, is_performance, utils::kTexture3D, utils::kBuffer);

      test_cases.push_back(create_test_case_from_config(
          config,
          tc.output_pad_h,
          tc.output_pad_w,
          vkapi::kFloat,
          utils::kTexture3D,
          int8_memory_layout));
    }
  }

  return test_cases;
}

// Reference implementation for quantized transposed conv2d
static void conv2d_transposed_q8ta_reference_impl(TestCase& test_case) {
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
  const ValueSpec& output_padding_spec = test_case.inputs()[idx++];
  (void)output_padding_spec; // output_padding only affects output size
  const ValueSpec& dilation_spec = test_case.inputs()[idx++];
  const ValueSpec& groups_spec = test_case.inputs()[idx++];
  const ValueSpec& activation_spec = test_case.inputs()[idx++];
  (void)activation_spec;
  const ValueSpec& layout_spec = test_case.inputs()[idx++];
  (void)layout_spec;

  ValueSpec& output_spec = test_case.outputs()[0];

  auto input_sizes = input_spec.get_tensor_sizes();
  auto output_sizes = output_spec.get_tensor_sizes();

  int64_t N = input_sizes[0];
  int64_t C_in = input_sizes[1];
  int64_t H_in = input_sizes[2];
  int64_t W_in = input_sizes[3];
  int64_t C_out = output_sizes[1];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];

  auto kernel_size_data = kernel_size_spec.get_int32_data();
  int64_t K_h = kernel_size_data[0];
  int64_t K_w = kernel_size_data[1];

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

  if (N > kRefDimSizeLimit || C_in > kRefDimSizeLimit ||
      H_in > kRefDimSizeLimit || W_in > kRefDimSizeLimit ||
      C_out > kRefDimSizeLimit) {
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

  int64_t C_in_per_group = C_in / groups;
  int64_t C_out_per_group = C_out / groups;

  int64_t num_output_elements = N * C_out * H_out * W_out;

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_output_elements);

  const int64_t in_features = utils::align_up_4(C_in_per_group * K_h * K_w);

  // Transposed convolution reference implementation.
  // For transposed conv, we scatter each input element across the output
  // rather than gather. But for the reference we compute it by iterating
  // over output positions and finding which input positions contribute.
  //
  // For each output position (oh, ow), an input position (iy, ix) contributes
  // via kernel position (kh, kw) if:
  //   oh + pad_h - kh * dilation_h == iy * stride_h
  //   ow + pad_w - kw * dilation_w == ix * stride_w
  // i.e., (oh + pad_h - kh * dilation_h) must be divisible by stride_h
  // and the quotient must be a valid input index.
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t out_c = 0; out_c < C_out; ++out_c) {
      int64_t group_idx = out_c / C_out_per_group;
      int64_t in_c_start = group_idx * C_in_per_group;

      for (int64_t out_h = 0; out_h < H_out; ++out_h) {
        for (int64_t out_w = 0; out_w < W_out; ++out_w) {
          int32_t int_sum = 0;
          int32_t weight_sum = 0;

          for (int64_t kh = 0; kh < K_h; ++kh) {
            int64_t h_offset = out_h + pad_h - kh * dilation_h;
            if (h_offset < 0 || h_offset % stride_h != 0) {
              continue;
            }
            int64_t iy = h_offset / stride_h;
            if (iy >= H_in) {
              continue;
            }

            for (int64_t kw = 0; kw < K_w; ++kw) {
              int64_t w_offset = out_w + pad_w - kw * dilation_w;
              if (w_offset < 0 || w_offset % stride_w != 0) {
                continue;
              }
              int64_t ix = w_offset / stride_w;
              if (ix >= W_in) {
                continue;
              }

              for (int64_t ic_local = 0; ic_local < C_in_per_group;
                   ++ic_local) {
                int64_t in_c = in_c_start + ic_local;

                int64_t input_idx = n * (C_in * H_in * W_in) +
                    in_c * (H_in * W_in) + iy * W_in + ix;

                float quant_input_f =
                    std::round(input_data[input_idx] / input_scale) +
                    input_zero_point;
                quant_input_f =
                    std::min(std::max(quant_input_f, -128.0f), 127.0f);
                int8_t quantized_input = static_cast<int8_t>(quant_input_f);

                // Weight layout: [C_out, align_up_4(C_in_per_group * K_h *
                // K_w)] Inner dimension order: kh, kw, ic_local
                int64_t weight_idx = out_c * in_features +
                    (kh * (K_w * C_in_per_group) + kw * C_in_per_group +
                     ic_local);
                int8_t quantized_weight = weight_data[weight_idx];

                int_sum += static_cast<int32_t>(quantized_input) *
                    static_cast<int32_t>(quantized_weight);

                weight_sum += static_cast<int32_t>(quantized_weight);
              }
            }
          }

          int32_t zero_point_correction = input_zero_point * weight_sum;
          int32_t accum_adjusted = int_sum - zero_point_correction;
          float float_result =
              accum_adjusted * input_scale * weight_scales_data[out_c];

          float_result += bias_data[out_c];

          float quant_output_f =
              std::round(float_result / output_scale) + output_zero_point;
          quant_output_f = std::min(std::max(quant_output_f, -128.0f), 127.0f);
          int8_t quantized_output = static_cast<int8_t>(quant_output_f);

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
  conv2d_transposed_q8ta_reference_impl(test_case);
}

static int64_t quantized_conv2d_transposed_flop_calculator(
    const TestCase& test_case) {
  int kernel_idx = 9;

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

  int64_t output_elements = N * C_out * H_out * W_out;
  int64_t ops_per_output = C_in * K_h * K_w;

  return output_elements * ops_per_output;
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
      << "Quantized Transposed Conv2d Operation with Output Quantization Prototyping Framework"
      << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  auto results = execute_test_cases(
#ifdef DEBUG_MODE
      generate_quantized_conv2d_transposed_easy_cases,
#else
      generate_quantized_conv2d_transposed_test_cases,
#endif
      quantized_conv2d_transposed_flop_calculator,
      "QuantizedTransposedConv2d",
      /*warmup_runs = */ 1,
      /*benchmark_runs = */ 1,
      ref_fn);

  return 0;
}
