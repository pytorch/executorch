// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <vector>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include "conv2d_utils.h"
#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 64;

struct InputDims {
  int64_t N;
  int64_t C;
  int64_t H;
  int64_t W;

  InputDims(int64_t n, int64_t c, int64_t h, int64_t w)
      : N(n), C(c), H(h), W(w) {}
};

struct Conv2dDwConfig {
  InputDims dims;
  KernelSize kernel;
  Stride stride;
  Padding padding;
  Dilation dilation;
  bool has_bias;
};

static int64_t calc_out_size(
    int64_t in_size,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {
  return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride +
      1;
}

static TestCase create_conv2d_dw_test_case(
    const Conv2dDwConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout,
    const std::string& impl_selector = "") {
  TestCase test_case;

  bool is_perf = config.dims.C > kRefDimSizeLimit ||
      config.dims.H > kRefDimSizeLimit || config.dims.W > kRefDimSizeLimit;

  std::string prefix = is_perf ? "PERF" : "ACCU";
  std::string storage_str = storage_type_abbrev(storage_type);
  std::string layout_str = layout_abbrev(memory_layout);
  std::string dtype_str = (dtype == vkapi::kHalf) ? "f16" : "f32";
  std::string bias_str = config.has_bias ? "+bias" : "";

  int64_t H_out = calc_out_size(
      config.dims.H,
      config.kernel.h,
      config.stride.h,
      config.padding.h,
      config.dilation.h);
  int64_t W_out = calc_out_size(
      config.dims.W,
      config.kernel.w,
      config.stride.w,
      config.padding.w,
      config.dilation.w);

  std::string shape = "[" + std::to_string(config.dims.N) + "," +
      std::to_string(config.dims.C) + "," + std::to_string(config.dims.H) +
      "," + std::to_string(config.dims.W) + "] k" +
      std::to_string(config.kernel.h) + "x" + std::to_string(config.kernel.w) +
      " s" + std::to_string(config.stride.h) + " p" +
      std::to_string(config.padding.h) + " d" +
      std::to_string(config.dilation.h) + "->[" +
      std::to_string(config.dims.N) + "," + std::to_string(config.dims.C) +
      "," + std::to_string(H_out) + "," + std::to_string(W_out) + "]";

  std::string selector_str =
      impl_selector.empty() ? "" : " [" + impl_selector + "]";

  std::string name = prefix + "  conv2d_dw" + bias_str + " " + shape + "  " +
      storage_str + "(" + layout_str + ") " + dtype_str + selector_str;

  test_case.set_name(name);
  test_case.set_operator_name("test_etvk.test_conv2d_dw.default");

  // Input tensor [N, C, H, W]
  ValueSpec input(
      {config.dims.N, config.dims.C, config.dims.H, config.dims.W},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::RANDOM);

  // Weight tensor [C, 1, K_h, K_w] - constant
  ValueSpec weight(
      {config.dims.C, 1, config.kernel.h, config.kernel.w},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::RANDOM);
  weight.set_constant(true);

  test_case.add_input_spec(input);
  test_case.add_input_spec(weight);

  // Bias (or none)
  if (config.has_bias) {
    ValueSpec bias(
        {config.dims.C},
        dtype,
        storage_type,
        memory_layout,
        DataGenType::RANDOM);
    bias.set_constant(true);
    test_case.add_input_spec(bias);
  } else {
    ValueSpec none_bias(static_cast<int32_t>(0));
    none_bias.set_none(true);
    test_case.add_input_spec(none_bias);
  }

  // stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.stride.h)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.stride.w)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.padding.h)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.padding.w)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.dilation.h)));
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(config.dilation.w)));

  // impl_selector string
  test_case.add_input_spec(ValueSpec::make_string(impl_selector));

  // Output tensor [N, C, H_out, W_out]
  ValueSpec output(
      {config.dims.N, config.dims.C, H_out, W_out},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::ZEROS);
  test_case.add_output_spec(output);

  if (dtype == vkapi::kHalf) {
    test_case.set_abs_tolerance(1e-1f);
    test_case.set_rel_tolerance(1e-2f);
  } else {
    test_case.set_abs_tolerance(1e-3f);
    test_case.set_rel_tolerance(1e-3f);
  }

  test_case.set_shader_filter({"nchw_to", "to_nchw", "view_copy"});

  return test_case;
}

// Reference implementation for depthwise conv2d
static void conv2d_dw_reference_impl(TestCase& test_case) {
  const ValueSpec& input = test_case.inputs()[0];
  const ValueSpec& weight = test_case.inputs()[1];
  const ValueSpec& bias_spec = test_case.inputs()[2];
  ValueSpec& output = test_case.outputs()[0];

  if (input.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Reference only supports float");
  }

  auto input_sizes = input.get_tensor_sizes();
  auto weight_sizes = weight.get_tensor_sizes();
  auto output_sizes = output.get_tensor_sizes();

  int64_t N = input_sizes[0];
  int64_t C = input_sizes[1];
  int64_t H_in = input_sizes[2];
  int64_t W_in = input_sizes[3];
  int64_t K_h = weight_sizes[2];
  int64_t K_w = weight_sizes[3];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];

  int64_t stride_h = test_case.inputs()[3].get_int_value();
  int64_t stride_w = test_case.inputs()[4].get_int_value();
  int64_t padding_h = test_case.inputs()[5].get_int_value();
  int64_t padding_w = test_case.inputs()[6].get_int_value();
  int64_t dilation_h = test_case.inputs()[7].get_int_value();
  int64_t dilation_w = test_case.inputs()[8].get_int_value();

  auto& input_data = input.get_float_data();
  auto& weight_data = weight.get_float_data();
  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(N * C * H_out * W_out, 0.0f);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t oh = 0; oh < H_out; ++oh) {
        for (int64_t ow = 0; ow < W_out; ++ow) {
          float sum = 0.0f;
          for (int64_t kh = 0; kh < K_h; ++kh) {
            for (int64_t kw = 0; kw < K_w; ++kw) {
              int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
              int64_t iw = ow * stride_w - padding_w + kw * dilation_w;
              if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                float in_val = input_data
                    [n * (C * H_in * W_in) + c * (H_in * W_in) + ih * W_in +
                     iw];
                // weight is [C, 1, K_h, K_w]
                float w_val = weight_data[c * (K_h * K_w) + kh * K_w + kw];
                sum += in_val * w_val;
              }
            }
          }
          if (!bias_spec.is_none()) {
            auto& bias_data = bias_spec.get_float_data();
            sum += bias_data[c];
          }
          ref_data
              [n * (C * H_out * W_out) + c * (H_out * W_out) + oh * W_out +
               ow] = sum;
        }
      }
    }
  }
}

static std::vector<TestCase> generate_conv2d_dw_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<utils::StorageType> storage_types = {utils::kTexture3D};
  utils::GPUMemoryLayout layout = utils::kChannelsPacked;

  // Accuracy shapes (small enough for float reference validation)
  std::vector<Conv2dDwConfig> accuracy_configs = {
      {InputDims(1, 8, 16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 8, 16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      {InputDims(1, 8, 16, 16),
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(1, 1),
       false},
      {InputDims(1, 8, 16, 16),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // Non-multiple-of-4 channels
      {InputDims(1, 11, 16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 3, 16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
  };

  // EdgeTAM depthwise shapes (from profiling data)
  std::vector<Conv2dDwConfig> perf_configs = {
      // Backbone stem and early stages
      {InputDims(1, 24, 512, 512),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 48, 256, 256),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 48, 256, 256),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 96, 128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 96, 128, 128),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 192, 64, 64),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 192, 64, 64),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 384, 32, 32),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // 5x5 kernels
      {InputDims(1, 48, 256, 256),
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(1, 1),
       false},
      {InputDims(1, 96, 128, 128),
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(1, 1),
       false},
      // FPN/Neck
      {InputDims(1, 256, 256, 256),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 256, 128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
  };

  // Generate accuracy test cases (float only)
  for (const auto& config : accuracy_configs) {
    for (auto st : storage_types) {
      test_cases.push_back(
          create_conv2d_dw_test_case(config, vkapi::kFloat, st, layout));
    }
  }

  // Generate performance test cases (float and half)
  for (const auto& config : perf_configs) {
    std::vector<vkapi::ScalarType> dtypes = {vkapi::kFloat, vkapi::kHalf};
    for (auto dtype : dtypes) {
      for (auto st : storage_types) {
        // Auto-selection (empty impl_selector)
        test_cases.push_back(
            create_conv2d_dw_test_case(config, dtype, st, layout));

        // Force b4x2 variant
        test_cases.push_back(
            create_conv2d_dw_test_case(config, dtype, st, layout, "b4x2"));

        // Force b1x1 variant (only for 3x3 kernels; for 5x5 it falls back
        // to default, but we still generate it to test the fallback path)
        test_cases.push_back(
            create_conv2d_dw_test_case(config, dtype, st, layout, "b1x1"));
      }
    }
  }

  return test_cases;
}

static int64_t conv2d_dw_flop_calculator(const TestCase& test_case) {
  auto input_sizes = test_case.inputs()[0].get_tensor_sizes();
  auto weight_sizes = test_case.inputs()[1].get_tensor_sizes();
  auto output_sizes = test_case.outputs()[0].get_tensor_sizes();

  int64_t N = output_sizes[0];
  int64_t C = output_sizes[1];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];
  int64_t K_h = weight_sizes[2];
  int64_t K_w = weight_sizes[3];

  // Each output element: K_h * K_w multiplies + (K_h * K_w - 1) adds
  return 2 * N * C * H_out * W_out * K_h * K_w;
}

static void reference_impl(TestCase& test_case) {
  conv2d_dw_reference_impl(test_case);
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Depthwise Conv2d Benchmark" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  auto results = execute_test_cases(
      generate_conv2d_dw_test_cases,
      conv2d_dw_flop_calculator,
      "Conv2dDW",
      3,
      10,
      ref_fn);

  return 0;
}
