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

struct Conv2dTestConfig {
  InputDims dims;
  int64_t C_out;
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

static TestCase create_conv2d_test_case(
    const Conv2dTestConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout,
    const std::string& impl_selector = "") {
  TestCase test_case;

  bool is_perf = config.dims.C > kRefDimSizeLimit ||
      config.C_out > kRefDimSizeLimit || config.dims.H > kRefDimSizeLimit ||
      config.dims.W > kRefDimSizeLimit;

  std::string prefix = is_perf ? "PERF" : "ACCU";
  std::string storage_str = repr_str(storage_type, memory_layout);
  std::string dtype_str = dtype_short(dtype);
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
      "," + std::to_string(config.dims.W) + "]->[" +
      std::to_string(config.C_out) + "] k" + std::to_string(config.kernel.h) +
      "x" + std::to_string(config.kernel.w) + " s" +
      std::to_string(config.stride.h) + " p" +
      std::to_string(config.padding.h) + " d" +
      std::to_string(config.dilation.h);

  std::string suffix = bias_str;
  if (!impl_selector.empty()) {
    if (!suffix.empty()) {
      suffix += " ";
    }
    suffix += "[" + impl_selector + "]";
  }

  std::string name =
      make_test_label(prefix, dtype_str, dtype_str, shape, storage_str, suffix);

  test_case.set_name(name);
  test_case.set_operator_name("test_etvk.test_conv2d.default");

  // Input tensor [N, C_in, H, W]
  ValueSpec input(
      {config.dims.N, config.dims.C, config.dims.H, config.dims.W},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::RANDOM);

  // Weight tensor [C_out, C_in, K_h, K_w] - constant
  ValueSpec weight(
      {config.C_out, config.dims.C, config.kernel.h, config.kernel.w},
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
        {config.C_out},
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

  // Output tensor [N, C_out, H_out, W_out]
  ValueSpec output(
      {config.dims.N, config.C_out, H_out, W_out},
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

// Reference implementation for general conv2d (groups=1)
static void conv2d_reference_impl(TestCase& test_case) {
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
  int64_t C_in = input_sizes[1];
  int64_t H_in = input_sizes[2];
  int64_t W_in = input_sizes[3];
  int64_t C_out = weight_sizes[0];
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
  ref_data.resize(N * C_out * H_out * W_out, 0.0f);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t co = 0; co < C_out; ++co) {
      for (int64_t oh = 0; oh < H_out; ++oh) {
        for (int64_t ow = 0; ow < W_out; ++ow) {
          float sum = 0.0f;
          for (int64_t ci = 0; ci < C_in; ++ci) {
            for (int64_t kh = 0; kh < K_h; ++kh) {
              for (int64_t kw = 0; kw < K_w; ++kw) {
                int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
                int64_t iw = ow * stride_w - padding_w + kw * dilation_w;
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                  float in_val = input_data
                      [n * (C_in * H_in * W_in) + ci * (H_in * W_in) +
                       ih * W_in + iw];
                  // weight is [C_out, C_in, K_h, K_w]
                  float w_val = weight_data
                      [co * (C_in * K_h * K_w) + ci * (K_h * K_w) + kh * K_w +
                       kw];
                  sum += in_val * w_val;
                }
              }
            }
          }
          if (!bias_spec.is_none()) {
            auto& bias_data = bias_spec.get_float_data();
            sum += bias_data[co];
          }
          ref_data
              [n * (C_out * H_out * W_out) + co * (H_out * W_out) + oh * W_out +
               ow] = sum;
        }
      }
    }
  }
}

static std::vector<TestCase> generate_conv2d_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<utils::StorageType> storage_types = {utils::kTexture3D};
  utils::GPUMemoryLayout layout = utils::kChannelsPacked;

  // Accuracy shapes (small enough for float reference validation)
  std::vector<Conv2dTestConfig> accuracy_configs = {
      // 3x3 stride=1 pad=1 same-channels (the bottleneck pattern in TinyCNN)
      {InputDims(1, 8, 8, 8),
       8,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 8, 8, 8),
       8,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      {InputDims(1, 16, 16, 16),
       16,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // 3x3 stride=2 (downsample) with channel expansion
      {InputDims(1, 8, 16, 16),
       16,
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // 3x3 stride=1 with channel reduction
      {InputDims(1, 16, 8, 8),
       8,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // Non-multiple-of-4 channels
      {InputDims(1, 11, 8, 8),
       13,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // 3-channel input (like RGB stem)
      {InputDims(1, 3, 16, 16),
       8,
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
  };

  // TinyCNN depth estimator hotspots (from profiling
  // UNTRAINED_TinyCNNDepthEstimatorRealTime_Vulkan.pte).
  // Each entry lists (C_in, H, W) -> C_out, all 3x3 stride=1 pad=1 unless
  // noted. Together the first 6 entries account for ~89% of all conv time.
  std::vector<Conv2dTestConfig> perf_configs = {
      // #1: 21.25% — (1,128,36,48)->(1,128,36,48)
      {InputDims(1, 128, 36, 48),
       128,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // #2: 20.68% — (1,256,18,24)->(1,256,18,24)
      {InputDims(1, 256, 18, 24),
       256,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // #3: 20.01% — (1,64,72,96)->(1,64,72,96)
      {InputDims(1, 64, 72, 96),
       64,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // #4: 13.25% — (1,32,144,192)->(1,32,144,192)
      {InputDims(1, 32, 144, 192),
       32,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // #5: 6.74% — (1,64,36,48)->(1,64,36,48)
      {InputDims(1, 64, 36, 48),
       64,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // #6: 5.90% — (1,32,72,96)->(1,32,72,96)
      {InputDims(1, 32, 72, 96),
       32,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // Secondary cases
      // 3x3 stride=2 downsample with channel expansion: 1.52%
      {InputDims(1, 32, 72, 96),
       128,
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // 3x3 stride=1 same-shape, smaller spatial: 1.51%
      {InputDims(1, 128, 18, 24),
       128,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // 3x3 stride=1, channel reduction
      {InputDims(1, 128, 18, 24),
       64,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 64, 36, 48),
       32,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // 3x3 stride=2 downsample, same channels
      {InputDims(1, 32, 72, 96),
       32,
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      {InputDims(1, 64, 36, 48),
       64,
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
      // RGB stem
      {InputDims(1, 3, 144, 192),
       32,
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       false},
  };

  // Two implementation variants: direct sliding-window (default) and im2col.
  const std::vector<std::string> impls = {"", "im2col"};

  // Generate accuracy test cases (float only).  We run both impls so we can
  // catch correctness regressions in either path.
  for (const auto& config : accuracy_configs) {
    for (auto st : storage_types) {
      for (const auto& impl : impls) {
        test_cases.push_back(
            create_conv2d_test_case(config, vkapi::kFloat, st, layout, impl));
      }
    }
  }

  // Generate performance test cases (float and half) for both impls.
  for (const auto& config : perf_configs) {
    std::vector<vkapi::ScalarType> dtypes = {vkapi::kFloat, vkapi::kHalf};
    for (auto dtype : dtypes) {
      for (auto st : storage_types) {
        for (const auto& impl : impls) {
          test_cases.push_back(
              create_conv2d_test_case(config, dtype, st, layout, impl));
        }
      }
    }
  }

  return test_cases;
}

static int64_t conv2d_flop_calculator(const TestCase& test_case) {
  auto input_sizes = test_case.inputs()[0].get_tensor_sizes();
  auto weight_sizes = test_case.inputs()[1].get_tensor_sizes();
  auto output_sizes = test_case.outputs()[0].get_tensor_sizes();

  int64_t N = output_sizes[0];
  int64_t C_out = output_sizes[1];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];
  int64_t C_in = input_sizes[1];
  int64_t K_h = weight_sizes[2];
  int64_t K_w = weight_sizes[3];

  return 2 * N * C_out * C_in * H_out * W_out * K_h * K_w;
}

static void reference_impl(TestCase& test_case) {
  conv2d_reference_impl(test_case);
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "General Conv2d (SlidingWindow, groups=1) Benchmark"
            << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  auto results = execute_test_cases(
      generate_conv2d_test_cases,
      conv2d_flop_calculator,
      "Conv2d",
      /*warmup_runs = */ 5,
      /*benchmark_runs = */ 20,
      ref_fn);

  return 0;
}
