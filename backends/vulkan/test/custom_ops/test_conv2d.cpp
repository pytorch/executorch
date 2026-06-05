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

// Shared perf/skip classification used by both create_conv2d_test_case (to tag
// PERF vs ACCU) and conv2d_reference_impl (to gate the large-K FP16 reference
// check). A shape is "perf" if any dimension reaches kRefDimSizeLimit; the
// boundary is inclusive (>=) so a 64-wide dim counts as perf — FP16
// accumulation error at K = K_h * K_w * C_in for such shapes can exceed the
// half tolerance and false-fail. Keep both call sites on this single helper to
// avoid the two predicates drifting apart.
static bool
conv2d_is_perf_shape(int64_t C_in, int64_t C_out, int64_t H, int64_t W) {
  return C_in >= kRefDimSizeLimit || C_out >= kRefDimSizeLimit ||
      H >= kRefDimSizeLimit || W >= kRefDimSizeLimit;
}

static TestCase create_conv2d_test_case(
    const Conv2dTestConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout,
    const std::string& impl_selector = "") {
  TestCase test_case;

  bool is_perf = conv2d_is_perf_shape(
      config.dims.C, config.C_out, config.dims.H, config.dims.W);

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

// Reference implementation for general conv2d (groups=1).
//
// Supports both FP32 and (small-shape) FP16 inputs. The math is always done in
// float; for FP16 the master input/weight/bias values are dequantized from
// their half storage via get_element(), and the resulting float reference is
// compared against the dequantized GPU output by validate_against_reference().
//
// FP16 accumulation error grows with K (= K_h * K_w * C_in). For large-K PERF
// shapes the FP32 reference would diverge from the GPU's FP16 accumulation
// enough to trip even the relaxed half tolerance, producing false failures, so
// those are intentionally left timing-only: this function throws
// std::invalid_argument, which execute_test_cases() catches to skip the
// correctness check (ref_computed stays false) while still benchmarking.
static void conv2d_reference_impl(TestCase& test_case) {
  const ValueSpec& input = test_case.inputs()[0];
  const ValueSpec& weight = test_case.inputs()[1];
  const ValueSpec& bias_spec = test_case.inputs()[2];
  ValueSpec& output = test_case.outputs()[0];

  if (input.dtype != vkapi::kFloat && input.dtype != vkapi::kHalf) {
    throw std::invalid_argument("Reference only supports float and half");
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

  // For FP16, only compute a reference for small (ACCU) shapes where K is small
  // enough that FP32-vs-FP16 accumulation error stays within the half
  // tolerance. Large-K PERF half shapes stay timing-only via the throw below.
  // The predicate mirrors create_conv2d_test_case's is_perf classification.
  if (input.dtype == vkapi::kHalf) {
    const bool is_perf = conv2d_is_perf_shape(C_in, C_out, H_in, W_in);
    if (is_perf) {
      throw std::invalid_argument(
          "Half reference skipped for large-K PERF shape (timing-only)");
    }
  }

  int64_t stride_h = test_case.inputs()[3].get_int_value();
  int64_t stride_w = test_case.inputs()[4].get_int_value();
  int64_t padding_h = test_case.inputs()[5].get_int_value();
  int64_t padding_w = test_case.inputs()[6].get_int_value();
  int64_t dilation_h = test_case.inputs()[7].get_int_value();
  int64_t dilation_w = test_case.inputs()[8].get_int_value();

  // get_element() materializes a float regardless of dtype (it dequantizes
  // half master data), so the same loop body serves both FP32 and FP16.
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
                  float in_val = input.get_element(
                      n * (C_in * H_in * W_in) + ci * (H_in * W_in) +
                      ih * W_in + iw);
                  // weight is [C_out, C_in, K_h, K_w]
                  float w_val = weight.get_element(
                      co * (C_in * K_h * K_w) + ci * (K_h * K_w) + kh * K_w +
                      kw);
                  sum += in_val * w_val;
                }
              }
            }
          }
          if (!bias_spec.is_none()) {
            sum += bias_spec.get_element(co);
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

  // Small shapes used to exercise each im2col intermediate-storage variant
  // (buffer / texture2d / texture3d) deterministically and independently of
  // the device's auto-selection. All dims <= kRefDimSizeLimit so the float
  // reference validates them. For the texture3d case the im2col intermediate
  // is the channels-packed [1, K_total, H_out, W_out] = [1, 144, 16, 16] for
  // the 16x16 shape — tiny, so it always fits texture3d even on the small
  // shape (texture3d would never be naturally selected for a small shape).
  std::vector<Conv2dTestConfig> per_variant_configs = {
      // 3x3 s1 p1, channels multiple of 4
      {InputDims(1, 16, 16, 16),
       16,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       true},
      // Non-multiple-of-4 channels exercise the Cin padding path
      {InputDims(1, 11, 12, 12),
       13,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       false},
  };

  // Two implementation variants: direct sliding-window (default) and im2col.
  const std::vector<std::string> impls = {"", "im2col"};
  // Forced-storage im2col variants for the per-variant ACCU coverage.
  const std::vector<std::string> forced_storage_impls = {
      "im2col_buffer", "im2col_tex2d", "im2col_tex3d"};

  // Generate accuracy test cases for both impls and both dtypes. FP16 small
  // shapes get a real reference check (gated in conv2d_reference_impl); we run
  // both dtypes so we catch correctness regressions in either path. Large-K
  // half stays timing-only via the reference's PERF-shape throw.
  const std::vector<vkapi::ScalarType> accu_dtypes = {
      vkapi::kFloat, vkapi::kHalf};
  for (const auto& config : accuracy_configs) {
    for (auto st : storage_types) {
      for (auto dtype : accu_dtypes) {
        for (const auto& impl : impls) {
          test_cases.push_back(
              create_conv2d_test_case(config, dtype, st, layout, impl));
        }
      }
    }
  }

  // Generate per-variant forced-storage ACCU cases (FP32 and FP16) so all
  // three im2col intermediate-storage variants get deterministic,
  // device-independent, reference-checked coverage at small K.
  for (const auto& config : per_variant_configs) {
    for (auto st : storage_types) {
      for (auto dtype : accu_dtypes) {
        for (const auto& impl : forced_storage_impls) {
          test_cases.push_back(
              create_conv2d_test_case(config, dtype, st, layout, impl));
        }
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

  execute_test_cases(
      generate_conv2d_test_cases,
      conv2d_flop_calculator,
      "Conv2d",
      /*warmup_runs = */ 5,
      /*benchmark_runs = */ 20,
      ref_fn);

  return 0;
}
