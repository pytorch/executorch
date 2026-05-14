// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>
#include <vector>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 64;

struct Conv2dPwConfig {
  int64_t N;
  int64_t C_in;
  int64_t C_out;
  int64_t H;
  int64_t W;
  bool has_bias;
};

static TestCase create_conv2d_pw_test_case(
    const Conv2dPwConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout) {
  TestCase test_case;

  bool is_perf = config.C_in > kRefDimSizeLimit ||
      config.C_out > kRefDimSizeLimit || config.H > kRefDimSizeLimit ||
      config.W > kRefDimSizeLimit;

  std::string prefix = is_perf ? "PERF" : "ACCU";
  std::string storage_str = storage_type_abbrev(storage_type);
  std::string layout_str = layout_abbrev(memory_layout);
  std::string dtype_str = (dtype == vkapi::kHalf) ? "f16" : "f32";
  std::string bias_str = config.has_bias ? "+bias" : "";

  std::string shape = "[" + std::to_string(config.N) + "," +
      std::to_string(config.C_in) + "," + std::to_string(config.H) + "," +
      std::to_string(config.W) + "]->[" + std::to_string(config.N) + "," +
      std::to_string(config.C_out) + "," + std::to_string(config.H) + "," +
      std::to_string(config.W) + "]";

  std::string name = prefix + "  conv2d_pw" + bias_str + " " + shape + "  " +
      storage_str + "(" + layout_str + ") " + dtype_str;

  test_case.set_name(name);
  test_case.set_operator_name("test_etvk.test_conv2d_pw.default");

  // Input tensor [N, C_in, H, W]
  ValueSpec input(
      {config.N, config.C_in, config.H, config.W},
      dtype,
      storage_type,
      memory_layout,
      DataGenType::RANDOM);

  // Weight tensor [C_out, C_in, 1, 1] - constant
  ValueSpec weight(
      {config.C_out, config.C_in, 1, 1},
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

  // impl_selector
  ValueSpec impl_selector_spec = ValueSpec::make_string("default");
  test_case.add_input_spec(impl_selector_spec);

  // Output tensor [N, C_out, H, W]
  ValueSpec output(
      {config.N, config.C_out, config.H, config.W},
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

// Reference implementation: pointwise conv2d is essentially a matmul
// output[n][c_out][h][w] = bias[c_out] +
//   sum_over_c_in(input[n][c_in][h][w] * weight[c_out][c_in][0][0])
static void conv2d_pw_reference_impl(TestCase& test_case) {
  // input[0], weight[1], bias[2], impl_selector[3]
  const ValueSpec& input = test_case.inputs()[0];
  const ValueSpec& weight = test_case.inputs()[1];
  const ValueSpec& bias_spec = test_case.inputs()[2];
  ValueSpec& output = test_case.outputs()[0];

  if (input.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Reference only supports float");
  }

  auto input_sizes = input.get_tensor_sizes();
  auto weight_sizes = weight.get_tensor_sizes();

  int64_t N = input_sizes[0];
  int64_t C_in = input_sizes[1];
  int64_t H = input_sizes[2];
  int64_t W = input_sizes[3];
  int64_t C_out = weight_sizes[0];

  auto& input_data = input.get_float_data();
  auto& weight_data = weight.get_float_data();
  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(N * C_out * H * W, 0.0f);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t co = 0; co < C_out; ++co) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          float sum = 0.0f;
          for (int64_t ci = 0; ci < C_in; ++ci) {
            float in_val =
                input_data[n * (C_in * H * W) + ci * (H * W) + h * W + w];
            // weight is [C_out, C_in, 1, 1]
            float w_val = weight_data[co * C_in + ci];
            sum += in_val * w_val;
          }
          if (!bias_spec.is_none()) {
            auto& bias_data = bias_spec.get_float_data();
            sum += bias_data[co];
          }
          ref_data[n * (C_out * H * W) + co * (H * W) + h * W + w] = sum;
        }
      }
    }
  }
}

static std::vector<TestCase> generate_conv2d_pw_test_cases() {
  std::vector<TestCase> test_cases;

  // Conv2d shaders are texture-only and require channels-packed layout
  std::vector<utils::StorageType> storage_types = {utils::kTexture3D};
  utils::GPUMemoryLayout layout = utils::kChannelsPacked;

  // Accuracy shapes (small enough for float reference validation)
  std::vector<Conv2dPwConfig> accuracy_configs = {
      {1, 16, 32, 8, 8, false},
      {1, 32, 16, 8, 8, false},
      {1, 16, 32, 8, 8, true},
      {1, 48, 96, 16, 16, false},
      {1, 96, 48, 16, 16, false},
      // Non-multiple-of-4 channels
      {1, 13, 27, 8, 8, false},
      {1, 33, 17, 8, 8, false},
  };

  // EdgeTAM performance shapes
  std::vector<Conv2dPwConfig> perf_configs = {
      // EdgeTAM backbone stages
      {1, 48, 96, 256, 256, false},
      {1, 96, 48, 256, 256, false},
      {1, 96, 192, 128, 128, false},
      {1, 192, 96, 128, 128, false},
      {1, 192, 384, 64, 64, false},
      {1, 384, 192, 64, 64, false},
      {1, 384, 768, 32, 32, false},
      {1, 768, 384, 32, 32, false},
      // EdgeTAM FPN/Neck
      {1, 48, 256, 256, 256, false},
      {1, 256, 32, 256, 256, false},
      {1, 96, 256, 128, 128, false},
      {1, 256, 64, 128, 128, false},
  };

  // Generate accuracy test cases (float only)
  for (const auto& config : accuracy_configs) {
    for (auto st : storage_types) {
      test_cases.push_back(
          create_conv2d_pw_test_case(config, vkapi::kFloat, st, layout));
    }
  }

  // Generate performance test cases (float and half)
  for (const auto& config : perf_configs) {
    std::vector<vkapi::ScalarType> dtypes = {vkapi::kFloat, vkapi::kHalf};
    for (auto dtype : dtypes) {
      for (auto st : storage_types) {
        test_cases.push_back(
            create_conv2d_pw_test_case(config, dtype, st, layout));
      }
    }
  }

  return test_cases;
}

static int64_t conv2d_pw_flop_calculator(const TestCase& test_case) {
  auto input_sizes = test_case.inputs()[0].get_tensor_sizes();
  auto weight_sizes = test_case.inputs()[1].get_tensor_sizes();

  int64_t N = input_sizes[0];
  int64_t C_in = input_sizes[1];
  int64_t H = input_sizes[2];
  int64_t W = input_sizes[3];
  int64_t C_out = weight_sizes[0];

  return 2 * N * C_out * C_in * H * W;
}

static void reference_impl(TestCase& test_case) {
  conv2d_pw_reference_impl(test_case);
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Pointwise Conv2d (1x1) Benchmark" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;

  auto results = execute_test_cases(
      generate_conv2d_pw_test_cases,
      conv2d_pw_flop_calculator,
      "Conv2dPW",
      /*warmup_runs = */ 1,
      /*benchmark_runs = */ 1,
      ref_fn);

  return 0;
}
