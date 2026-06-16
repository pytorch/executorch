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

static constexpr int64_t kRefDimSizeLimit = 256;

struct Conv1dPWConfig {
  int64_t N;
  int64_t C_in;
  int64_t C_out;
  int64_t L;
  bool has_bias;
};

static TestCase create_conv1d_pw_test_case(
    const Conv1dPWConfig& config,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type) {
  TestCase test_case;

  bool is_perf = config.C_in > kRefDimSizeLimit ||
      config.C_out > kRefDimSizeLimit || config.L > kRefDimSizeLimit;

  std::string prefix = is_perf ? "PERF" : "ACCU";
  std::string storage_str = storage_type_abbrev(storage_type) + "(HP)";
  std::string dtype_str = dtype_short(dtype);

  std::string bias_str = config.has_bias ? "+bias" : "";

  std::string shape = "[" + std::to_string(config.N) + "," +
      std::to_string(config.C_in) + "," + std::to_string(config.L) + "]x[" +
      std::to_string(config.C_out) + "," + std::to_string(config.C_in) + ",1]";

  std::string name = make_test_label(
      prefix, dtype_str, dtype_str, shape, storage_str, bias_str);

  test_case.set_name(name);
  test_case.set_operator_name("test_etvk.test_conv1d_pw.default");

  // Input: [N, C_in, L] height-packed
  ValueSpec input(
      {config.N, config.C_in, config.L},
      dtype,
      storage_type,
      utils::kHeightPacked,
      DataGenType::RANDOM);
  test_case.add_input_spec(input);

  // Weight: [C_out, C_in, 1] height-packed, constant
  ValueSpec weight(
      {config.C_out, config.C_in, 1},
      dtype,
      storage_type,
      utils::kHeightPacked,
      DataGenType::RANDOM);
  weight.set_constant(true);
  test_case.add_input_spec(weight);

  // Bias: [C_out] or None
  if (config.has_bias) {
    ValueSpec bias(
        {config.C_out},
        dtype,
        storage_type,
        utils::kWidthPacked,
        DataGenType::RANDOM);
    bias.set_constant(true);
    test_case.add_input_spec(bias);
  } else {
    ValueSpec none_bias(static_cast<int32_t>(0));
    none_bias.set_none(true);
    test_case.add_input_spec(none_bias);
  }

  // stride = [1]
  test_case.add_input_spec(ValueSpec(std::vector<int32_t>{1}));
  // padding = [0]
  test_case.add_input_spec(ValueSpec(std::vector<int32_t>{0}));
  // dilation = [1]
  test_case.add_input_spec(ValueSpec(std::vector<int32_t>{1}));
  // groups = 1
  test_case.add_input_spec(ValueSpec(static_cast<int32_t>(1)));

  // Output: [N, C_out, L] height-packed
  ValueSpec output(
      {config.N, config.C_out, config.L},
      dtype,
      storage_type,
      utils::kHeightPacked,
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

static void conv1d_pw_reference_impl(TestCase& test_case) {
  const auto& input_spec = test_case.inputs()[0];
  const auto& weight_spec = test_case.inputs()[1];
  const auto& bias_spec = test_case.inputs()[2];
  ValueSpec& output = test_case.outputs()[0];

  if (input_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Reference only supports float");
  }

  auto in_sizes = input_spec.get_tensor_sizes();
  auto w_sizes = weight_spec.get_tensor_sizes();

  int64_t N = in_sizes[0];
  int64_t C_in = in_sizes[1];
  int64_t L = in_sizes[2];
  int64_t C_out = w_sizes[0];

  const auto& in_data = input_spec.get_float_data();
  const auto& w_data = weight_spec.get_float_data();
  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(N * C_out * L, 0.0f);

  // input is NCHW-contiguous: [N, C_in, L]
  // weight is [C_out, C_in, 1]
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t oc = 0; oc < C_out; ++oc) {
      for (int64_t l = 0; l < L; ++l) {
        float sum = 0.0f;
        for (int64_t ic = 0; ic < C_in; ++ic) {
          sum += in_data[n * C_in * L + ic * L + l] * w_data[oc * C_in + ic];
        }
        ref_data[n * C_out * L + oc * L + l] = sum;
      }
    }
  }

  if (!bias_spec.is_none()) {
    const auto& bias_data = bias_spec.get_float_data();
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t oc = 0; oc < C_out; ++oc) {
        for (int64_t l = 0; l < L; ++l) {
          ref_data[n * C_out * L + oc * L + l] += bias_data[oc];
        }
      }
    }
  }
}

static std::vector<TestCase> generate_conv1d_pw_test_cases() {
  std::vector<TestCase> test_cases;

  std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  // Accuracy shapes (float, small)
  std::vector<Conv1dPWConfig> accu_configs = {
      {1, 16, 32, 64, false},
      {1, 16, 32, 64, true},
      {1, 32, 16, 128, false},
      {1, 32, 16, 128, true},
      {1, 64, 64, 32, false},
      {1, 128, 256, 16, true},
      {2, 16, 32, 64, false},
      {2, 16, 32, 64, true},
      // Non-aligned channel counts (not a multiple of 4)
      {1, 5, 7, 64, false},
      {1, 5, 7, 64, true},
      {1, 13, 17, 48, false},
      {1, 13, 17, 48, true},
      {1, 7, 5, 32, false},
      {2, 5, 13, 64, true},
  };

  for (const auto& cfg : accu_configs) {
    for (auto st : storage_types) {
      test_cases.push_back(create_conv1d_pw_test_case(cfg, vkapi::kFloat, st));
    }
  }

  // Performance shapes (half + float)
  std::vector<Conv1dPWConfig> perf_configs = {
      {1, 256, 512, 1024, false},
      {1, 256, 512, 1024, true},
      {1, 512, 256, 2048, false},
      {1, 128, 128, 4096, true},
  };

  for (const auto& cfg : perf_configs) {
    for (auto st : storage_types) {
      test_cases.push_back(create_conv1d_pw_test_case(cfg, vkapi::kFloat, st));
      test_cases.push_back(create_conv1d_pw_test_case(cfg, vkapi::kHalf, st));
    }
  }

  return test_cases;
}

static int64_t conv1d_pw_flop_calculator(const TestCase& test_case) {
  auto in_sizes = test_case.inputs()[0].get_tensor_sizes();
  auto w_sizes = test_case.inputs()[1].get_tensor_sizes();

  int64_t N = in_sizes[0];
  int64_t C_in = in_sizes[1];
  int64_t L = in_sizes[2];
  int64_t C_out = w_sizes[0];

  return 2 * N * C_in * C_out * L;
}

int main(int argc, char* argv[]) {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Conv1d Pointwise (Height-Packed) Benchmark" << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = conv1d_pw_reference_impl;

  auto results = execute_test_cases(
      generate_conv1d_pw_test_cases,
      conv1d_pw_flop_calculator,
      "Conv1dPW",
      /*warmup_runs = */ 1,
      /*benchmark_runs = */ 1,
      ref_fn);

  return 0;
}
