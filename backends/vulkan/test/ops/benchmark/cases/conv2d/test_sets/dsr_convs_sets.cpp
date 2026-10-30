// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/conv2d/conv2d.h>

#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace conv2d {

namespace {

std::vector<Conv2dTestConfig> dsr_conv_configs() {
  const auto config = [](InputDims dims, int64_t c_out, int32_t kernel) {
    return Conv2dTestConfig{
        dims,
        c_out,
        KernelSize(kernel, kernel),
        Stride(1, 1),
        Padding(kernel / 2, kernel / 2),
        Dilation(1, 1),
        false};
  };
  // DSR dim=64, config=[4,4,4,4,4], input 1512x2016 shapes.
  std::vector<Conv2dTestConfig> configs = {
      config(InputDims(1, 17, 756, 1008), 64, 3),
      config(InputDims(1, 32, 756, 1008), 32, 3),
      config(InputDims(1, 64, 378, 504), 64, 3),
      config(InputDims(1, 128, 189, 252), 128, 3),
      config(InputDims(1, 64, 756, 1008), 64, 3),
      config(InputDims(1, 32, 756, 1008), 32, 5),
      config(InputDims(1, 64, 378, 504), 64, 5),
      config(InputDims(1, 128, 189, 252), 128, 5)};
  const size_t count = configs.size();
  for (size_t i = 0; i < count; ++i) {
    auto biased = configs[i];
    biased.has_bias = true;
    biased.clamp =
        Conv2dTestConfig::Clamp{0.0f, std::numeric_limits<float>::infinity()};
    configs.push_back(biased);
  }
  return configs;
}

void set_half_accuracy_operands(TestCase& test, bool has_bias) {
  for (int i = 0; i < (has_bias ? 3 : 2); ++i) {
    auto& spec = test.inputs()[i];
    spec.ensure_data_generated(42 + i);
    for (auto& value : spec.get_half_data()) {
      value = float_to_half(half_to_float(value) >= 0.0f ? 0.25f : -0.25f);
    }
  }
}

} // namespace

REGISTER_TEST_CASE_SET("conv2d", "dsr_convs_set") {
  std::vector<TestCase> test_cases;
  for (const auto& config : dsr_conv_configs()) {
    test_cases.push_back(create_conv2d_test_case(
        config, vkapi::kHalf, kStorageType, kMemoryLayout));
  }
  // Full-resolution FP16 configurations are timing-only.
  return {test_cases, nullptr, conv2d_flop_calculator, 5, 20};
}

REGISTER_TEST_CASE_SET("conv2d", "dsr_convs_comparison") {
  std::vector<TestCase> test_cases;
  for (const auto& config : dsr_conv_configs()) {
    const std::string baseline = config.C_out >= 128 ? "im2col" : "direct";
    for (const bool optimized : {false, true}) {
      auto test_case = create_conv2d_test_case(
          config,
          vkapi::kHalf,
          kStorageType,
          kMemoryLayout,
          optimized ? "" : baseline);
      test_case.set_name(
          test_case.name() + (optimized ? " [optimized]" : " [baseline]"));
      test_cases.push_back(std::move(test_case));
    }
  }
  return {test_cases, nullptr, conv2d_flop_calculator, 5, 20};
}

REGISTER_TEST_CASE_SET("conv2d", "dsr_convs_correctness") {
  std::vector<TestCase> test_cases;
  std::unordered_set<std::string> names;
  for (auto config : dsr_conv_configs()) {
    config.dims.H = 8;
    config.dims.W = 12;
    for (auto dtype : {vkapi::kFloat, vkapi::kHalf}) {
      auto accuracy_config = config;
      // FP16 references require dimensions below kRefDimSizeLimit.
      if (dtype == vkapi::kHalf) {
        if (accuracy_config.dims.C == 17) {
          accuracy_config.C_out = 31;
        }
        if (accuracy_config.dims.C >= kRefDimSizeLimit ||
            accuracy_config.C_out >= kRefDimSizeLimit) {
          continue;
        }
      }
      auto test_case = create_conv2d_test_case(
          accuracy_config, dtype, kStorageType, kMemoryLayout);
      if (dtype == vkapi::kHalf && accuracy_config.has_bias) {
        set_half_accuracy_operands(test_case, true);
      }
      if (dtype == vkapi::kHalf && accuracy_config.dims.C == 32 &&
          accuracy_config.C_out == 32 && accuracy_config.kernel.h == 5 &&
          accuracy_config.kernel.w == 5) {
        // The 800-term FP16 GEMM sum rounds each FMA; the reference uses FP32.
        test_case.set_abs_tolerance(0.2f);
        test_case.set_rel_tolerance(0.02f);
      }
      if (names.insert(test_case.name()).second) {
        test_cases.push_back(test_case);
      }
    }
  }
  return {test_cases, conv2d_reference_impl, conv2d_flop_calculator, 1, 1};
}

REGISTER_TEST_CASE_SET("conv2d", "dsr_bias_clamp_correctness") {
  std::vector<TestCase> test_cases;
  size_t index = 0;
  for (auto config : dsr_conv_configs()) {
    if (config.has_bias) {
      continue;
    }
    ++index;
    config.dims.H = 9;
    config.dims.W = 13;
    for (int feature = 1; feature < 4; ++feature) {
      config.has_bias = feature & 1;
      config.clamp = feature & 2
          ? std::optional<Conv2dTestConfig::Clamp>(Conv2dTestConfig::Clamp{
                0.0f,
                config.has_bias ? std::numeric_limits<float>::infinity()
                                : 6.0f})
          : std::nullopt;
      auto test = create_conv2d_test_case(
          config, vkapi::kHalf, kStorageType, kMemoryLayout, "direct");
      test.set_name("C" + std::to_string(index) + " " + test.name());
      set_half_accuracy_operands(test, config.has_bias);
      test.set_op_invocations_per_execute(1);
      test_cases.push_back(std::move(test));
    }
  }
  return {
      test_cases,
      conv2d_exact_operands_reference,
      conv2d_flop_calculator,
      1,
      1};
}

} // namespace conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
