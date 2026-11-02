// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/conv2d/conv2d.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace conv2d {

namespace {

struct DirectConfig {
  InputDims dims;
  int64_t C_out;
  std::array<int64_t, 2> perf_hw;
  int64_t groups = 1;
  KernelSize kernel{3, 3};
  Stride stride{1, 1};
  Padding padding{1, 1};
  Dilation dilation{1, 1};
  bool has_bias = false;
};

// Quarter-valued Half products keep these feature tests independent of long
// FP16 reduction error. Group tests encode the input channel in each value.
void set_accuracy_operands(TestCase& test, int seed, int64_t groups) {
  for (int i = 0; i < 2; ++i) {
    auto& spec = test.inputs()[i];
    spec.ensure_data_generated(seed + i);
    if (groups == 1 && spec.dtype != vkapi::kHalf) {
      continue;
    }
    const auto value_at = [&](size_t j, float value) {
      if (groups != 1) {
        return i == 0 ? float(j / (spec.sizes[2] * spec.sizes[3]) + 1)
                      : 0.0625f;
      }
      return value >= 0.0f ? 0.5f : -0.5f;
    };
    if (spec.dtype == vkapi::kHalf) {
      auto& values = spec.get_half_data();
      for (size_t j = 0; j < values.size(); ++j) {
        values[j] = float_to_half(value_at(j, half_to_float(values[j])));
      }
    } else {
      auto& values = spec.get_float_data();
      for (size_t j = 0; j < values.size(); ++j) {
        values[j] = value_at(j, values[j]);
      }
    }
  }
}

std::vector<TestCase> direct_test_cases(bool performance) {
  const std::vector<DirectConfig> configs = {
      {InputDims(1, 16, 9, 13), 16, {48, 64}},
      {InputDims(1, 3, 15, 19), 8, {65, 97}, 1, KernelSize(3, 3), Stride(2, 2)},
      {InputDims(1, 11, 9, 13), 13, {48, 64}},
      {InputDims(1, 17, 9, 13),
       17,
       {49, 65},
       1,
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2)},
      {InputDims(1, 16, 9, 13), 32, {48, 64}},
      {InputDims(1, 16, 11, 17),
       24,
       {31, 47},
       1,
       KernelSize(3, 5),
       Stride(2, 1),
       Padding(0, 3)},
      {InputDims(1, 16, 11, 17),
       16,
       {48, 64},
       1,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(2, 3),
       Dilation(2, 3)},
      {InputDims(1, 8, 11, 17),
       8,
       {32, 48},
       1,
       KernelSize(7, 7),
       Stride(1, 1),
       Padding(3, 3)},
      {InputDims(1, 16, 1, 29),
       32,
       {1, 1537},
       1,
       KernelSize(1, 5),
       Stride(1, 1),
       Padding(0, 2)},
      {InputDims(1, 8, 1, 1),
       8,
       {0, 0},
       1,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(3, 4),
       Dilation(1, 1),
       true},
      {InputDims(1, 64, 8, 12), 64, {32, 48}},
      {InputDims(1, 64, 8, 12),
       64,
       {32, 48},
       1,
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2)},
      {InputDims(1, 128, 8, 12),
       128,
       {16, 24},
       1,
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2)},
      {InputDims(1, 16, 9, 13), 16, {48, 64}, 2},
      {InputDims(1, 16, 9, 13), 24, {48, 64}, 2},
      {InputDims(1, 6, 9, 13), 16, {0, 0}, 2},
      {InputDims(1, 16, 9, 13), 10, {0, 0}, 2},
      {InputDims(1, 6, 9, 13), 10, {0, 0}, 2},
      {InputDims(1, 4, 9, 13), 8, {0, 0}, 4},
      {InputDims(1, 3, 3, 3),
       1,
       {3, 3},
       1,
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(0, 0)},
      {InputDims(1, 6, 9, 13), 8, {0, 0}, 2},
      {InputDims(1, 6, 9, 13), 24, {0, 0}, 2},
  };

  std::vector<TestCase> tests;
  for (int i = 0; i < static_cast<int>(configs.size()); ++i) {
    const auto& config = configs[i];
    if (performance && config.perf_hw[0] == 0) {
      continue;
    }
    const int features = i == 4 && !performance ? 4 : 1;
    for (int feature = 0; feature < features; ++feature) {
      Conv2dTestConfig conv{
          performance
              ? InputDims(
                    1, config.dims.C, config.perf_hw[0], config.perf_hw[1])
              : config.dims,
          config.C_out,
          config.kernel,
          config.stride,
          config.padding,
          config.dilation,
          config.has_bias,
          config.groups};
      if (i == 4) {
        conv.has_bias = performance || (feature & 1);
        if (performance || (feature & 2)) {
          conv.clamp = Conv2dTestConfig::Clamp{-0.25f, 0.5f};
        }
      }
      for (auto dtype : {vkapi::kFloat, vkapi::kHalf}) {
        const std::vector<const char*> selectors = performance
            ? std::vector<const char*>{"direct"}
            : std::vector<const char*>{"direct", "direct_masked"};
        for (const char* impl : selectors) {
          auto test = create_conv2d_test_case(
              conv, dtype, kStorageType, kMemoryLayout, impl);
          test.set_name("M" + std::to_string(i + 1) + " " + test.name());
          if (!performance) {
            set_accuracy_operands(test, 37 + i * 3, config.groups);
          }
          test.set_op_invocations_per_execute(performance ? 0 : 1);
          tests.push_back(std::move(test));
        }
      }
    }
  }
  return tests;
}

} // namespace

REGISTER_TEST_CASE_SET("conv2d", "direct_correctness") {
  return {
      direct_test_cases(false),
      conv2d_exact_operands_reference,
      conv2d_flop_calculator,
      1,
      1};
}

REGISTER_TEST_CASE_SET("conv2d", "direct_performance") {
  return {direct_test_cases(true), nullptr, conv2d_flop_calculator, 5, 20};
}

} // namespace conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
