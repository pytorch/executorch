// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/conv2d_transposed/conv2d_transposed.h>

#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_conv2d_transposed {

namespace {

// Each entry: {config, output_pad_h, output_pad_w}
struct TransposedConvTestConfig {
  Conv2dConfig config;
  int32_t output_pad_h;
  int32_t output_pad_w;
};

std::vector<TestCase> generate_test_cases(
    const std::vector<TransposedConvTestConfig>& configs,
    const std::vector<utils::GPUMemoryLayout>& int8_memory_layouts,
    bool require_int8_dot_product = true) {
  std::vector<TestCase> test_cases;
  if (require_int8_dot_product &&
      !vkcompute::api::context()->adapter_ptr()->supports_int8_dot_product()) {
    return test_cases;
  }

  for (auto tc : configs) {
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

} // namespace

REGISTER_TEST_CASE_SET("q8ta/conv2d_transposed", "correctness") {
  const std::vector<TransposedConvTestConfig> configs = {
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
  };
  const std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W,
      utils::kPackedInt8_4W4C,
      utils::kPackedInt8_4C,
  };
  return {
      generate_test_cases(configs, int8_memory_layouts),
      reference_impl,
      quantized_conv2d_transposed_flop_calculator};
}

REGISTER_TEST_CASE_SET("q8ta/conv2d_transposed", "debug") {
  const std::vector<TransposedConvTestConfig> configs = {
      // Easy test case for debugging
      {{OutInChannels(16, 32),
        InputSize2D(8, 8),
        KernelSize(3, 3),
        Stride(2, 2),
        Padding(1, 1),
        Dilation(1, 1),
        1},
       /*output_pad_h=*/1,
       /*output_pad_w=*/1},
  };
  const std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W,
      utils::kPackedInt8_4W4C,
      utils::kPackedInt8_4C,
  };
  return {
      generate_test_cases(
          configs, int8_memory_layouts, /*require_int8_dot_product=*/false),
      reference_impl,
      quantized_conv2d_transposed_flop_calculator};
}

REGISTER_TEST_CASE_SET("q8ta/conv2d_transposed", "performance") {
  const std::vector<TransposedConvTestConfig> configs = {
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
  const std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W,
      utils::kPackedInt8_4W4C,
      utils::kPackedInt8_4C,
  };
  return {
      generate_test_cases(configs, int8_memory_layouts),
      reference_impl,
      quantized_conv2d_transposed_flop_calculator};
}

} // namespace q8ta_conv2d_transposed
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
