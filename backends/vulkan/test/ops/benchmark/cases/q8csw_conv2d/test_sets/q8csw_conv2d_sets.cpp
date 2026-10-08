// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8csw_conv2d/q8csw_conv2d.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8csw_conv2d {

namespace {

std::vector<TestCase> generate_test_cases(
    const std::vector<Conv2dConfig>& configs,
    const std::vector<utils::StorageType>& storage_types) {
  std::vector<TestCase> test_cases;

  // Generate test cases for each combination
  for (auto config : configs) {
    for (const auto& storage_type : storage_types) {
      // Generate test case name programmatically
      bool is_performance = config.channels.out > kRefDimSizeLimit ||
          config.channels.in > kRefDimSizeLimit ||
          config.input_size.h > kRefDimSizeLimit ||
          config.input_size.w > kRefDimSizeLimit;
      std::string prefix = is_performance ? "performance_" : "correctness_";
      std::string suffix = std::to_string(config.channels.out) + "/" +
          std::to_string(config.channels.in) + "_" +
          std::to_string(config.input_size.h) + "/" +
          std::to_string(config.input_size.w) + "_" +
          std::to_string(config.kernel.h) + "/" +
          std::to_string(config.kernel.w);

      config.op_name = "conv2d_q8ta_q8csw";
      config.test_case_name = prefix + suffix;
      // The default operator tested is activation + weight quantized conv2d;
      // however, only test this if the int8 dot product extension is supported
      if (vkcompute::api::context()
              ->adapter_ptr()
              ->supports_int8_dot_product()) {
        test_cases.push_back(
            create_test_case_from_config(config, storage_type, vkapi::kFloat));
      }

      Conv2dConfig wo_quant_config = config;
      wo_quant_config.op_name = "conv2d_q8csw";
      test_cases.push_back(create_test_case_from_config(
          wo_quant_config, storage_type, vkapi::kFloat));
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8csw_conv2d", "correctness") {
  const std::vector<Conv2dConfig> configs = {
      {OutInChannels(32, 3),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(32, 16),
       InputSize2D(32, 32),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 32),
       InputSize2D(16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      // One output channel case
      {OutInChannels(1, 32),
       InputSize2D(55, 55),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},

      // Stride 2 convolutions
      {OutInChannels(32, 3),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 32),
       InputSize2D(32, 32),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      // Different kernel sizes
      {OutInChannels(32, 16),
       InputSize2D(28, 28),
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 32),
       InputSize2D(14, 14),
       KernelSize(7, 7),
       Stride(1, 1),
       Padding(3, 3),
       Dilation(1, 1),
       1},

      // Dilated convolutions
      {OutInChannels(32, 16),
       InputSize2D(32, 32),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(2, 2),
       1},
      {OutInChannels(64, 32),
       InputSize2D(16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(3, 3),
       Dilation(3, 3),
       1},

      // Grouped convolutions
      {OutInChannels(32, 32),
       InputSize2D(32, 32),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       4},
      {OutInChannels(64, 64),
       InputSize2D(16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       8},
  };
  const std::vector<utils::StorageType> storage_types = {utils::kTexture3D};

  return {
      generate_test_cases(configs, storage_types),
      reference_impl,
      quantized_conv2d_flop_calculator};
}

REGISTER_TEST_CASE_SET("q8csw_conv2d", "performance") {
  const std::vector<Conv2dConfig> configs = {
      // Performance test cases
      {OutInChannels(256, 128),
       InputSize2D(128, 128),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(128, 64),
       InputSize2D(128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(128, 1024),
       InputSize2D(128, 128),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
  };
  const std::vector<utils::StorageType> storage_types = {utils::kTexture3D};

  return {
      generate_test_cases(configs, storage_types),
      reference_impl,
      quantized_conv2d_flop_calculator};
}

} // namespace q8csw_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
