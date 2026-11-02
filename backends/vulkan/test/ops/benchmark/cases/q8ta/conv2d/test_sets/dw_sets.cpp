// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/conv2d/conv2d.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>

#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_conv2d {

namespace {

std::vector<TestCase> generate_dw_test_cases(
    const std::vector<Conv2dConfig>& configs,
    const std::vector<Conv2dConfig>& batch_configs,
    const std::vector<utils::StorageType>& fp_storage_types,
    const std::vector<utils::GPUMemoryLayout>& int8_memory_layouts) {
  std::vector<TestCase> test_cases;
  if (!vkcompute::api::context()->adapter_ptr()->supports_int8_dot_product()) {
    return test_cases;
  }

  // Generate test cases for each combination
  for (auto config : configs) {
    bool is_performance = config.channels.out > kRefDimSizeLimit ||
        config.channels.in > kRefDimSizeLimit ||
        config.input_size.h > kRefDimSizeLimit ||
        config.input_size.w > kRefDimSizeLimit;

    config.op_name = "conv2d_q8ta_q8csw_q8to";

    for (const utils::StorageType fp_storage_type : fp_storage_types) {
      for (const utils::GPUMemoryLayout int8_memory_layout :
           int8_memory_layouts) {
        config.test_case_name = make_test_case_name(
            config, is_performance, fp_storage_type, utils::kBuffer);
        test_cases.push_back(create_dw_test_case_from_config(
            config, vkapi::kFloat, fp_storage_type, int8_memory_layout));

        // For 4W4C layout, also test the legacy implementation
        if (int8_memory_layout == utils::kPackedInt8_4W4C) {
          test_cases.push_back(create_dw_test_case_from_config(
              config,
              vkapi::kFloat,
              fp_storage_type,
              int8_memory_layout,
              /*impl_selector=*/"legacy_4w4c"));
        }
      }
    }
  }

  for (auto config : batch_configs) {
    const bool is_performance = config.channels.out > kRefDimSizeLimit ||
        config.channels.in > kRefDimSizeLimit;
    config.op_name = "conv2d_q8ta_q8csw_q8to";
    config.test_case_name = make_test_case_name(
        config, is_performance, utils::kTexture3D, utils::kBuffer);
    test_cases.push_back(create_dw_test_case_from_config(
        config, vkapi::kFloat, utils::kTexture3D, utils::kPackedInt8_4C1W));
    if (config.batch == 2) {
      test_cases.push_back(create_dw_test_case_from_config(
          config, vkapi::kFloat, utils::kTexture3D, utils::kPackedInt8_4W4C));
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/conv2d", "dw_correctness") {
  const std::vector<Conv2dConfig> configs = {
      // Depthwise convolutions: groups = channels.out, channels.in =
      // channels.out
      {OutInChannels(32, 32),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       32},
      {OutInChannels(64, 64),
       InputSize2D(32, 32),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(2, 2),
       Dilation(1, 1),
       64},
      {OutInChannels(64, 64),
       InputSize2D(32, 32),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       64},
      {OutInChannels(80, 80),
       InputSize2D(16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       80},
      {OutInChannels(16, 16),
       InputSize2D(57, 33),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       16},
      // Different kernel sizes for depthwise
      {OutInChannels(32, 32),
       InputSize2D(64, 64),
       KernelSize(5, 5),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(1, 1),
       32},
      {OutInChannels(96, 96),
       InputSize2D(64, 64),
       KernelSize(5, 5),
       Stride(2, 2),
       Padding(2, 2),
       Dilation(1, 1),
       96},
  };
  const std::vector<Conv2dConfig> batch_configs = {
      {OutInChannels(8, 8),
       InputSize2D(8, 8),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       8,
       2},
  };
  const std::vector<utils::StorageType> fp_storage_types = {utils::kTexture3D};
  const std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W, utils::kPackedInt8_4W4C, utils::kPackedInt8_4C};
  return {
      generate_dw_test_cases(
          configs, batch_configs, fp_storage_types, int8_memory_layouts),
      dw_reference_impl,
      quantized_conv2d_dw_flop_calculator};
}

REGISTER_TEST_CASE_SET("q8ta/conv2d", "dw_performance") {
  const std::vector<Conv2dConfig> configs = {
      // Performance cases
      {OutInChannels(128, 128),
       InputSize2D(128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       128},
      {OutInChannels(64, 64),
       InputSize2D(256, 256),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       64},
      {OutInChannels(288, 288),
       InputSize2D(16, 16),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       288},
      {OutInChannels(32, 32),
       InputSize2D(128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(2, 2),
       Dilation(1, 1),
       32},
  };
  const std::vector<Conv2dConfig> batch_configs = {
      {OutInChannels(128, 128),
       InputSize2D(64, 64),
       KernelSize(5, 5),
       Stride(2, 2),
       Padding(2, 2),
       Dilation(1, 1),
       128,
       1},
      {OutInChannels(128, 128),
       InputSize2D(64, 64),
       KernelSize(5, 5),
       Stride(2, 2),
       Padding(2, 2),
       Dilation(1, 1),
       128,
       60},
  };
  const std::vector<utils::StorageType> fp_storage_types = {utils::kTexture3D};
  const std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W, utils::kPackedInt8_4W4C, utils::kPackedInt8_4C};
  return {
      generate_dw_test_cases(
          configs, batch_configs, fp_storage_types, int8_memory_layouts),
      dw_reference_impl,
      quantized_conv2d_dw_flop_calculator};
}

} // namespace q8ta_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
