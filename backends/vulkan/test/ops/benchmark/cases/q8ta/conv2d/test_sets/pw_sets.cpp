// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/conv2d/conv2d.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>

#include <string>
#include <utility>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_conv2d {

namespace {

// Generate test cases for quantized pointwise conv2d operation. A non-empty pw
// selector lets the test graph builder force or verify the unsigned-dot
// routing instead of the default automatic path.
std::vector<TestCase> generate_quantized_conv2d_pw_test_cases(
    const std::vector<Conv2dConfig>& configs,
    const std::vector<Conv2dConfig>& batch_configs,
    const std::vector<std::pair<Conv2dConfig, PointwiseTestOptions>>&
        edge_configs,
    const std::vector<utils::StorageType>& fp_storage_types,
    const std::vector<utils::GPUMemoryLayout>& int8_memory_layouts,
    const std::vector<std::string>& pw_selectors) {
  std::vector<TestCase> test_cases;
  if (!vkcompute::api::context()->adapter_ptr()->supports_int8_dot_product()) {
    return test_cases;
  }

  // Generate test cases for each combination
  for (Conv2dConfig config : configs) {
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
        for (const std::string& pw_selector : pw_selectors) {
          test_cases.push_back(create_pw_test_case_from_config(
              config,
              vkapi::kFloat,
              fp_storage_type,
              int8_memory_layout,
              pw_selector));
        }

        // For 4W4C layout, also test the legacy implementation
        if (int8_memory_layout == utils::kPackedInt8_4W4C) {
          test_cases.push_back(create_pw_test_case_from_config(
              config,
              vkapi::kFloat,
              fp_storage_type,
              int8_memory_layout,
              /*impl_selector=*/"legacy_4w4c"));
        }
      }
    }
  }

  for (Conv2dConfig config : batch_configs) {
    const bool is_performance = config.input_size.h > kRefDimSizeLimit ||
        config.input_size.w > kRefDimSizeLimit;
    config.op_name = "conv2d_q8ta_q8csw_q8to";
    config.test_case_name = make_test_case_name(
        config, is_performance, utils::kTexture3D, utils::kBuffer);
    for (const std::string& pw_selector : pw_selectors) {
      test_cases.push_back(create_pw_test_case_from_config(
          config,
          vkapi::kFloat,
          utils::kTexture3D,
          utils::kPackedInt8_4C1W,
          pw_selector));
      if (config.batch == 2) {
        test_cases.push_back(create_pw_test_case_from_config(
            config,
            vkapi::kFloat,
            utils::kTexture3D,
            utils::kPackedInt8_4W4C,
            pw_selector));
      }
    }
  }

  for (const auto& [edge_config, options] : edge_configs) {
    Conv2dConfig config = edge_config;
    config.op_name = "conv2d_q8ta_q8csw_q8to";
    config.test_case_name =
        make_test_case_name(config, false, utils::kTexture3D, utils::kBuffer);
    for (const std::string& pw_selector : pw_selectors) {
      test_cases.push_back(create_pw_test_case_from_config(
          config,
          vkapi::kFloat,
          utils::kTexture3D,
          utils::kPackedInt8_4W4C,
          pw_selector,
          options));
    }
  }

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/conv2d", "pw_correctness") {
  const std::vector<Conv2dConfig> configs = {
      // OC < 4 cases to test edge cases with partial output channel blocks
      {OutInChannels(1, 16),
       InputSize2D(8, 8),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(2, 16),
       InputSize2D(8, 8),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(3, 16),
       InputSize2D(8, 8),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      // Pointwise convolutions: kernel size 1x1
      {OutInChannels(32, 3),
       InputSize2D(64, 64),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 32),
       InputSize2D(32, 32),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(96, 64),
       InputSize2D(16, 16),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(13, 7),
       InputSize2D(57, 33),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(80, 40),
       InputSize2D(64, 64),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
  };
  const std::vector<Conv2dConfig> batch_configs = {
      {OutInChannels(8, 8),
       InputSize2D(8, 8),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1,
       2},
  };
  const std::vector<std::pair<Conv2dConfig, PointwiseTestOptions>>
      edge_configs = {
          {{OutInChannels(13, 7),
            InputSize2D(7, 5),
            KernelSize(1, 1),
            Stride(1, 1),
            Padding(0, 0),
            Dilation(1, 1),
            1},
           {.input_zero_point = -128, .has_bias = false}},
      };
  const std::vector<utils::StorageType> fp_storage_types = {utils::kTexture3D};
  const std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W, utils::kPackedInt8_4W4C, utils::kPackedInt8_4C};
  const std::vector<std::string> pw_selectors = {
      "", "pw_signed", "pw_unsigned", "pw_auto"};
  return {
      generate_quantized_conv2d_pw_test_cases(
          configs,
          batch_configs,
          edge_configs,
          fp_storage_types,
          int8_memory_layouts,
          pw_selectors),
      pw_reference_impl,
      quantized_conv2d_flop_calculator};
}

REGISTER_TEST_CASE_SET("q8ta/conv2d", "pw_performance") {
  const std::vector<Conv2dConfig> configs = {
      // Performance cases (pointwise - will use im2col)
      {OutInChannels(160, 480),
       InputSize2D(8, 8),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(22, 48),
       InputSize2D(256, 256),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(48, 48),
       InputSize2D(128, 128),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(128, 128),
       InputSize2D(128, 128),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 576),
       InputSize2D(128, 128),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
  };
  const std::vector<Conv2dConfig> batch_configs = {
      {OutInChannels(64, 32),
       InputSize2D(128, 128),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1,
       1},
      {OutInChannels(64, 32),
       InputSize2D(128, 128),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1,
       60},
  };
  const std::vector<utils::StorageType> fp_storage_types = {utils::kTexture3D};
  const std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W, utils::kPackedInt8_4W4C, utils::kPackedInt8_4C};
  const std::vector<std::string> pw_selectors = {
      "", "pw_signed", "pw_unsigned", "pw_auto"};
  return {
      generate_quantized_conv2d_pw_test_cases(
          configs,
          batch_configs,
          /*edge_configs=*/{},
          fp_storage_types,
          int8_memory_layouts,
          pw_selectors),
      pw_reference_impl,
      quantized_conv2d_flop_calculator};
}

} // namespace q8ta_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
