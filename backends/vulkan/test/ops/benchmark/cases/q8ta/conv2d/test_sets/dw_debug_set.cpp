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

// Generate easy test cases for quantized depthwise conv2d operation (for
// debugging)
std::vector<TestCase> generate_quantized_conv2d_dw_easy_cases(
    const Conv2dConfig& source_config,
    const std::vector<utils::StorageType>& fp_storage_types,
    const std::vector<utils::GPUMemoryLayout>& int8_memory_layouts) {
  std::vector<TestCase> test_cases;
  Conv2dConfig config = source_config;
  config.op_name = "conv2d_q8ta_q8csw_q8to";

  // Generate test cases for each combination
  for (const utils::StorageType fp_storage_type : fp_storage_types) {
    for (const utils::GPUMemoryLayout int8_memory_layout :
         int8_memory_layouts) {
      config.test_case_name =
          make_test_case_name(config, false, fp_storage_type, utils::kBuffer);
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

  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/conv2d", "dw_debug") {
  // Single simple configuration for debugging - depthwise convolution
  const Conv2dConfig config = {
      OutInChannels(8, 8), // channels (out, in) - equal for depthwise
      InputSize2D(8, 8), // input_size (h, w)
      KernelSize(3, 3), // kernel
      Stride(1, 1), // stride
      Padding(1, 1), // padding
      Dilation(1, 1), // dilation
      8, // groups = channels.out for depthwise
  };
  const std::vector<utils::StorageType> fp_storage_types = {utils::kTexture3D};
  const std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W, utils::kPackedInt8_4W4C, utils::kPackedInt8_4C};
  return {
      generate_quantized_conv2d_dw_easy_cases(
          config, fp_storage_types, int8_memory_layouts),
      dw_reference_impl,
      quantized_conv2d_dw_flop_calculator};
}

} // namespace q8ta_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
