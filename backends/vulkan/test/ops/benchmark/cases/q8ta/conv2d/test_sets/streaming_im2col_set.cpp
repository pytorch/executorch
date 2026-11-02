// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/conv2d/conv2d.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>

#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_conv2d {

namespace {

struct StreamingIm2colConfig {
  Conv2dConfig config;
  std::vector<utils::GPUMemoryLayout> int8_memory_layouts;
  std::string impl_selector;
};

std::vector<TestCase> generate_streaming_im2col_test_cases(
    const std::vector<StreamingIm2colConfig>& configs) {
  const bool has_int8_dot_product =
      vkcompute::api::context()->adapter_ptr()->supports_int8_dot_product();
  std::vector<TestCase> test_cases;
  for (const StreamingIm2colConfig& streaming_config : configs) {
    // Forced-fallback cases need no int8 dot-product support, so they run on
    // all devices; the other routes stay gated.
    if (!has_int8_dot_product &&
        streaming_config.impl_selector != "im2col_fallback") {
      continue;
    }
    Conv2dConfig config = streaming_config.config;
    config.op_name = "conv2d_q8ta_q8csw_q8to";
    config.test_case_name =
        make_test_case_name(config, false, utils::kTexture3D, utils::kBuffer);
    for (const utils::GPUMemoryLayout int8_memory_layout :
         streaming_config.int8_memory_layouts) {
      test_cases.push_back(create_test_case_from_config(
          config,
          vkapi::kFloat,
          utils::kTexture3D,
          int8_memory_layout,
          streaming_config.impl_selector,
          /*im2col_options=*/nullptr,
          /*input_scale_val=*/1.0f,
          /*input_data_gen=*/DataGenType::RANDINT));
    }
  }
  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/conv2d", "streaming_im2col") {
  const std::vector<StreamingIm2colConfig> configs = {
      // Full fit
      {{OutInChannels(4, 32),
        InputSize2D(30, 99),
        KernelSize(3, 3),
        Stride(1, 1),
        Padding(1, 1),
        Dilation(1, 1),
        1,
        10},
       {utils::kPackedInt8_4W4C},
       "im2col_fallback"},
      // Streaming
      {{OutInChannels(4, 64),
        InputSize2D(30, 99),
        KernelSize(3, 3),
        Stride(1, 1),
        Padding(1, 1),
        Dilation(1, 1),
        1,
        10},
       {utils::kPackedInt8_4W4C},
       "im2col_fallback"},
      // Full fit
      {{OutInChannels(4, 32),
        InputSize2D(30, 99),
        KernelSize(3, 3),
        Stride(1, 1),
        Padding(1, 1),
        Dilation(1, 1),
        1,
        10},
       {utils::kPackedInt8_4C1W, utils::kPackedInt8_4W4C},
       "im2col_auto"},
      // Streaming
      {{OutInChannels(4, 64),
        InputSize2D(30, 99),
        KernelSize(3, 3),
        Stride(1, 1),
        Padding(1, 1),
        Dilation(1, 1),
        1,
        10},
       {utils::kPackedInt8_4C1W},
       "im2col_auto"},
      // Grouped
      {{OutInChannels(16, 32),
        InputSize2D(30, 99),
        KernelSize(3, 3),
        Stride(1, 1),
        Padding(1, 1),
        Dilation(1, 1),
        2,
        20},
       {utils::kPackedInt8_4W4C},
       "im2col_auto"},
      // Output channel tail
      {{OutInChannels(10, 32),
        InputSize2D(30, 99),
        KernelSize(3, 3),
        Stride(1, 1),
        Padding(1, 1),
        Dilation(1, 1),
        1,
        20},
       {utils::kPackedInt8_4W4C},
       "im2col_auto"},
  };
  return {
      generate_streaming_im2col_test_cases(configs),
      reference_impl,
      quantized_conv2d_flop_calculator};
}

} // namespace q8ta_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
