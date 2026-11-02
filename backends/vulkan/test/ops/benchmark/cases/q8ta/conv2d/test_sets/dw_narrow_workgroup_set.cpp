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

std::vector<TestCase> generate_quantized_conv2d_dw_narrow_workgroup_cases(
    const std::vector<Conv2dConfig>& configs) {
  std::vector<TestCase> test_cases;
  for (Conv2dConfig config : configs) {
    const bool is_performance = config.channels.out > kRefDimSizeLimit;
    config.op_name = "conv2d_q8ta_q8csw_q8to";
    config.test_case_name = make_test_case_name(
        config, is_performance, utils::kTexture3D, utils::kBuffer);
    test_cases.push_back(create_dw_test_case_from_config(
        config, vkapi::kFloat, utils::kTexture3D, utils::kPackedInt8_4C));
  }
  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/conv2d", "dw_narrow_workgroup") {
  const std::vector<Conv2dConfig> configs = {
      {OutInChannels(128, 128),
       InputSize2D(7, 7),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       128},
      {OutInChannels(64, 64),
       InputSize2D(9, 9),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       64},
      {OutInChannels(64, 64),
       InputSize2D(13, 13),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       64},
  };
  return {
      generate_quantized_conv2d_dw_narrow_workgroup_cases(configs),
      dw_reference_impl,
      quantized_conv2d_dw_flop_calculator};
}

} // namespace q8ta_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
