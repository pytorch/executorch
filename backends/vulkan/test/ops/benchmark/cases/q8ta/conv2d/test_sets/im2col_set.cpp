// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/conv2d/conv2d.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/Q8taConv2dPW.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/im2col/Q8taConv2dIm2Col.h>

#include <string>
#include <utility>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_conv2d {

namespace {

TestCase create_im2col_test_case(
    const Conv2dConfig& source_config,
    utils::StorageType fp_storage_type,
    const std::string& impl_selector,
    const Im2colUnsignedTestOptions& options) {
  Conv2dConfig config = source_config;
  config.op_name = "conv2d_q8ta_q8csw_q8to";
  config.test_case_name =
      make_test_case_name(config, false, fp_storage_type, utils::kBuffer);
  return create_test_case_from_config(
      config,
      vkapi::kFloat,
      fp_storage_type,
      utils::kPackedInt8_4W4C,
      impl_selector,
      &options);
}

std::vector<TestCase> generate_im2col_test_cases(
    const std::vector<std::pair<Conv2dConfig, Im2colUnsignedTestOptions>>&
        configs,
    const std::vector<std::string>& impl_selectors,
    const std::vector<Conv2dConfig>& auto_configs,
    const std::vector<Conv2dConfig>& buffer_output_configs) {
  const vkapi::Adapter& adapter = *vkcompute::api::context()->adapter_ptr();
  std::vector<TestCase> test_cases;
  for (const std::string& impl_selector : impl_selectors) {
    if (impl_selector == "im2col_unsigned" &&
        !adapter.supports_int8_dot_product()) {
      continue;
    }

    for (const auto& [config, options] : configs) {
      test_cases.push_back(create_im2col_test_case(
          config, utils::kTexture3D, impl_selector, options));
    }

    if (impl_selector == "im2col_auto") {
      for (const Conv2dConfig& config : auto_configs) {
        test_cases.push_back(create_im2col_test_case(
            config,
            utils::kTexture3D,
            impl_selector,
            Im2colUnsignedTestOptions{}));
      }
    }

    if (impl_selector == "im2col_unsigned" ||
        (impl_selector == "im2col_auto" &&
         can_use_unsigned_pw_dot(adapter, 4))) {
      for (Conv2dConfig config : buffer_output_configs) {
        config.channels.out = utils::safe_downcast<int32_t>(
            static_cast<int64_t>(adapter.max_texture2d_dim()) * 4 +
            config.channels.out);
        test_cases.push_back(create_im2col_test_case(
            config,
            utils::kBuffer,
            impl_selector,
            Im2colUnsignedTestOptions{}));
      }
    }
  }
  return test_cases;
}

} // namespace

REGISTER_TEST_CASE_SET("q8ta/conv2d", "im2col") {
  const std::vector<std::pair<Conv2dConfig, Im2colUnsignedTestOptions>>
      configs = {
          {{OutInChannels(5, 4),
            InputSize2D(5, 5),
            KernelSize(3, 3),
            Stride(1, 1),
            Padding(0, 0),
            Dilation(1, 1),
            1},
           {.input_zero_point = 2,
            .output_zero_point = -1,
            .use_extreme_values = true,
            .use_accumulator_limit_values = false,
            .has_bias = true,
            .activation = "none"}},
          {{OutInChannels(8, 8),
            InputSize2D(5, 5),
            KernelSize(3, 3),
            Stride(1, 1),
            Padding(1, 1),
            Dilation(1, 1),
            1},
           {.input_zero_point = -7,
            .output_zero_point = 3,
            .use_extreme_values = true,
            .use_accumulator_limit_values = false,
            .has_bias = false,
            .activation = "none"}},
          {{OutInChannels(12, 8),
            InputSize2D(7, 7),
            KernelSize(3, 3),
            Stride(2, 2),
            Padding(1, 1),
            Dilation(1, 1),
            1},
           {.input_zero_point = 127,
            .output_zero_point = -5,
            .use_extreme_values = true,
            .use_accumulator_limit_values = false,
            .has_bias = true,
            .activation = "relu"}},
          {{OutInChannels(12, 8),
            InputSize2D(9, 9),
            KernelSize(3, 3),
            Stride(1, 1),
            Padding(2, 2),
            Dilation(2, 2),
            1},
           {.input_zero_point = -128,
            .output_zero_point = 5,
            .use_extreme_values = true,
            .use_accumulator_limit_values = false,
            .has_bias = false,
            .activation = "none"}},
          {{OutInChannels(8, 8),
            InputSize2D(6, 7),
            KernelSize(3, 3),
            Stride(1, 1),
            Padding(1, 1),
            Dilation(1, 1),
            2},
           {.input_zero_point = 11,
            .output_zero_point = -3,
            .use_extreme_values = true,
            .use_accumulator_limit_values = false,
            .has_bias = true,
            .activation = "relu"}},
          {{OutInChannels(1, 4),
            InputSize2D(90, 91),
            KernelSize(90, 91),
            Stride(1, 1),
            Padding(0, 0),
            Dilation(1, 1),
            1},
           {.input_zero_point = 0,
            .output_zero_point = -1,
            .use_extreme_values = false,
            .use_accumulator_limit_values = true,
            .has_bias = true,
            .activation = "none",
            .weight_scale = 1.0f / 1000000.0f}},
      };
  const std::vector<std::string> impl_selectors = {
      "im2col", "im2col_unsigned", "im2col_auto"};
  const std::vector<Conv2dConfig> auto_configs = {
      {OutInChannels(1, 4),
       InputSize2D(91, 91),
       KernelSize(91, 91),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
  };
  // Output channels are offset by max_texture2d_dim * 4 so that the output
  // exceeds the texture limit and uses buffer storage.
  const std::vector<Conv2dConfig> buffer_output_configs = {
      {OutInChannels(1, 4),
       InputSize2D(1, 1),
       KernelSize(1, 1),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
  };
  return {
      generate_im2col_test_cases(
          configs, impl_selectors, auto_configs, buffer_output_configs),
      reference_impl,
      quantized_conv2d_flop_calculator};
}

} // namespace q8ta_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
