// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taConv2d.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include "conv2d_utils.h"
#include "utils.h"

// #define DEBUG_MODE

using namespace executorch::vulkan::prototyping;

using namespace vkcompute;

static constexpr int64_t kRefDimSizeLimit = 100;
static constexpr int64_t kRefOperationLimit = 2 * 1024 * 1024;

struct Im2colUnsignedTestOptions {
  int32_t input_zero_point = 2;
  int32_t output_zero_point = -1;
  bool use_extreme_values = false;
  bool use_accumulator_limit_values = false;
  bool has_bias = true;
  const char* activation = "none";
  float weight_scale = 1.0f / 256.0f;
};

// Utility function to create a test case from a Conv2dConfig
static TestCase create_test_case_from_config_with_layouts(
    const Conv2dConfig& config,
    vkapi::ScalarType input_dtype,
    utils::StorageType fp_storage_type,
    utils::GPUMemoryLayout input_int8_memory_layout,
    utils::GPUMemoryLayout output_int8_memory_layout,
    const std::string& impl_selector = "",
    const Im2colUnsignedTestOptions* im2col_options = nullptr,
    const float input_scale_val = 0.008123f,
    const DataGenType input_data_gen = DataGenType::RANDOM) {
  TestCase test_case;

  // Calculate output dimensions
  int64_t H_out = config.get_output_height();
  int64_t W_out = config.get_output_width();

  // Input tensor (float/half) - [N, C_in, H_in, W_in]
  std::vector<int64_t> input_size = {
      config.batch,
      config.channels.in,
      config.input_size.h,
      config.input_size.w};

  utils::GPUMemoryLayout fp_memory_layout = fp_storage_type == utils::kBuffer
      ? utils::kWidthPacked
      : utils::kChannelsPacked;

  // Create test case name
  std::string prefix = config.test_case_name.substr(0, 4); // "ACCU" or "PERF"
  std::string dtype_str = dtype_short(input_dtype);
  std::string in_shape = "[" + std::to_string(config.batch) + "," +
      std::to_string(config.channels.in) + "," +
      std::to_string(config.input_size.h) + "," +
      std::to_string(config.input_size.w) + "]";
  std::string weight_shape = "[" + std::to_string(config.channels.out) + "," +
      std::to_string(config.channels.in / config.groups) + "," +
      std::to_string(config.kernel.h) + "," + std::to_string(config.kernel.w) +
      "]";
  std::string shape_str = in_shape + "x" + weight_shape + " s" +
      std::to_string(config.stride.h) + " p" +
      std::to_string(config.padding.h) + " d" +
      std::to_string(config.dilation.h) + " g" + std::to_string(config.groups);
  std::string storage_str = repr_str(utils::kBuffer, input_int8_memory_layout);
  if (input_int8_memory_layout != output_int8_memory_layout) {
    storage_str += "->" + repr_str(utils::kBuffer, output_int8_memory_layout);
  }
  std::string suffix = impl_selector.empty() ? "" : "[" + impl_selector + "]";
  std::string test_name = make_test_label(
      prefix, dtype_str, dtype_str, shape_str, storage_str, suffix);
  test_case.set_name(test_name);

  // Set the operator name for the test case - use the unified test operator
  std::string operator_name = "test_etvk.test_q8ta_conv2d.default";
  test_case.set_operator_name(operator_name);

  ValueSpec input_tensor(
      input_size,
      input_dtype,
      fp_storage_type,
      fp_memory_layout,
      input_data_gen);

  if (debugging()) {
    print_valuespec_data(input_tensor, "input_tensor");
  }

  ValueSpec input_scale(input_scale_val);

  const int32_t input_zero_point_val =
      im2col_options == nullptr ? 2 : im2col_options->input_zero_point;
  ValueSpec input_zero_point(input_zero_point_val);

  if (im2col_options != nullptr &&
      im2col_options->use_accumulator_limit_values) {
    input_tensor.ensure_data_generated(2401);
    std::fill(
        input_tensor.get_float_data().begin(),
        input_tensor.get_float_data().end(),
        (127.0f - input_zero_point_val) * input_scale_val);
  } else if (im2col_options != nullptr && im2col_options->use_extreme_values) {
    input_tensor.ensure_data_generated(2401);
    constexpr std::array<int8_t, 5> values = {-128, -1, 0, 1, 127};
    std::vector<float>& input_data = input_tensor.get_float_data();
    for (size_t i = 0; i < input_data.size(); ++i) {
      input_data.at(i) = (static_cast<float>(values.at(i % values.size())) -
                          input_zero_point_val) *
          input_scale_val;
    }
  }

  // Quantized weight tensor (int8) - [C_out, C_in_per_group * K_h * K_w]
  // Memory layout: height, width, then channels - in_c is innermost (stride 1)
  // in the second dimension
  const int64_t in_channels_per_group = config.channels.in / config.groups;
  const int64_t in_features = utils::align_up_4(
      in_channels_per_group * config.kernel.h * config.kernel.w);
  std::vector<int64_t> weight_size = {config.channels.out, in_features};
  ValueSpec quantized_weight(
      weight_size,
      vkapi::kChar, // int8 for quantized weights
      fp_storage_type,
      utils::kWidthPacked,
      DataGenType::RANDINT8);
  quantized_weight.set_constant(true);

  if (im2col_options != nullptr &&
      im2col_options->use_accumulator_limit_values) {
    quantized_weight.ensure_data_generated(2402);
    std::fill(
        quantized_weight.get_int8_data().begin(),
        quantized_weight.get_int8_data().end(),
        127);
  } else if (im2col_options != nullptr && im2col_options->use_extreme_values) {
    quantized_weight.ensure_data_generated(2402);
    constexpr std::array<int8_t, 5> values = {-128, -1, 0, 1, 127};
    std::vector<int8_t>& weight_data = quantized_weight.get_int8_data();
    for (size_t i = 0; i < weight_data.size(); ++i) {
      weight_data.at(i) = values.at((i * 3 + 1) % values.size());
    }
  }

  if (debugging()) {
    print_valuespec_data(quantized_weight, "weight_tensor");
  }

  const int64_t aligned_out_channels = utils::align_up_4(config.channels.out);

  // Weight quantization scales (float/half, per-channel)
  ValueSpec weight_scales(
      {aligned_out_channels}, // Per output channel
      input_dtype,
      fp_storage_type,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  weight_scales.set_constant(true);
  if (im2col_options != nullptr) {
    weight_scales.ensure_data_generated(2403);
    std::fill(
        weight_scales.get_float_data().begin(),
        weight_scales.get_float_data().end(),
        im2col_options->weight_scale);
  }

  ValueSpec weight_sums(
      {aligned_out_channels}, // Per output channel
      vkapi::kInt,
      fp_storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  weight_sums.set_constant(true);

  // Compute weight_sums data based on quantized weights
  compute_weight_sums(
      weight_sums, quantized_weight, config.channels.out, in_features);

  // Bias (optional, float/half) - [C_out]
  ValueSpec bias(
      {aligned_out_channels}, // Per output channel
      input_dtype,
      fp_storage_type,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  bias.set_constant(true);
  if (im2col_options != nullptr && !im2col_options->has_bias) {
    bias.set_none(true);
  }

  // Output quantization parameters
  float output_scale_val = 0.05314;
  ValueSpec output_scale(output_scale_val);

  const int32_t output_zero_point_val =
      im2col_options == nullptr ? -1 : im2col_options->output_zero_point;
  ValueSpec output_zero_point(output_zero_point_val);

  // Stride and padding parameters
  ValueSpec stride({config.stride.h, config.stride.w});
  ValueSpec padding({config.padding.h, config.padding.w});

  // Dilation and groups parameters
  ValueSpec dilation({config.dilation.h, config.dilation.w});
  ValueSpec groups(config.groups);

  // Kernel size parameters
  ValueSpec kernel_size({config.kernel.h, config.kernel.w});

  // Output tensor (float/half) - [N, C_out, H_out, W_out]
  ValueSpec output(
      {config.batch, config.channels.out, H_out, W_out},
      input_dtype,
      fp_storage_type,
      fp_memory_layout,
      DataGenType::ZEROS);

  // Add all specs to test case for q8ta_q8csw_q8to operation
  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(input_scale);
  test_case.add_input_spec(input_zero_point);
  test_case.add_input_spec(quantized_weight);
  test_case.add_input_spec(weight_sums);
  test_case.add_input_spec(weight_scales);
  test_case.add_input_spec(output_scale);
  test_case.add_input_spec(output_zero_point);
  test_case.add_input_spec(bias);
  test_case.add_input_spec(kernel_size);
  test_case.add_input_spec(stride);
  test_case.add_input_spec(padding);
  test_case.add_input_spec(dilation);
  test_case.add_input_spec(groups);

  // Activation (none = no activation)
  ValueSpec activation = ValueSpec::make_string(
      im2col_options == nullptr ? "none" : im2col_options->activation);
  test_case.add_input_spec(activation);

  // Add memory layout parameter for the quantized tensors
  ValueSpec input_layout_int(static_cast<int32_t>(input_int8_memory_layout));
  test_case.add_input_spec(input_layout_int);

  ValueSpec output_layout_int(static_cast<int32_t>(output_int8_memory_layout));
  test_case.add_input_spec(output_layout_int);

  // Add impl_selector string
  ValueSpec impl_selector_spec = ValueSpec::make_string(impl_selector);
  test_case.add_input_spec(impl_selector_spec);

  test_case.add_output_spec(output);

  test_case.set_abs_tolerance(
      im2col_options == nullptr ? output_scale_val + 1e-4f
                                : output_scale_val * 0.25f);

  // Filter out quantize/dequantize shaders from timing measurements
  test_case.set_shader_filter({
      "nchw_to",
      "to_nchw",
      "q8ta_quantize",
      "q8ta_dequantize",
  });

  return test_case;
}

static TestCase create_test_case_from_config(
    const Conv2dConfig& config,
    vkapi::ScalarType input_dtype,
    utils::StorageType fp_storage_type,
    utils::GPUMemoryLayout int8_memory_layout,
    const std::string& impl_selector = "",
    const Im2colUnsignedTestOptions* im2col_options = nullptr,
    const float input_scale_val = 0.008123f,
    const DataGenType input_data_gen = DataGenType::RANDOM) {
  return create_test_case_from_config_with_layouts(
      config,
      input_dtype,
      fp_storage_type,
      int8_memory_layout,
      int8_memory_layout,
      impl_selector,
      im2col_options,
      input_scale_val,
      input_data_gen);
}

static std::vector<TestCase> generate_narrow_workgroup_test_cases() {
  std::vector<TestCase> test_cases;
  std::vector<Conv2dConfig> configs = {
      {OutInChannels(64, 32),
       InputSize2D(9, 9),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(128, 32),
       InputSize2D(7, 7),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
  };

  for (auto& config : configs) {
    const bool is_performance = config.channels.out > kRefDimSizeLimit;
    config.op_name = "conv2d_q8ta_q8csw_q8to";
    config.test_case_name = make_test_case_name(
        config, is_performance, utils::kTexture3D, utils::kBuffer);
    test_cases.push_back(create_test_case_from_config(
        config,
        vkapi::kFloat,
        utils::kTexture3D,
        utils::kPackedInt8_4C,
        /*impl_selector=*/"general"));
  }
  return test_cases;
}

// Generate easy test cases for quantized conv2d operation (for debugging)
std::vector<TestCase> generate_quantized_conv2d_easy_cases() {
  std::vector<TestCase> test_cases;

  // Single simple configuration for debugging
  Conv2dConfig config = {
      OutInChannels(16, 8), // channels (out, in)
      InputSize2D(5, 5), // input_size (h, w)
      KernelSize(3, 3), // kernel
      Stride(1, 1), // stride
      Padding(1, 1), // padding
      Dilation(1, 1), // dilation
      1, // groups
  };
  config.op_name = "conv2d_q8ta_q8csw_q8to";

  std::vector<utils::StorageType> fp_storage_types = {utils::kTexture3D};

  // Memory layouts for int8 tensors - test both optimized (4W4C) and general
  // paths
  std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W, utils::kPackedInt8_4W4C, utils::kPackedInt8_4C};

  // Generate test cases for each combination
  for (const utils::StorageType fp_storage_type : fp_storage_types) {
    for (const utils::GPUMemoryLayout int8_memory_layout :
         int8_memory_layouts) {
      config.test_case_name =
          make_test_case_name(config, false, fp_storage_type, utils::kBuffer);
      test_cases.push_back(create_test_case_from_config(
          config, vkapi::kFloat, fp_storage_type, int8_memory_layout));

      // Test im2col implementation when input channels per group is a
      // multiple of 4
      if ((config.channels.in / config.groups) % 4 == 0) {
        test_cases.push_back(create_test_case_from_config(
            config,
            vkapi::kFloat,
            fp_storage_type,
            int8_memory_layout,
            /*impl_selector=*/"im2col"));
      }
      // For 4W4C layout, also test the legacy implementation
      if (int8_memory_layout == utils::kPackedInt8_4W4C) {
        test_cases.push_back(create_test_case_from_config(
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

static std::vector<TestCase> generate_im2col_unsigned_test_cases(
    const std::string& impl_selector);

static std::vector<TestCase> generate_streaming_im2col_test_cases() {
  std::vector<TestCase> test_cases;

  Conv2dConfig full_fit_config = {
      OutInChannels(4, 32),
      InputSize2D(30, 99),
      KernelSize(3, 3),
      Stride(1, 1),
      Padding(1, 1),
      Dilation(1, 1),
      1,
      10};
  full_fit_config.op_name = "conv2d_q8ta_q8csw_q8to";
  full_fit_config.test_case_name = make_test_case_name(
      full_fit_config, false, utils::kTexture3D, utils::kBuffer);

  Conv2dConfig streaming_fallback_config = full_fit_config;
  streaming_fallback_config.channels.in = 64;
  streaming_fallback_config.test_case_name = make_test_case_name(
      streaming_fallback_config, false, utils::kTexture3D, utils::kBuffer);

  // Forced-fallback cases need no int8 dot-product support, so they run on
  // all devices; the kAuto cases below stay gated.
  test_cases.push_back(create_test_case_from_config(
      full_fit_config,
      vkapi::kFloat,
      utils::kTexture3D,
      utils::kPackedInt8_4W4C,
      /*impl_selector=*/"im2col_fallback",
      /*im2col_options=*/nullptr,
      /*input_scale_val=*/1.0f,
      /*input_data_gen=*/DataGenType::RANDINT));
  test_cases.push_back(create_test_case_from_config(
      streaming_fallback_config,
      vkapi::kFloat,
      utils::kTexture3D,
      utils::kPackedInt8_4W4C,
      /*impl_selector=*/"im2col_fallback",
      /*im2col_options=*/nullptr,
      /*input_scale_val=*/1.0f,
      /*input_data_gen=*/DataGenType::RANDINT));

  if (!vkcompute::api::context()->adapter_ptr()->supports_int8_dot_product()) {
    return test_cases;
  }

  for (const utils::GPUMemoryLayout layout :
       {utils::kPackedInt8_4C1W, utils::kPackedInt8_4W4C}) {
    test_cases.push_back(create_test_case_from_config(
        full_fit_config,
        vkapi::kFloat,
        utils::kTexture3D,
        layout,
        /*impl_selector=*/"im2col_auto",
        /*im2col_options=*/nullptr,
        /*input_scale_val=*/1.0f,
        /*input_data_gen=*/DataGenType::RANDINT));
  }
  Conv2dConfig streaming_config = full_fit_config;
  streaming_config.channels.in = 64;
  streaming_config.test_case_name = make_test_case_name(
      streaming_config, false, utils::kTexture3D, utils::kBuffer);
  test_cases.push_back(create_test_case_from_config(
      streaming_config,
      vkapi::kFloat,
      utils::kTexture3D,
      utils::kPackedInt8_4C1W,
      /*impl_selector=*/"im2col_auto",
      /*im2col_options=*/nullptr,
      /*input_scale_val=*/1.0f,
      /*input_data_gen=*/DataGenType::RANDINT));
  Conv2dConfig grouped_config = {
      OutInChannels(16, 32),
      InputSize2D(30, 99),
      KernelSize(3, 3),
      Stride(1, 1),
      Padding(1, 1),
      Dilation(1, 1),
      2,
      20};
  grouped_config.op_name = "conv2d_q8ta_q8csw_q8to";
  grouped_config.test_case_name = make_test_case_name(
      grouped_config, false, utils::kTexture3D, utils::kBuffer);
  test_cases.push_back(create_test_case_from_config(
      grouped_config,
      vkapi::kFloat,
      utils::kTexture3D,
      utils::kPackedInt8_4W4C,
      /*impl_selector=*/"im2col_auto",
      /*im2col_options=*/nullptr,
      /*input_scale_val=*/1.0f,
      /*input_data_gen=*/DataGenType::RANDINT));

  Conv2dConfig output_channel_tail_config = {
      OutInChannels(10, 32),
      InputSize2D(30, 99),
      KernelSize(3, 3),
      Stride(1, 1),
      Padding(1, 1),
      Dilation(1, 1),
      1,
      20};
  output_channel_tail_config.op_name = "conv2d_q8ta_q8csw_q8to";
  output_channel_tail_config.test_case_name = make_test_case_name(
      output_channel_tail_config, false, utils::kTexture3D, utils::kBuffer);
  test_cases.push_back(create_test_case_from_config(
      output_channel_tail_config,
      vkapi::kFloat,
      utils::kTexture3D,
      utils::kPackedInt8_4W4C,
      /*impl_selector=*/"im2col_auto",
      /*im2col_options=*/nullptr,
      /*input_scale_val=*/1.0f,
      /*input_data_gen=*/DataGenType::RANDINT));
  return test_cases;
}

// SceneX route tests. The kPackedInt8_4C input + kPackedInt8_4W4C output
// layout combination is only exercised here (default generators never pair
// them); run via --scenex-regular <auto|direct|im2col> <case>. Zero
// tolerances are exact by construction: both pipelines accumulate in int32
// with identical requantize, so any mismatch is a real regression, not noise.
static TestCase create_scenex_test_case(
    const Conv2dConfig& source_config,
    const std::string& route) {
  Conv2dConfig config = source_config;
  config.op_name = "conv2d_q8ta_q8csw_q8to";
  config.test_case_name =
      make_test_case_name(config, false, utils::kTexture3D, utils::kBuffer);

  const std::string impl_selector = route == "auto" ? ""
      : route == "direct"                           ? "general"
                                                    : "im2col";
  TestCase test_case = create_test_case_from_config_with_layouts(
      config,
      vkapi::kFloat,
      utils::kTexture3D,
      utils::kPackedInt8_4C,
      utils::kPackedInt8_4W4C,
      impl_selector,
      /*im2col_options=*/nullptr,
      /*input_scale_val=*/1.0f,
      /*input_data_gen=*/DataGenType::RANDINT);
  test_case.set_abs_tolerance(0.0f);
  test_case.set_rel_tolerance(0.0f);
  return test_case;
}

static TestCase generate_scenex_regular_test_case(
    const std::string& route,
    const int case_index) {
  const std::vector<Conv2dConfig> configs = {
      {OutInChannels(128, 64),
       InputSize2D(40, 51),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1,
       60},
      {OutInChannels(256, 128),
       InputSize2D(20, 26),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1,
       60},
  };
  return create_scenex_test_case(configs.at(case_index), route);
}

static TestCase generate_scenex_grouped_test_case(
    const std::string& route,
    const int case_index) {
  const std::vector<Conv2dConfig> configs = {
      {OutInChannels(64, 64),
       InputSize2D(128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       2,
       60},
      {OutInChannels(128, 128),
       InputSize2D(128, 128),
       KernelSize(5, 5),
       Stride(2, 2),
       Padding(2, 2),
       Dilation(1, 1),
       4,
       60},
      {OutInChannels(64, 64),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       2,
       60},
  };

  return create_scenex_test_case(configs.at(case_index), route);
}

// Generate test cases for quantized conv2d operation
static std::vector<TestCase> generate_quantized_conv2d_test_cases() {
  std::vector<TestCase> test_cases;
  api::Context* const context = vkcompute::api::context();
  if (!context->adapter_ptr()->supports_int8_dot_product()) {
    for (const std::string& impl_selector : {"im2col", "im2col_auto"}) {
      std::vector<TestCase> im2col_cases =
          generate_im2col_unsigned_test_cases(impl_selector);
      for (TestCase& test_case : im2col_cases) {
        test_cases.push_back(std::move(test_case));
      }
    }
    return test_cases;
  }

  std::vector<Conv2dConfig> configs = {
      // General 2D convolutions
      {OutInChannels(32, 3),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(32, 3),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 32),
       InputSize2D(8, 8),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 32),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 32),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(16, 32),
       InputSize2D(77, 77),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      // Grouped convolutions
      {OutInChannels(64, 32),
       InputSize2D(64, 64),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       2},
      {OutInChannels(96, 96),
       InputSize2D(81, 81),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       3},
      {OutInChannels(96, 96),
       InputSize2D(64, 64),
       KernelSize(5, 5),
       Stride(2, 2),
       Padding(2, 2),
       Dilation(1, 1),
       4},
      // Performance cases (3x3 convs - will use im2col)
      {OutInChannels(32, 3),
       InputSize2D(256, 256),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(0, 0),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 32),
       InputSize2D(128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      {OutInChannels(64, 64),
       InputSize2D(128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      // Performance cases (grouped convs)
      {OutInChannels(64, 64),
       InputSize2D(128, 128),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       2},
      {OutInChannels(96, 96),
       InputSize2D(128, 128),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       3},
      {OutInChannels(128, 128),
       InputSize2D(128, 128),
       KernelSize(5, 5),
       Stride(2, 2),
       Padding(2, 2),
       Dilation(1, 1),
       4},
      // SceneX v9 grouped convolutions (large spatial)
      {OutInChannels(128, 128),
       InputSize2D(256, 256),
       KernelSize(5, 5),
       Stride(2, 2),
       Padding(2, 2),
       Dilation(1, 1),
       4},
      {OutInChannels(64, 64),
       InputSize2D(256, 256),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       2},
      // Deep channels + small spatial (ResNet50 stage 5 bottleneck)
      {OutInChannels(512, 512),
       InputSize2D(7, 7),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1},
      // Strided 1x1 shortcut (worst-case strided downsample)
      {OutInChannels(2048, 1024),
       InputSize2D(14, 14),
       KernelSize(1, 1),
       Stride(2, 2),
       Padding(0, 0),
       Dilation(1, 1),
       1}};

  // Test with different storage types and memory layouts
  std::vector<utils::StorageType> fp_storage_types = {utils::kTexture3D};

  // Memory layouts for int8 tensors - test both optimized (4W4C) and general
  // paths
  std::vector<utils::GPUMemoryLayout> int8_memory_layouts = {
      utils::kPackedInt8_4C1W, utils::kPackedInt8_4W4C, utils::kPackedInt8_4C};

  // Generate test cases for each combination
  for (auto& config : configs) {
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

        test_cases.push_back(create_test_case_from_config(
            config,
            vkapi::kFloat,
            fp_storage_type,
            int8_memory_layout,
            /*impl_selector=*/"general"));

        // Test im2col implementation when input channels per group is a
        // multiple of 4
        const int64_t in_channels_per_group =
            config.channels.in / config.groups;
        if (in_channels_per_group % 4 == 0) {
          test_cases.push_back(create_test_case_from_config(
              config,
              vkapi::kFloat,
              fp_storage_type,
              int8_memory_layout,
              /*impl_selector=*/"im2col"));
        }

        // For 4W4C layout, also test the legacy implementation
        if (int8_memory_layout == utils::kPackedInt8_4W4C) {
          test_cases.push_back(create_test_case_from_config(
              config,
              vkapi::kFloat,
              fp_storage_type,
              int8_memory_layout,
              /*impl_selector=*/"legacy_4w4c"));
        }

        test_cases.push_back(create_test_case_from_config(
            config, vkapi::kFloat, fp_storage_type, int8_memory_layout));
      }
    }
  }

  std::vector<Conv2dConfig> batch_configs = {
      {OutInChannels(16, 32),
       InputSize2D(7, 7),
       KernelSize(3, 3),
       Stride(1, 1),
       Padding(1, 1),
       Dilation(1, 1),
       1,
       2},
      {OutInChannels(32, 3),
       InputSize2D(256, 256),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1,
       1},
      {OutInChannels(32, 3),
       InputSize2D(256, 256),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1,
       60},
      {OutInChannels(512, 256),
       InputSize2D(10, 13),
       KernelSize(3, 3),
       Stride(2, 2),
       Padding(1, 1),
       Dilation(1, 1),
       1,
       60}};

  for (auto& config : batch_configs) {
    const bool is_performance = config.batch > kRefDimSizeLimit ||
        config.channels.out > kRefDimSizeLimit ||
        config.channels.in > kRefDimSizeLimit ||
        config.input_size.h > kRefDimSizeLimit ||
        config.input_size.w > kRefDimSizeLimit;
    config.op_name = "conv2d_q8ta_q8csw_q8to";
    config.test_case_name = make_test_case_name(
        config, is_performance, utils::kTexture3D, utils::kBuffer);
    test_cases.push_back(create_test_case_from_config(
        config, vkapi::kFloat, utils::kTexture3D, utils::kPackedInt8_4C1W));
    if (config.batch == 2) {
      test_cases.push_back(create_test_case_from_config(
          config,
          vkapi::kFloat,
          utils::kTexture3D,
          utils::kPackedInt8_4C1W,
          /*impl_selector=*/"im2col"));
      test_cases.push_back(create_test_case_from_config(
          config, vkapi::kFloat, utils::kTexture3D, utils::kPackedInt8_4W4C));
      test_cases.push_back(create_test_case_from_config(
          config,
          vkapi::kFloat,
          utils::kTexture3D,
          utils::kPackedInt8_4W4C,
          /*impl_selector=*/"im2col"));
    }
  }

  for (const std::string& impl_selector :
       {"im2col", "im2col_unsigned", "im2col_auto"}) {
    std::vector<TestCase> im2col_cases =
        generate_im2col_unsigned_test_cases(impl_selector);
    for (TestCase& test_case : im2col_cases) {
      test_cases.push_back(std::move(test_case));
    }
  }

  auto narrow_workgroup_cases = generate_narrow_workgroup_test_cases();
  test_cases.insert(
      test_cases.end(),
      narrow_workgroup_cases.begin(),
      narrow_workgroup_cases.end());

  auto streaming_cases = generate_streaming_im2col_test_cases();
  test_cases.insert(
      test_cases.end(), streaming_cases.begin(), streaming_cases.end());

  return test_cases;
}

static std::vector<TestCase> generate_im2col_unsigned_test_cases(
    const std::string& impl_selector) {
  api::Context* const context = vkcompute::api::context();
  if (impl_selector == "im2col_unsigned" &&
      !context->adapter_ptr()->supports_int8_dot_product()) {
    return {};
  }

  std::vector<std::pair<Conv2dConfig, Im2colUnsignedTestOptions>> configs = {
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

  std::vector<TestCase> test_cases;
  test_cases.reserve(configs.size());
  for (auto& [config, options] : configs) {
    config.op_name = "conv2d_q8ta_q8csw_q8to";
    config.test_case_name =
        make_test_case_name(config, false, utils::kTexture3D, utils::kBuffer);
    test_cases.push_back(create_test_case_from_config(
        config,
        vkapi::kFloat,
        utils::kTexture3D,
        utils::kPackedInt8_4W4C,
        impl_selector,
        &options));
  }

  if (impl_selector == "im2col_auto") {
    Conv2dConfig config{
        OutInChannels(1, 4),
        InputSize2D(91, 91),
        KernelSize(91, 91),
        Stride(1, 1),
        Padding(0, 0),
        Dilation(1, 1),
        1};
    Im2colUnsignedTestOptions options;
    config.op_name = "conv2d_q8ta_q8csw_q8to";
    config.test_case_name =
        make_test_case_name(config, false, utils::kTexture3D, utils::kBuffer);
    test_cases.push_back(create_test_case_from_config(
        config,
        vkapi::kFloat,
        utils::kTexture3D,
        utils::kPackedInt8_4W4C,
        impl_selector,
        &options));
  }

  const vkapi::Adapter& adapter = *context->adapter_ptr();
  if (impl_selector == "im2col_unsigned" ||
      (impl_selector == "im2col_auto" && can_use_unsigned_pw_dot(adapter, 4))) {
    const int32_t buffer_output_channels = utils::safe_downcast<int32_t>(
        static_cast<int64_t>(adapter.max_texture2d_dim()) * 4 + 1);
    Conv2dConfig config{
        OutInChannels(buffer_output_channels, 4),
        InputSize2D(1, 1),
        KernelSize(1, 1),
        Stride(1, 1),
        Padding(0, 0),
        Dilation(1, 1),
        1};
    Im2colUnsignedTestOptions options;
    config.op_name = "conv2d_q8ta_q8csw_q8to";
    config.test_case_name =
        make_test_case_name(config, false, utils::kBuffer, utils::kBuffer);
    test_cases.push_back(create_test_case_from_config(
        config,
        vkapi::kFloat,
        utils::kBuffer,
        utils::kPackedInt8_4W4C,
        impl_selector,
        &options));
  }

  return test_cases;
}

// Reference implementation for activation, weight, and output quantized conv2d
static void conv2d_q8ta_q8csw_q8to_reference_impl(TestCase& test_case) {
  // Extract input specifications
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& input_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& input_zeros_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_spec = test_case.inputs()[idx++];
  const ValueSpec& weight_sums_spec = test_case.inputs()[idx++];
  (void)weight_sums_spec;
  const ValueSpec& weight_scales_spec = test_case.inputs()[idx++];
  const ValueSpec& output_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& output_zeros_spec = test_case.inputs()[idx++];
  const ValueSpec& bias_spec = test_case.inputs()[idx++];
  const ValueSpec& kernel_size_spec = test_case.inputs()[idx++];
  const ValueSpec& stride_spec = test_case.inputs()[idx++];
  const ValueSpec& padding_spec = test_case.inputs()[idx++];
  const ValueSpec& dilation_spec = test_case.inputs()[idx++];
  const ValueSpec& groups_spec = test_case.inputs()[idx++];
  const ValueSpec& activation_spec = test_case.inputs()[idx++];
  const ValueSpec& layout_spec = test_case.inputs()[idx++];
  (void)layout_spec; // Not used in reference implementation
  const ValueSpec& output_layout_spec = test_case.inputs()[idx++];
  (void)output_layout_spec; // Not used in reference implementation
  const ValueSpec& impl_selector_spec = test_case.inputs()[idx++];
  (void)impl_selector_spec; // Not used in reference implementation

  // Extract output specification (mutable reference)
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions
  auto input_sizes = input_spec.get_tensor_sizes(); // [N, C_in, H_in, W_in]
  auto weight_sizes =
      weight_spec.get_tensor_sizes(); // [C_out, C_in_per_group * K_h * K_w]
  auto output_sizes =
      output_spec.get_tensor_sizes(); // [N, C_out, H_out, W_out]

  int64_t N = input_sizes[0];
  int64_t C_in = input_sizes[1];
  int64_t H_in = input_sizes[2];
  int64_t W_in = input_sizes[3];
  int64_t C_out = output_sizes[1];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];

  // Get kernel dimensions from kernel_size ValueSpec
  auto kernel_size_data = kernel_size_spec.get_int32_data();
  int64_t K_h = kernel_size_data[0];
  int64_t K_w = kernel_size_data[1];

  // Get stride, padding, dilation, and groups
  auto stride_data = stride_spec.get_int32_data();
  auto padding_data = padding_spec.get_int32_data();
  auto dilation_data = dilation_spec.get_int32_data();
  int64_t stride_h = stride_data[0];
  int64_t stride_w = stride_data[1];
  int64_t pad_h = padding_data[0];
  int64_t pad_w = padding_data[1];
  int64_t dilation_h = dilation_data[0];
  int64_t dilation_w = dilation_data[1];
  int64_t groups = groups_spec.get_int_value();

  const int64_t reference_operations =
      N * C_out * H_out * W_out * (C_in / groups) * K_h * K_w;
  const bool has_large_dimension = N > kRefDimSizeLimit ||
      C_in > kRefDimSizeLimit || H_in > kRefDimSizeLimit ||
      W_in > kRefDimSizeLimit || C_out > kRefDimSizeLimit;
  if (has_large_dimension && reference_operations > kRefOperationLimit) {
    throw std::invalid_argument(
        "One or more dimensions exceed the allowed limit for reference implementation.");
    std::cout
        << "Reference implementation: computation may take some time for large tensors..."
        << std::endl;
  }

  if (input_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  // Get raw data pointers
  auto& input_data = input_spec.get_float_data();
  const float input_scale = input_scale_spec.get_float_value();
  const int32_t input_zero_point = input_zeros_spec.get_int_value();

  auto& weight_data = weight_spec.get_int8_data();
  auto& weight_scales_data = weight_scales_spec.get_float_data();
  auto& bias_data = bias_spec.get_float_data();

  const float output_scale = output_scale_spec.get_float_value();
  const int32_t output_zero_point = output_zeros_spec.get_int_value();

  // Calculate channels per group for grouped convolution
  int64_t C_in_per_group = C_in / groups;
  int64_t C_out_per_group = C_out / groups;

  // Calculate number of output elements
  int64_t num_output_elements = N * C_out * H_out * W_out;

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_output_elements);

  const int64_t in_features = utils::align_up_4(C_in_per_group * K_h * K_w);

  // Perform activation, weight, and output quantized conv2d operation
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t out_c = 0; out_c < C_out; ++out_c) {
      for (int64_t out_h = 0; out_h < H_out; ++out_h) {
        for (int64_t out_w = 0; out_w < W_out; ++out_w) {
          int32_t int_sum = 0;
          int32_t weight_sum = 0; // Track weight sum on the fly

          // Determine which group this output channel belongs to
          int64_t group_idx = out_c / C_out_per_group;
          int64_t in_c_start = group_idx * C_in_per_group;
          int64_t in_c_end = (group_idx + 1) * C_in_per_group;

          // Convolution operation with integer accumulation
          for (int64_t in_c = in_c_start; in_c < in_c_end; ++in_c) {
            for (int64_t kh = 0; kh < K_h; ++kh) {
              for (int64_t kw = 0; kw < K_w; ++kw) {
                // Calculate input position with dilation
                int64_t in_h = out_h * stride_h - pad_h + kh * dilation_h;
                int64_t in_w = out_w * stride_w - pad_w + kw * dilation_w;

                // Check bounds (zero padding)
                if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
                  // Get input value and quantize to int8
                  int64_t input_idx = n * (C_in * H_in * W_in) +
                      in_c * (H_in * W_in) + in_h * W_in + in_w;

                  float quant_input_f =
                      std::round(input_data[input_idx] / input_scale) +
                      input_zero_point;
                  quant_input_f =
                      std::min(std::max(quant_input_f, -128.0f), 127.0f);
                  int8_t quantized_input = static_cast<int8_t>(quant_input_f);

                  // Get quantized weight (already int8)
                  // Weight layout: [C_out, C_in_per_group * K_h * K_w]
                  int64_t weight_idx = out_c * in_features +
                      (kh * (K_w * C_in_per_group) + kw * C_in_per_group +
                       (in_c % C_in_per_group));
                  int8_t quantized_weight = weight_data[weight_idx];

                  // Integer multiplication and accumulation
                  int_sum += static_cast<int32_t>(quantized_input) *
                      static_cast<int32_t>(quantized_weight);

                  // Track weight sum for this output channel on the fly
                  weight_sum += static_cast<int32_t>(quantized_weight);
                } else {
                  // For zero padding, we still need to account for the weight
                  // in weight_sum when input is effectively 0 (but quantized 0
                  // is input_zero_point)
                  int64_t weight_idx = out_c * in_features +
                      (kh * (K_w * C_in_per_group) + kw * C_in_per_group +
                       (in_c % C_in_per_group));
                  int8_t quantized_weight = weight_data[weight_idx];

                  // Add contribution from zero-padded input (quantized zero =
                  // input_zero_point)
                  int_sum += static_cast<int32_t>(input_zero_point) *
                      static_cast<int32_t>(quantized_weight);

                  // Track weight sum for this output channel on the fly
                  weight_sum += static_cast<int32_t>(quantized_weight);
                }
              }
            }
          }

          // Convert accumulated integer result to float and apply scales
          // Final result = (int_sum - zero_point_correction) * input_scale *
          // weight_scale + bias zero_point_correction = input_zero_point *
          // sum_of_weights_for_this_output_channel
          int32_t zero_point_correction = input_zero_point * weight_sum;
          int32_t accum_adjusted = int_sum - zero_point_correction;
          float float_result =
              accum_adjusted * input_scale * weight_scales_data[out_c];

          if (!bias_spec.is_none()) {
            float_result += bias_data[out_c];
          }
          if (activation_spec.get_string_value() == "relu") {
            float_result = std::max(float_result, 0.0f);
          }

          // Quantize the output to int8
          float quant_output_f =
              std::round(float_result / output_scale) + output_zero_point;
          quant_output_f = std::min(std::max(quant_output_f, -128.0f), 127.0f);
          int8_t quantized_output = static_cast<int8_t>(quant_output_f);

          // Dequantize back to float
          float dequant_output =
              (static_cast<float>(quantized_output) - output_zero_point) *
              output_scale;

          int64_t output_idx = n * (C_out * H_out * W_out) +
              out_c * (H_out * W_out) + out_h * W_out + out_w;
          ref_data[output_idx] = dequant_output;
        }
      }
    }
  }
}

static void reference_impl(TestCase& test_case) {
  conv2d_q8ta_q8csw_q8to_reference_impl(test_case);
}

// The impl selector holds one of "", "general", or "im2col"; activation and
// other string inputs use disjoint values. Overwrite it by value so a
// reordered input list fails loudly instead of mutating the wrong spec.
// Note: for route == "direct" the measured run already forces "general", so
// this reference re-executes the identical implementation and only checks
// determinism; genuine cross-implementation correctness comes from the
// auto/im2col legs.
static void scenex_direct_reference(TestCase& test_case) {
  TestCase direct_case = test_case;
  bool found_selector = false;
  for (auto it = direct_case.inputs().rbegin();
       it != direct_case.inputs().rend();
       ++it) {
    if (it->is_string() &&
        (it->get_string_value().empty() ||
         it->get_string_value() == "general" ||
         it->get_string_value() == "im2col")) {
      it->string_data = "general";
      found_selector = true;
      break;
    }
  }
  if (!found_selector) {
    throw std::runtime_error("scenex reference: impl selector input not found");
  }
  execute_test_case(
      direct_case,
      /*warmup_runs=*/1,
      /*benchmark_runs=*/1,
      /*chained_dispatches=*/1,
      /*write_outputs=*/true);
  test_case.outputs().at(0).get_ref_float_data() =
      direct_case.outputs().at(0).get_float_data();
}

// Custom FLOP calculator for quantized conv2d operation
static int64_t quantized_conv2d_flop_calculator(const TestCase& test_case) {
  int kernel_idx = 9; // kernel_size is at index 9 for q8ta_q8csw_q8to

  // Get input and weight dimensions
  const auto& input_sizes = test_case.inputs()[0].get_tensor_sizes();
  const auto& output_sizes = test_case.outputs()[0].get_tensor_sizes();

  const auto& kernel_sizes = test_case.inputs()[kernel_idx].get_int32_data();

  int64_t N = input_sizes[0];
  int64_t C_in = input_sizes[1];
  int64_t C_out = output_sizes[1];
  int64_t K_h = kernel_sizes[0];
  int64_t K_w = kernel_sizes[1];
  int64_t H_out = output_sizes[2];
  int64_t W_out = output_sizes[3];

  // Calculate FLOPs for quantized conv2d operation
  // Each output element requires:
  // - C_in * K_h * K_w multiply-accumulate operations
  // - Additional operations for quantization/dequantization
  int64_t output_elements = N * C_out * H_out * W_out;
  int64_t ops_per_output = C_in * K_h * K_w;

  int64_t flop = output_elements * (ops_per_output);

  return flop;
}

static void execute_streaming_dynamic_shrink_test() {
  Conv2dConfig config = {
      OutInChannels(4, 64),
      InputSize2D(30, 99),
      KernelSize(3, 3),
      Stride(1, 1),
      Padding(1, 1),
      Dilation(1, 1),
      1,
      10};
  config.op_name = "conv2d_q8ta_q8csw_q8to";
  config.test_case_name =
      make_test_case_name(config, false, utils::kTexture3D, utils::kBuffer);
  TestCase test_case = create_test_case_from_config(
      config,
      vkapi::kFloat,
      utils::kTexture3D,
      utils::kPackedInt8_4W4C,
      /*impl_selector=*/"im2col_auto",
      /*im2col_options=*/nullptr,
      /*input_scale_val=*/1.0f,
      /*input_data_gen=*/DataGenType::RANDINT);
  for (ValueSpec& input : test_case.inputs()) {
    input.ensure_data_generated(/*seed=*/0);
  }

  BenchmarkGraph benchmark_graph = setup_compute_graph(
      test_case, test_case.operator_name(), /*op_invocations_per_execute=*/1);
  ComputeGraph& graph = *benchmark_graph.graph;
  graph.prepare();
  graph.prepack();

  const std::vector<int64_t> upper_input_sizes = test_case.inputs().at(0).sizes;
  const std::vector<int64_t> shrunk_input_sizes = {1, 64, 20, 51};
  const std::vector<int64_t> shrunk_output_sizes = {1, 4, 20, 51};
  TestCase shrunk_case = test_case;
  shrunk_case.inputs().at(0).sizes = shrunk_input_sizes;
  shrunk_case.inputs().at(0).resize_data(1 * 64 * 20 * 51);
  shrunk_case.outputs().at(0).sizes = shrunk_output_sizes;
  shrunk_case.outputs().at(0).resize_data(1 * 4 * 20 * 51);
  reference_impl(shrunk_case);

  constexpr int kRepetitions = 4;
  for (int repetition = 0; repetition < kRepetitions; ++repetition) {
    graph.resize_input(0, upper_input_sizes);
    graph.propagate_resize();
    graph.maybe_cast_and_copy_into_staging(
        graph.inputs().at(0).staging,
        test_case.inputs().at(0).get_data_ptr(),
        test_case.inputs().at(0).numel(),
        vkapi::kFloat);
    graph.execute();

    graph.resize_input(0, shrunk_input_sizes);
    graph.propagate_resize();
    if (graph.sizes_of(graph.outputs().at(0).value) != shrunk_output_sizes) {
      throw std::runtime_error("streaming im2col output did not shrink");
    }
    graph.maybe_cast_and_copy_into_staging(
        graph.inputs().at(0).staging,
        shrunk_case.inputs().at(0).get_data_ptr(),
        shrunk_case.inputs().at(0).numel(),
        vkapi::kFloat);
    graph.execute();
    graph.maybe_cast_and_copy_from_staging(
        graph.outputs().at(0).staging,
        shrunk_case.outputs().at(0).get_mutable_data_ptr(),
        shrunk_case.outputs().at(0).numel(),
        vkapi::kFloat);
    if (!shrunk_case.outputs().at(0).validate_against_reference(
            shrunk_case.get_abs_tolerance(), shrunk_case.get_rel_tolerance())) {
      throw std::runtime_error("streaming im2col shrink output was stale");
    }
  }

  const Q8taConv2dStreamPlan upper_plan = make_q8ta_conv2d_stream_plan(
      /*batch=*/10,
      /*flattened_kernel_size=*/576,
      /*out_height=*/30,
      /*out_width=*/99,
      kQ8taConv2dIm2ColScratchBudgetBytes);
  if (!upper_plan.feasible || upper_plan.num_tiles != 2 ||
      upper_plan.rows_per_tile <= 20) {
    throw std::runtime_error("dynamic shrink test did not create dead tiles");
  }
  const int64_t resized_scratch_bytes =
      576 * upper_plan.rows_per_tile * utils::align_up_4(51);
  if (resized_scratch_bytes > kQ8taConv2dIm2ColScratchBudgetBytes) {
    throw std::runtime_error("streaming im2col scratch exceeded its cap");
  }
  std::cout << "Streaming im2col dynamic shrink PASSED" << std::endl;
}

// Single usage string for the scenex route-test modes; argument errors
// return 2 like the other CLI errors in main.
static int print_scenex_usage(const char* mode, const char* cases) {
  std::cerr << "Usage: " << mode << " <auto|direct|im2col> <" << cases << ">"
            << std::endl;
  return 2;
}

int main(int argc, char* argv[]) {
  const vkapi::Adapter& adapter = *vkcompute::api::context()->adapter_ptr();
  const bool prefers_unsigned_dot =
      adapter.accelerates_unsigned_packed4x8_dot() &&
      !adapter.accelerates_signed_packed4x8_dot();
  VK_CHECK_COND(
      can_use_unsigned_pw_dot(adapter, kMaxUnsignedDotAccumulatorBytes) ==
      prefers_unsigned_dot);
  VK_CHECK_COND(
      !can_use_unsigned_pw_dot(adapter, kMaxUnsignedDotAccumulatorBytes + 1));

  std::string im2col_impl_selector;
  bool narrow_workgroups_only = false;
  bool streaming_im2col_only = false;
  bool streaming_dynamic_shrink_only = false;
  const bool scenex_regular =
      argc == 4 && std::string(argv[1]) == "--scenex-regular";
  const bool scenex_grouped =
      argc == 4 && std::string(argv[1]) == "--scenex-grouped";
  if (argc >= 2 && std::string(argv[1]) == "--scenex-regular" &&
      !scenex_regular) {
    return print_scenex_usage("--scenex-regular", "0|1");
  }
  if (argc >= 2 && std::string(argv[1]) == "--scenex-grouped" &&
      !scenex_grouped) {
    return print_scenex_usage("--scenex-grouped", "0|1|2");
  }
  if (!scenex_regular && !scenex_grouped) {
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--im2col-path=signed") {
        im2col_impl_selector = "im2col";
      } else if (arg == "--im2col-path=unsigned") {
        im2col_impl_selector = "im2col_unsigned";
      } else if (arg == "--im2col-path=auto") {
        im2col_impl_selector = "im2col_auto";
      } else if (arg == "--narrow-workgroups-only") {
        narrow_workgroups_only = true;
      } else if (arg == "--streaming-im2col-only") {
        streaming_im2col_only = true;
      } else if (arg == "--streaming-dynamic-shrink-only") {
        streaming_dynamic_shrink_only = true;
      } else {
        std::cerr << "Unknown argument: " << arg << std::endl;
        return 2;
      }
    }
  }
  const int selected_modes = static_cast<int>(!im2col_impl_selector.empty()) +
      static_cast<int>(narrow_workgroups_only) +
      static_cast<int>(streaming_im2col_only) +
      static_cast<int>(streaming_dynamic_shrink_only) +
      static_cast<int>(scenex_regular) + static_cast<int>(scenex_grouped);
  if (selected_modes > 1) {
    std::cerr << "Test mode selectors are mutually exclusive" << std::endl;
    return 2;
  }
  set_debugging(false);
  set_print_output(false);
#ifdef DEBUG_MODE
  set_print_latencies(true);
#else
  set_print_latencies(false);
#endif
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout
      << "Quantized Conv2d Operation with Output Quantization Prototyping Framework"
      << std::endl;
  print_separator();

  ReferenceComputeFunc ref_fn = reference_impl;
  int warmup_runs = 1;
  int benchmark_runs = 1;

  if (streaming_dynamic_shrink_only) {
    execute_streaming_dynamic_shrink_test();
    return 0;
  }
#ifdef DEBUG_MODE
  std::function<std::vector<TestCase>()> test_case_generator =
      generate_quantized_conv2d_easy_cases;
#else
  std::function<std::vector<TestCase>()> test_case_generator =
      [im2col_impl_selector]() {
        return im2col_impl_selector.empty()
            ? generate_quantized_conv2d_test_cases()
            : generate_im2col_unsigned_test_cases(im2col_impl_selector);
      };
#endif
  if (narrow_workgroups_only) {
    test_case_generator = generate_narrow_workgroup_test_cases;
  } else if (streaming_im2col_only) {
    test_case_generator = generate_streaming_im2col_test_cases;
#ifndef DEBUG_MODE
  } else if (selected_modes == 0) {
    // The default run also covers the unified tile path: multi-tile
    // dispatches, grouped/streaming shapes, and the fallback kernel.
    test_case_generator = [base_generator = test_case_generator]() {
      auto cases = base_generator();
      const auto streaming_cases = generate_streaming_im2col_test_cases();
      cases.insert(cases.end(), streaming_cases.begin(), streaming_cases.end());
      return cases;
    };
#endif
  } else if (scenex_regular) {
    const std::string route = argv[2];
    const std::string case_arg = argv[3];
    if ((route != "auto" && route != "direct" && route != "im2col") ||
        (case_arg != "0" && case_arg != "1")) {
      return print_scenex_usage("--scenex-regular", "0|1");
    }
    const int case_index = case_arg == "0" ? 0 : 1;
    test_case_generator = [route, case_index]() {
      return std::vector<TestCase>{
          generate_scenex_regular_test_case(route, case_index)};
    };
    ref_fn = scenex_direct_reference;
    warmup_runs = 3;
    benchmark_runs = 10;
  } else if (scenex_grouped) {
    const std::string route = argv[2];
    const std::string case_arg = argv[3];
    if ((route != "auto" && route != "direct" && route != "im2col") ||
        (case_arg != "0" && case_arg != "1" && case_arg != "2")) {
      return print_scenex_usage("--scenex-grouped", "0|1|2");
    }
    const int case_index = case_arg == "0" ? 0 : case_arg == "1" ? 1 : 2;
    test_case_generator = [route, case_index]() {
      return std::vector<TestCase>{
          generate_scenex_grouped_test_case(route, case_index)};
    };
    ref_fn = scenex_direct_reference;
    warmup_runs = 3;
    benchmark_runs = 10;
  }

  auto results = execute_test_cases(
      test_case_generator,
      quantized_conv2d_flop_calculator,
      "QuantizedConv2dQ8ToQ8To",
      warmup_runs,
      benchmark_runs,
      ref_fn);

#ifndef DEBUG_MODE
  if (selected_modes == 0) {
    execute_streaming_dynamic_shrink_test();
  }
#endif

  return 0;
}
