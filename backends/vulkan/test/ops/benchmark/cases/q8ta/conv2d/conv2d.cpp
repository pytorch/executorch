// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/conv2d/conv2d.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_conv2d {

namespace {

// Utility function to create a test case from a Conv2dConfig
TestCase create_test_case_from_config_with_layouts(
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

} // namespace

TestCase create_test_case_from_config(
    const Conv2dConfig& config,
    vkapi::ScalarType input_dtype,
    utils::StorageType fp_storage_type,
    utils::GPUMemoryLayout int8_memory_layout,
    const std::string& impl_selector,
    const Im2colUnsignedTestOptions* im2col_options,
    const float input_scale_val,
    const DataGenType input_data_gen) {
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

// SceneX route tests. The kPackedInt8_4C input + kPackedInt8_4W4C output
// layout combination is only exercised here (default generators never pair
// them); run via the scenex_regular and scenex_grouped sets. Zero
// tolerances are exact by construction: both pipelines accumulate in int32
// with identical requantize, so any mismatch is a real regression, not noise.
TestCase create_scenex_test_case(
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

// Custom FLOP calculator for quantized conv2d operation
int64_t quantized_conv2d_flop_calculator(const TestCase& test_case) {
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

namespace {

// Reference implementation for activation, weight, and output quantized conv2d
void conv2d_q8ta_q8csw_q8to_reference_impl(TestCase& test_case) {
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

} // namespace

void reference_impl(TestCase& test_case) {
  conv2d_q8ta_q8csw_q8to_reference_impl(test_case);
}

// The impl selector holds one of "", "general", or "im2col"; activation and
// other string inputs use disjoint values. Overwrite it by value so a
// reordered input list fails loudly instead of mutating the wrong spec.
// Note: for route == "direct" the measured run already forces "general", so
// this reference re-executes the identical implementation and only checks
// determinism; genuine cross-implementation correctness comes from the
// auto/im2col legs.
void scenex_direct_reference(TestCase& test_case) {
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

} // namespace q8ta_conv2d
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
