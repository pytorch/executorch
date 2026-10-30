// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/unary/unary.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_unary {

TestCase create_test_case_from_config(
    const Q8taUnaryConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype,
    utils::GPUMemoryLayout fp_memory_layout,
    utils::GPUMemoryLayout quant_layout) {
  TestCase test_case;

  std::string prefix = config.test_case_name; // "ACCU" or "PERF"
  std::string storage_str = repr_str(storage_type, fp_memory_layout) + "->" +
      repr_str(utils::kBuffer, quant_layout);
  std::string test_name = make_test_label(
      prefix,
      dtype_short(input_dtype),
      dtype_short(input_dtype),
      shape_bracket(config.shape),
      storage_str);
  test_case.set_name(test_name);

  std::string operator_name = "test_etvk." + config.op_name + ".default";
  test_case.set_operator_name(operator_name);

  // Input tensor (float)
  ValueSpec input_tensor(
      config.shape,
      input_dtype,
      storage_type,
      fp_memory_layout,
      DataGenType::RANDOM);

  float scale_val = 0.007112;
  ValueSpec input_scale(scale_val);

  int32_t zero_point_val = 0;
  ValueSpec input_zero_point(zero_point_val);

  // For relu, output scale and zero point can differ from input
  float output_scale_val = 0.007112;
  ValueSpec output_scale(output_scale_val);

  int32_t output_zp_val = 0;
  ValueSpec output_zero_point(output_zp_val);

  int32_t layout_int = static_cast<int32_t>(quant_layout);
  ValueSpec layout_spec(layout_int);

  // Output tensor (float) - same shape as input
  ValueSpec output_tensor(
      config.shape,
      input_dtype,
      storage_type,
      fp_memory_layout,
      DataGenType::ZEROS);

  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(input_scale);
  test_case.add_input_spec(input_zero_point);
  test_case.add_input_spec(output_scale);
  test_case.add_input_spec(output_zero_point);
  test_case.add_input_spec(layout_spec);
  test_case.add_output_spec(output_tensor);

  test_case.set_abs_tolerance(scale_val + 1e-4);

  test_case.set_shader_filter({
      "nchw_to",
      "to_nchw",
      "q8ta_quantize",
      "q8ta_dequantize",
  });

  return test_case;
}

// Reference implementation: quantize -> relu -> dequantize
void q8ta_unary_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& input_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& input_zp_spec = test_case.inputs()[idx++];
  const ValueSpec& output_scale_spec = test_case.inputs()[idx++];
  const ValueSpec& output_zp_spec = test_case.inputs()[idx++];
  const ValueSpec& layout_spec = test_case.inputs()[idx++];
  (void)layout_spec;

  ValueSpec& output_spec = test_case.outputs()[0];

  auto input_sizes = input_spec.get_tensor_sizes();

  int64_t num_elements = 1;
  for (const auto& dim : input_sizes) {
    num_elements *= dim;
  }

  for (const auto& dim : input_sizes) {
    if (dim > kRefDimSizeLimit) {
      throw std::invalid_argument(
          "One or more dimensions exceed the allowed limit for reference "
          "implementation.");
    }
  }

  if (input_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  auto& input_data = input_spec.get_float_data();

  float input_scale = input_scale_spec.get_float_value();
  int32_t input_zp = input_zp_spec.get_int_value();
  float output_scale = output_scale_spec.get_float_value();
  int32_t output_zp = output_zp_spec.get_int_value();
  int32_t quant_min = -128;
  int32_t quant_max = 127;

  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_elements);

  for (int64_t i = 0; i < num_elements; ++i) {
    float input_val = input_data[i];

    // Quantize with input scale/zp
    float quantized_float = std::round(input_val / input_scale) + input_zp;
    quantized_float = std::max(quantized_float, static_cast<float>(quant_min));
    quantized_float = std::min(quantized_float, static_cast<float>(quant_max));
    int32_t quantized_int = static_cast<int32_t>(quantized_float);

    // Dequantize to float
    float dequantized = (quantized_int - input_zp) * input_scale;

    // Apply ReLU
    float activated = std::max(dequantized, 0.0f);

    // Requantize with output scale/zp
    float requantized_float = std::round(activated / output_scale) + output_zp;
    requantized_float =
        std::max(requantized_float, static_cast<float>(quant_min));
    requantized_float =
        std::min(requantized_float, static_cast<float>(quant_max));
    int32_t requantized_int = static_cast<int32_t>(requantized_float);

    // Dequantize back to float for comparison
    ref_data[i] = (requantized_int - output_zp) * output_scale;
  }
}

} // namespace q8ta_unary
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
