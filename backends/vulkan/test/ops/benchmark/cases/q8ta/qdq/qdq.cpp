// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/qdq/qdq.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_qdq {

// Utility function to create a test case from a QDQ8BitConfig
TestCase create_test_case_from_config(
    const QDQ8BitConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype,
    utils::GPUMemoryLayout fp_memory_layout,
    utils::GPUMemoryLayout quantized_memory_layout,
    const std::string& impl_selector) {
  TestCase test_case;

  // Create a descriptive name for the test case
  // QDQ: fp input -> int8 staging -> fp output (label as f32->f32)
  std::string prefix = config.test_case_name; // "ACCU" or "PERF"
  std::string storage_str = repr_str(storage_type, fp_memory_layout) + "->" +
      repr_str(utils::kBuffer, quantized_memory_layout);
  std::string suffix = impl_selector.empty() ? "" : "[" + impl_selector + "]";
  std::string test_name = make_test_label(
      prefix,
      dtype_short(input_dtype),
      dtype_short(input_dtype),
      shape_bracket(config.shape),
      storage_str,
      suffix);
  test_case.set_name(test_name);

  // Set the operator name for the test case
  std::string operator_name = "test_etvk." + config.op_name + ".default";
  test_case.set_operator_name(operator_name);

  // Input tensor (float) - any dimensionality
  ValueSpec input_tensor(
      config.shape,
      input_dtype,
      storage_type,
      fp_memory_layout,
      DataGenType::RANDOM);

  float scale_val = 0.007112;
  ValueSpec scale(scale_val);

  // Generate random zero point within quantization range
  int32_t zero_point_val = 0;
  ValueSpec zero_point(zero_point_val);

  // GPUMemoryLayout as integer (will be cast in the operator)
  int32_t layout_int = static_cast<int32_t>(quantized_memory_layout);
  ValueSpec layout_spec(layout_int);

  // impl_selector string
  ValueSpec impl_selector_spec = ValueSpec::make_string(impl_selector);

  // Output tensor (float) - same shape as input
  ValueSpec output_tensor(
      config.shape,
      input_dtype,
      storage_type,
      fp_memory_layout,
      DataGenType::ZEROS);

  // Add all specs to test case
  test_case.add_input_spec(input_tensor);
  test_case.add_input_spec(scale);
  test_case.add_input_spec(zero_point);
  test_case.add_input_spec(layout_spec);
  test_case.add_input_spec(impl_selector_spec);
  test_case.add_output_spec(output_tensor);

  test_case.set_abs_tolerance(scale_val + 1e-4);

  // Use layout-only filter for this test since quantize/dequantize ARE the
  // operations being tested, not overhead
  test_case.set_shader_filter(kLayoutOnlyShaderFilter);

  return test_case;
}

// Reference implementation for q_dq_8bit operation
void q_dq_8bit_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& scale_spec = test_case.inputs()[idx++];
  const ValueSpec& zero_point_spec = test_case.inputs()[idx++];
  const ValueSpec& layout_spec = test_case.inputs()[idx++];
  (void)layout_spec; // Not used in reference implementation

  // Extract output specification
  ValueSpec& output_spec = test_case.outputs()[0];

  // Get tensor dimensions (arbitrary dimensionality)
  auto input_sizes = input_spec.get_tensor_sizes();

  // Calculate total number of elements
  int64_t num_elements = 1;
  for (const auto& dim : input_sizes) {
    num_elements *= dim;
  }

  // Skip for large tensors since computation time will be extremely slow
  for (const auto& dim : input_sizes) {
    if (dim > kRefDimSizeLimit) {
      throw std::invalid_argument(
          "One or more dimensions exceed the allowed limit for reference implementation.");
    }
  }

  if (input_spec.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  // Get raw data pointers
  auto& input_data = input_spec.get_float_data();

  // Extract the randomized scale and zero point values
  float scale = scale_spec.get_float_value();
  int32_t zero_point = zero_point_spec.get_int_value();
  int32_t quant_min = -128;
  int32_t quant_max = 127;

  // Prepare output data
  auto& ref_data = output_spec.get_ref_float_data();
  ref_data.resize(num_elements);

  // Perform quantize-dequantize operation on each element
  for (int64_t i = 0; i < num_elements; ++i) {
    float input_val = input_data[i];

    // Quantize: quantized = round(input / scale + zero_point)
    float quantized_float = std::round(input_val / scale) + zero_point;

    // Clamp to quantization range
    quantized_float = std::max(quantized_float, static_cast<float>(quant_min));
    quantized_float = std::min(quantized_float, static_cast<float>(quant_max));

    int32_t quantized_int = static_cast<int32_t>(quantized_float);

    // Dequantize: output = (quantized - zero_point) * scale
    float dequantized = (quantized_int - zero_point) * scale;

    ref_data[i] = dequantized;
  }
}

} // namespace q8ta_qdq
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
