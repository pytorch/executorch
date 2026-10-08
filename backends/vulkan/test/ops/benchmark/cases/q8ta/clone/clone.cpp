// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q8ta/clone/clone.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q8ta_clone {

// Utility function to create a test case from a Q8taCloneConfig
TestCase create_test_case_from_config(
    const Q8taCloneConfig& config,
    utils::StorageType storage_type,
    vkapi::ScalarType input_dtype,
    utils::GPUMemoryLayout fp_memory_layout,
    utils::GPUMemoryLayout inp_quant_layout,
    utils::GPUMemoryLayout outp_quant_layout) {
  TestCase test_case;

  // Create a descriptive name for the test case
  // Q-clone-DQ: fp input transitions through int8 layouts and back to fp.
  // Label dtype-wise as i8->i8 since the clone happens in the quantized domain.
  std::string prefix = config.test_case_name; // "ACCU" or "PERF"
  std::string storage_str = repr_str(storage_type, fp_memory_layout) + "->" +
      repr_str(utils::kBuffer, inp_quant_layout) + "->" +
      repr_str(utils::kBuffer, outp_quant_layout);
  std::string test_name = make_test_label(
      prefix,
      dtype_short(vkapi::kChar),
      dtype_short(vkapi::kChar),
      shape_bracket(config.shape),
      storage_str);
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

  // Zero point for quantization
  int32_t zero_point_val = 0;
  ValueSpec zero_point(zero_point_val);

  // Input and output quantized layouts as integers
  int32_t inp_layout_int = static_cast<int32_t>(inp_quant_layout);
  ValueSpec inp_layout_spec(inp_layout_int);

  int32_t outp_layout_int = static_cast<int32_t>(outp_quant_layout);
  ValueSpec outp_layout_spec(outp_layout_int);

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
  test_case.add_input_spec(inp_layout_spec);
  test_case.add_input_spec(outp_layout_spec);
  test_case.add_output_spec(output_tensor);

  test_case.set_abs_tolerance(scale_val + 1e-4);

  // Use layout-only filter for this test since clone IS the operation being
  // tested
  test_case.set_shader_filter({
      "nchw_to",
      "to_nchw",
      "q8ta_quantize",
      "q8ta_dequantize",
  });

  return test_case;
}

// Reference implementation for q8ta_clone operation
// Since clone just copies data, the result should be the same as
// quantize-dequantize
void q8ta_clone_reference_impl(TestCase& test_case) {
  int32_t idx = 0;
  const ValueSpec& input_spec = test_case.inputs()[idx++];
  const ValueSpec& scale_spec = test_case.inputs()[idx++];
  const ValueSpec& zero_point_spec = test_case.inputs()[idx++];
  const ValueSpec& inp_layout_spec = test_case.inputs()[idx++];
  const ValueSpec& outp_layout_spec = test_case.inputs()[idx++];
  (void)inp_layout_spec; // Not used in reference implementation
  (void)outp_layout_spec; // Not used in reference implementation

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
          "One or more dimensions exceed the allowed limit for reference "
          "implementation.");
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

  // Perform quantize-clone-dequantize operation on each element
  // Clone preserves the quantized values, so result is same as Q-DQ
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

std::string get_prefix(const std::vector<int64_t>& shape) {
  std::string prefix = "ACCU";
  for (const auto& dim : shape) {
    if (dim > kRefDimSizeLimit) {
      prefix = "PERF";
      break;
    }
  }
  return prefix;
}

} // namespace q8ta_clone
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
