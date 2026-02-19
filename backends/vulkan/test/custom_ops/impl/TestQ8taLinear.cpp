/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taQuantizeDequantize.h>

namespace vkcompute {

void test_q8ta_linear(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef output_scale = args.at(idx++);
  const ValueRef output_zp = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  // Create temporary packed int8 tensors for input and output
  // Input uses 4H4W layout to match the linear shader's ivec4 reading pattern
  // where each ivec4 contains data from 4 rows
  TmpTensor packed_int8_input(
      &graph,
      graph.sizes_of(fp_input),
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4H4W);

  // Output uses 4H4W layout to match the linear shader's ivec4 writing pattern
  TmpTensor packed_int8_output(
      &graph,
      graph.sizes_of(fp_output),
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4H4W);

  // Quantize floating point input to packed int8
  add_q8ta_quantize_node(
      graph, fp_input, input_scale, input_zp, packed_int8_input);

  // Call the q8ta_linear operator
  std::vector<ValueRef> linear_args = {
      packed_int8_input,
      input_scale,
      input_zp,
      weight_data,
      weight_sums_data,
      weight_scales_data,
      output_scale,
      output_zp,
      bias_data,
      packed_int8_output};
  VK_GET_OP_FN("et_vk.q8ta_linear.default")(graph, linear_args);

  // Dequantize packed int8 output to floating point
  add_q8ta_dequantize_node(
      graph, packed_int8_output, output_scale, output_zp, fp_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_q8ta_linear.default, test_q8ta_linear);
}

} // namespace vkcompute
