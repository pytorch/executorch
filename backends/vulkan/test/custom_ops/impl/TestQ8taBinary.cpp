/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taBinary.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taQuantizeDequantize.h>

namespace vkcompute {

void q8ta_add_test(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input_a = args.at(idx++);
  const ValueRef fp_input_b = args.at(idx++);
  const ValueRef input_a_scale = args.at(idx++);
  const ValueRef input_a_zp = args.at(idx++);
  const ValueRef input_b_scale = args.at(idx++);
  const ValueRef input_b_zp = args.at(idx++);
  const ValueRef output_scale = args.at(idx++);
  const ValueRef output_zp = args.at(idx++);
  const ValueRef alpha = args.at(idx++);
  const ValueRef quant_layout_int = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  // Extract the layout parameter and cast to GPUMemoryLayout
  int32_t layout_value = graph.extract_scalar<int32_t>(quant_layout_int);
  utils::GPUMemoryLayout quant_layout =
      static_cast<utils::GPUMemoryLayout>(layout_value);

  // Create temporary tensors for quantized data with the specified layout
  TmpTensor packed_int8_input_a(
      &graph,
      graph.sizes_of(fp_input_a),
      vkapi::kInt8x4,
      utils::kBuffer,
      quant_layout);

  TmpTensor packed_int8_input_b(
      &graph,
      graph.sizes_of(fp_input_b),
      vkapi::kInt8x4,
      utils::kBuffer,
      quant_layout);

  TmpTensor packed_int8_output(
      &graph,
      graph.sizes_of(fp_output),
      vkapi::kInt8x4,
      utils::kBuffer,
      quant_layout);

  // Quantize: FP -> int8x4 with specified layout
  add_q8ta_quantize_node(
      graph, fp_input_a, input_a_scale, input_a_zp, packed_int8_input_a);

  add_q8ta_quantize_node(
      graph, fp_input_b, input_b_scale, input_b_zp, packed_int8_input_b);

  // Binary add: int8x4 -> int8x4 (same layout for all tensors)
  add_q8ta_binary_node(
      graph,
      packed_int8_input_a,
      packed_int8_input_b,
      input_a_scale,
      input_a_zp,
      input_b_scale,
      input_b_zp,
      output_scale,
      output_zp,
      alpha,
      packed_int8_output,
      "add");

  // Dequantize: int8x4 -> FP
  add_q8ta_dequantize_node(
      graph, packed_int8_output, output_scale, output_zp, fp_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.q8ta_add.test, q8ta_add_test);
}

} // namespace vkcompute
