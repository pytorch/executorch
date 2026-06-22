/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taQuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taUnary.h>

namespace vkcompute {

void q8ta_unary_test(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef output_scale = args.at(idx++);
  const ValueRef output_zp = args.at(idx++);
  const ValueRef quant_layout_int = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  int32_t layout_value = graph.extract_scalar<int32_t>(quant_layout_int);
  utils::GPUMemoryLayout quant_layout =
      static_cast<utils::GPUMemoryLayout>(layout_value);

  // Create temporary tensor for quantized input
  TmpTensor packed_int8_input(
      &graph,
      graph.sizes_of(fp_input),
      vkapi::kInt8x4,
      utils::kBuffer,
      quant_layout);

  // Create temporary tensor for quantized output
  TmpTensor packed_int8_output(
      &graph,
      graph.sizes_of(fp_input),
      vkapi::kInt8x4,
      utils::kBuffer,
      quant_layout);

  // Quantize: FP -> int8x4
  add_q8ta_quantize_node(
      graph, fp_input, input_scale, input_zp, packed_int8_input);

  // Unary op: int8x4 -> int8x4
  add_q8ta_unary_node(
      graph,
      packed_int8_input,
      input_scale,
      input_zp,
      output_scale,
      output_zp,
      packed_int8_output,
      "relu");

  // Dequantize: int8x4 -> FP
  add_q8ta_dequantize_node(
      graph, packed_int8_output, output_scale, output_zp, fp_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.q8ta_unary_test.default, q8ta_unary_test);
}

} // namespace vkcompute
