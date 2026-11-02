/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taClone.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taQuantizeDequantize.h>

namespace vkcompute {

void q8ta_clone_test(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef scale = args.at(idx++);
  const ValueRef zero_point = args.at(idx++);
  const ValueRef inp_layout_int = args.at(idx++);
  const ValueRef outp_layout_int = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  // Extract the layout parameters and cast to GPUMemoryLayout
  int32_t inp_layout_value = graph.extract_scalar<int32_t>(inp_layout_int);
  utils::GPUMemoryLayout inp_layout =
      static_cast<utils::GPUMemoryLayout>(inp_layout_value);

  int32_t outp_layout_value = graph.extract_scalar<int32_t>(outp_layout_int);
  utils::GPUMemoryLayout outp_layout =
      static_cast<utils::GPUMemoryLayout>(outp_layout_value);

  // Create temporary tensor for quantized input with input layout
  TmpTensor packed_int8_input(
      &graph,
      graph.sizes_of(fp_input),
      vkapi::kInt8x4,
      utils::kBuffer,
      inp_layout);

  // Create temporary tensor for quantized output with output layout
  TmpTensor packed_int8_output(
      &graph,
      graph.sizes_of(fp_input),
      vkapi::kInt8x4,
      utils::kBuffer,
      outp_layout);

  // Quantize: FP -> int8x4 with input layout
  add_q8ta_quantize_node(graph, fp_input, scale, zero_point, packed_int8_input);

  // Clone: int8x4 (input layout) -> int8x4 (output layout)
  add_q8ta_clone_node(graph, packed_int8_input, packed_int8_output);

  // Dequantize: int8x4 with output layout -> FP
  add_q8ta_dequantize_node(
      graph, packed_int8_output, scale, zero_point, fp_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.q8ta_clone_test.default, q8ta_clone_test);
}

} // namespace vkcompute
