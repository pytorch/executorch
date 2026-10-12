/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taQuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>

namespace vkcompute {

void q_dq_8bit_per_tensor(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef scale = args.at(idx++);
  const ValueRef zero_point = args.at(idx++);
  const ValueRef layout_int = args.at(idx++);
  const ValueRef impl_selector_str = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  // Extract the layout parameter and cast to GPUMemoryLayout
  int32_t layout_value = graph.extract_scalar<int32_t>(layout_int);
  utils::GPUMemoryLayout layout =
      static_cast<utils::GPUMemoryLayout>(layout_value);

  // Extract the impl_selector string
  std::string impl_selector = graph.extract_string(impl_selector_str);

  // Use legacy 4W4C implementation if requested and layout matches
  if (impl_selector == "legacy_4w4c" && layout == utils::kPackedInt8_4W4C) {
    TmpTensor packed_int8_input(
        &graph,
        graph.sizes_of(fp_input),
        vkapi::kInt8x4,
        utils::kBuffer,
        utils::kPackedInt8_4W4C);

    add_quantize_and_pack_4w4c_node(
        graph, fp_input, scale, zero_point, packed_int8_input);

    add_unpack_4w4c_and_dequantize_node(
        graph, packed_int8_input, scale, zero_point, fp_output);
  } else {
    // Create temporary tensor with the specified layout
    TmpTensor packed_int8_input(
        &graph,
        graph.sizes_of(fp_input),
        vkapi::kInt8x4,
        utils::kBuffer,
        layout);

    // Use unified block-based dispatch
    add_q8ta_quantize_node(
        graph, fp_input, scale, zero_point, packed_int8_input);

    add_q8ta_dequantize_node(
        graph, packed_int8_input, scale, zero_point, fp_output);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.q_dq_8bit_per_tensor.default, q_dq_8bit_per_tensor);
}

} // namespace vkcompute
