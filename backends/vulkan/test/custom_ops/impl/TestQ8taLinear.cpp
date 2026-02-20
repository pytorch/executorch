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
  const ValueRef impl_selector_str = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  std::string impl_selector = graph.extract_string(impl_selector_str);

  if (impl_selector == "gemv") {
    // Use 4W layout for gemv variant
    TmpTensor packed_int8_input(
        &graph,
        graph.sizes_of(fp_input),
        vkapi::kInt8x4,
        utils::kBuffer,
        utils::kPackedInt8_4W);

    TmpTensor packed_int8_output(
        &graph,
        graph.sizes_of(fp_output),
        vkapi::kInt8x4,
        utils::kBuffer,
        utils::kPackedInt8_4W);

    add_q8ta_quantize_node(
        graph, fp_input, input_scale, input_zp, packed_int8_input);

    ValueRef activation_str = graph.add_string("none");
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
        activation_str,
        packed_int8_output};
    VK_GET_OP_FN("et_vk.q8ta_linear_gemv.default")(graph, linear_args);

    add_q8ta_dequantize_node(
        graph, packed_int8_output, output_scale, output_zp, fp_output);
  } else {
    // Default: use 4H4W layout for tiled variant
    TmpTensor packed_int8_input(
        &graph,
        graph.sizes_of(fp_input),
        vkapi::kInt8x4,
        utils::kBuffer,
        utils::kPackedInt8_4H4W);

    TmpTensor packed_int8_output(
        &graph,
        graph.sizes_of(fp_output),
        vkapi::kInt8x4,
        utils::kBuffer,
        utils::kPackedInt8_4H4W);

    add_q8ta_quantize_node(
        graph, fp_input, input_scale, input_zp, packed_int8_input);

    ValueRef activation_str = graph.add_string("none");
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
        activation_str,
        packed_int8_output};
    VK_GET_OP_FN("et_vk.q8ta_linear.default")(graph, linear_args);

    add_q8ta_dequantize_node(
        graph, packed_int8_output, output_scale, output_zp, fp_output);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_q8ta_linear.default, test_q8ta_linear);
}

} // namespace vkcompute
