/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taClone.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taQuantizeDequantize.h>

namespace vkcompute {

void test_q8ta_conv2d_transposed(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
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
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef output_padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef activation = args.at(idx++);
  const ValueRef layout_int = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  int32_t layout_value = graph.extract_scalar<int32_t>(layout_int);
  utils::GPUMemoryLayout layout =
      static_cast<utils::GPUMemoryLayout>(layout_value);

  TmpTensor packed_int8_input(
      &graph, graph.sizes_of(fp_input), vkapi::kInt8x4, utils::kBuffer, layout);

  TmpTensor packed_int8_output(
      &graph,
      graph.sizes_of(fp_output),
      vkapi::kInt8x4,
      utils::kBuffer,
      layout);

  add_q8ta_quantize_node(
      graph, fp_input, input_scale, input_zp, packed_int8_input);

  std::vector<ValueRef> conv_args = {
      packed_int8_input,
      input_scale,
      input_zp,
      weight_data,
      weight_sums_data,
      weight_scales_data,
      output_scale,
      output_zp,
      bias_data,
      kernel_size,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      activation,
      packed_int8_output};
  VK_GET_OP_FN("et_vk.q8ta_conv2d_transposed.default")(graph, conv_args);

  add_q8ta_dequantize_node(
      graph, packed_int8_output, output_scale, output_zp, fp_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      test_etvk.test_q8ta_conv2d_transposed.default,
      test_q8ta_conv2d_transposed);
}

} // namespace vkcompute
