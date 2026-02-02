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

void test_q8ta_conv2d_dw(
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
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef layout_int = args.at(idx++);
  const ValueRef impl_selector_int = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  // Extract the layout parameter and cast to GPUMemoryLayout
  int32_t layout_value = graph.extract_scalar<int32_t>(layout_int);
  utils::GPUMemoryLayout layout =
      static_cast<utils::GPUMemoryLayout>(layout_value);

  // Extract the impl_selector flag
  int32_t impl_selector = graph.extract_scalar<int32_t>(impl_selector_int);

  // Create temporary packed int8 tensors for input and output
  TmpTensor packed_int8_input(
      &graph, graph.sizes_of(fp_input), vkapi::kInt8x4, utils::kBuffer, layout);

  TmpTensor packed_int8_output(
      &graph,
      graph.sizes_of(fp_output),
      vkapi::kInt8x4,
      utils::kBuffer,
      layout);

  // Quantize floating point input to packed int8
  add_q8ta_quantize_node(
      graph, fp_input, input_scale, input_zp, packed_int8_input);

  // Build args for conv operator
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
      dilation,
      groups,
      packed_int8_output};

  if (impl_selector) {
    // Use the general quantized conv2d operator for legacy path
    VK_GET_OP_FN("et_vk.conv2d_q8ta_q8csw_q8to.default")(graph, conv_args);
  } else {
    // Use the dedicated depthwise conv2d operator
    VK_GET_OP_FN("etvk.q8ta_conv2d_dw.default")(graph, conv_args);
  }

  // Dequantize packed int8 output to floating point
  add_q8ta_dequantize_node(
      graph, packed_int8_output, output_scale, output_zp, fp_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_q8ta_conv2d_dw.default, test_q8ta_conv2d_dw);
}

} // namespace vkcompute
