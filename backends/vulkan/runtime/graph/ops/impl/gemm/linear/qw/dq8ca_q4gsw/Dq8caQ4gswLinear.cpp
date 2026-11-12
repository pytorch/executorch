/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/dq8ca_q4gsw/Dq8caQ4gswLinear.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

vkapi::ShaderInfo pick_linear_dqa_qw_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef int_input = args.at(1).refs.at(1);
  (void)int_input;
  const ValueRef input_zp = args.at(1).refs.at(4);
  const ValueRef int_weight = args.at(1).refs.at(5);

  const bool weight_is_4bit = resize_args.at(0) != kDummyValueRef;
  const bool is_gemv_case = is_gemv(graph, fp_input);

  // Use the coopmat<int8> shader for 4-bit dq8ca dispatches when the device
  // enumerates VK_COMPONENT_TYPE_SINT8_KHR in its cooperative matrix property
  // list and the shape aligns; tiled otherwise.
  if (weight_is_4bit && !is_gemv_case &&
      graph->context()->adapter_ptr()->supports_int8_cooperative_matrix()) {
    const int64_t group_size =
        graph->extract_scalar<int64_t>(resize_args.at(0));
    if (can_use_q4gsw_coopmat(
            graph,
            out,
            fp_input,
            group_size,
            resize_args.at(2),
            kDq8caQ4gswCoopmatDims.m,
            kDq8caQ4gswCoopmatDims.n,
            kDq8caQ4gswCoopmatDims.k)) {
      std::string kernel_name = "linear_dq8ca_q4gsw_coopmat";
      add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
      add_storage_type_suffix(kernel_name, graph->storage_type_of(int_weight));
      add_dtype_suffix(kernel_name, graph->dtype_of(out));
      return VK_KERNEL_FROM_STR(kernel_name);
    }
  }

  std::string kernel_name = "linear_dq8ca_q4gsw";
  kernel_name += is_gemv_case ? "_coop" : "_tiled";
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(int_weight));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  add_zp_dtype_mode_suffix(kernel_name, graph->dtype_of(input_zp));

  return VK_KERNEL_FROM_STR(kernel_name);
}

void add_linear_dqa_qw_node(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef packed_int_input,
    const ValueRef int_input_sums,
    const ValueRef packed_input_scale,
    const ValueRef packed_input_zp,
    const ValueRef input_scale_data,
    const ValueRef input_zp_data,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef output) {
  (void)input_scale_data;
  (void)input_zp_data;
  VK_CHECK_COND(input_quant_config.granularity == kPerChannel);
  VK_CHECK_COND(input_quant_config.nbits == 8);
  VK_CHECK_COND(input_quant_config.is_dynamic);

  VK_CHECK_COND(weight_quant_config.granularity == kPerGroup);
  VK_CHECK_COND(weight_quant_config.is_symmetric);
  VK_CHECK_COND(weight_quant_config.nbits == 4);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(fp_input)};

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  int32_t K4_per_group = 0;
  int32_t coopmat_k_iters = 0;
  const int32_t K_dim = graph.size_at<int32_t>(-1, fp_input);
  if (weight_quant_config.nbits == 4) {
    int32_t group_size_val = graph.extract_scalar<int32_t>(group_size);
    K4_per_group = utils::div_up(group_size_val, int32_t(4));
    coopmat_k_iters = K_dim / group_size_val;
  }

  const ValueRef is_4bit_flag =
      weight_quant_config.nbits == 4 ? group_size : kDummyValueRef;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_dqa_qw_shader,
      quantized_linear_gwg,
      quantized_linear_lwg,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{fp_input,
         packed_int_input,
         int_input_sums,
         packed_input_scale,
         packed_input_zp,
         packed_weight,
         packed_weight_sums,
         packed_weight_scales,
         packed_bias},
        vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      // 4th spec const: output width N for coopMatStore (see
      // add_linear_qw_node).
      {apply_bias,
       K4_per_group,
       coopmat_k_iters,
       graph.size_at<int32_t>(-1, output)},
      // Resize args (resize_args.at(2) = bias_data, read by the coopmat gate)
      {is_4bit_flag, weight_data, bias_data},
      // Resizing Logic
      resize_linear_qw_node));
}

void linear_dq8ca_q4gsw(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef group_size = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size);

  QuantizationConfig input_quant_config(8, kPerChannel, {}, false, true);
  QuantizationConfig weight_quant_config(4, kPerGroup, {group_size_val});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      input_scale,
      input_zp,
      weight_data,
      weight_sums_data,
      weight_scales_data,
      kDummyValueRef, // weight_zeros_data
      group_size, // group_size
      bias_data,
      output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_dq8ca_q4gsw.default, linear_dq8ca_q4gsw);
}

} // namespace vkcompute
