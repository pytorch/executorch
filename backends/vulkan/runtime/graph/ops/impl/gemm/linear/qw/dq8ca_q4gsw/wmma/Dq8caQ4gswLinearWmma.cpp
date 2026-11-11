/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/dq8ca_q4gsw/wmma/Dq8caQ4gswLinearWmma.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/q4gsw/Q4gswLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/wmma/WmmaLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

namespace {

vkapi::ShaderInfo pick_dq8ca_q4gsw_linear_wmma_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef input_zp = args.at(1).refs.at(4);
  const ValueRef packed_weight = args.at(1).refs.at(5);

  const ValueRef packed_int_input = args.at(1).refs.at(1);

  const CoopmatDeviceKey device = coopmat_device_key(graph);
  // resize_args: {group_size, weight_data, bias_data}
  WmmaShape shape = wmma_shape(
      *graph,
      WmmaOp::kDq8caQ4gsw,
      fp_input,
      output,
      resize_args.at(0),
      resize_args.at(2),
      graph->storage_type_of(packed_weight));
  shape.int8_layout = graph->estimate_memory_layout_of(packed_int_input) ==
          utils::kPackedInt8_4W
      ? WmmaInt8Layout::kRowMajor
      : WmmaInt8Layout::k4H4W;
  std::optional<WmmaChoice> choice = select_wmma_kernel(device, shape);
  if (!choice.has_value() && shape.int8_layout == WmmaInt8Layout::kRowMajor &&
      !shape.gemv) {
    // The row-major activation layout was chosen at build time and, of the
    // M > 1 kernels, only the row-major family kernels read it, so keep one
    // for unaligned M: the grid rounds M up. Decode (M == 1) takes the
    // linear_dq8ca_q4gsw_coop kernel below, which reads the fp activations
    // (the quantize/pack node dispatches nothing at M == 1).
    shape.ignore_alignment = true;
    choice = select_wmma_kernel(device, shape);
    VK_CHECK_COND(
        choice.has_value(), "no row-major dq8ca_q4gsw kernel for this shape");
  }

  std::string kernel_name;
  if (choice.has_value()) {
    kernel_name = choice->kernel_base;
  } else {
    // The kernels for the 4h4w layout (as in pick_linear_dqa_qw_shader).
    kernel_name = "linear_dq8ca_q4gsw";
    kernel_name += is_gemv(graph, fp_input) ? "_coop" : "_tiled";
  }
  add_storage_type_suffix(kernel_name, graph->storage_type_of(output));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph->dtype_of(output));
  add_zp_dtype_mode_suffix(kernel_name, graph->dtype_of(input_zp));
  return VK_KERNEL_FROM_STR(kernel_name);
}

} // namespace

bool try_add_dq8ca_q4gsw_linear_wmma_node(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  const ValueRef fp_input = args.at(0);
  const ValueRef input_scale = args.at(1);
  const ValueRef input_zp = args.at(2);
  const ValueRef weight_data = args.at(3);
  const ValueRef weight_sums_data = args.at(4);
  const ValueRef weight_scales_data = args.at(5);
  const ValueRef group_size = args.at(6);
  const ValueRef bias_data = args.at(7);
  const ValueRef output = args.at(8);

  if (graph.dtype_of(fp_input) != vkapi::kHalf) {
    return false;
  }
  // The decode and fallback kernels of this path use int8 dot products.
  if (!graph.can_use_int8_dot_product()) {
    return false;
  }
  const CoopmatDeviceKey device = coopmat_device_key(&graph);
  const WmmaShape shape = wmma_shape(
      graph,
      WmmaOp::kDq8caQ4gsw,
      fp_input,
      output,
      group_size,
      bias_data,
      predicted_q4_weight_storage(graph, weight_data));
  // Take the op over only if a row applies to its build-time shape; otherwise
  // it stays entirely on the existing path. The kernel chosen here also fixes
  // the int8 activation layout.
  const std::optional<WmmaChoice> build_choice =
      select_wmma_kernel(device, shape);
  if (!build_choice.has_value()) {
    return false;
  }
  const bool rowmajor = build_choice->rowmajor_a;

  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size);
  const QuantizationConfig input_quant_config(8, kPerChannel, {}, false, true);
  const QuantizationConfig weight_quant_config(4, kPerGroup, {group_size_val});

  const ValueRef packed_weight =
      prepack_quantized_linear_weight(graph, weight_quant_config, weight_data);
  // The kernels read the scales in the output dtype; fp32 scales of an fp16
  // model are converted here, as on the existing path.
  const ValueRef packed_weight_scales =
      prepack_q4_scales(graph, weight_scales_data, graph.dtype_of(output));
  const ValueRef packed_weight_sums = prepack_standard(
      graph, weight_sums_data, utils::kBuffer, utils::kWidthPacked);

  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);
  ValueRef packed_bias = dummy_bias.vref;
  uint32_t apply_bias = 0;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
    apply_bias = 1;
  }

  ValueRef packed_input_scale = input_scale;
  ValueRef packed_input_zp = input_zp;
  if (graph.val_is_tref(input_scale)) {
    VK_CHECK_COND(graph.val_is_tref(input_zp));
    packed_input_scale = prepack_standard(
        graph, input_scale, utils::kTexture3D, utils::kWidthPacked);
    packed_input_zp = prepack_standard(
        graph, input_zp, utils::kTexture3D, utils::kWidthPacked);
  }

  TmpTensor packed_int_input(
      &graph,
      graph.sizes_of(fp_input),
      vkapi::kInt8x4,
      utils::kBuffer,
      rowmajor ? utils::kPackedInt8_4W : utils::kPackedInt8_4H4W);

  // Per-group int8 input sums, indexed as ivec4[group_idx * M4 + m4].
  const int64_t num_groups = graph.size_at<int64_t>(-2, weight_scales_data);
  const int64_t M4 = utils::div_up(shape.M, int64_t(4));
  TmpTensor int_input_sums(
      &graph,
      {num_groups * M4 * 4},
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);

  add_quantize_and_pack_4h4w_with_group_sums_node(
      graph,
      input_quant_config,
      fp_input,
      int_input_sums,
      packed_input_scale,
      packed_input_zp,
      packed_int_input,
      group_size);

  const int32_t K4_per_group =
      utils::div_up(static_cast<int32_t>(group_size_val), int32_t(4));
  const int32_t num_groups_arg = graph.size_at<int32_t>(-1, fp_input) /
      static_cast<int32_t>(group_size_val);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_dq8ca_q4gsw_linear_wmma_shader,
      wmma_linear_gwg,
      quantized_linear_lwg,
      {{output, vkapi::kWrite},
       {{fp_input,
         packed_int_input.vref,
         int_input_sums.vref,
         packed_input_scale,
         packed_input_zp,
         packed_weight,
         packed_weight_sums,
         packed_weight_scales,
         packed_bias},
        vkapi::kRead}},
      {graph.sizes_ubo(output), graph.sizes_ubo(fp_input)},
      {},
      // The spec constants of add_linear_dqa_qw_node.
      {apply_bias,
       K4_per_group,
       num_groups_arg,
       graph.size_at<int32_t>(-1, output)},
      {group_size, weight_data, bias_data},
      resize_linear_qw_node));
  return true;
}

} // namespace vkcompute
