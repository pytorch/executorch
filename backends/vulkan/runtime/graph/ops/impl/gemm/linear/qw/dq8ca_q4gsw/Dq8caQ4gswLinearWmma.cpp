/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/dq8ca_q4gsw/Dq8caQ4gswLinearWmma.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/WmmaLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

namespace {

// Dispatch node args:
// {{output}, {fp_input, packed_int_input, int_input_sums, input_scale,
//   input_zp, packed_weight, weight_sums, weight_scales, bias}}
// resize_args: {group_size, weight_data, bias_data}
WmmaShape dq8ca_runtime_shape(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef packed_int_input = args.at(1).refs.at(1);
  const ValueRef packed_weight = args.at(1).refs.at(5);

  const std::vector<int64_t> out_sizes = graph->sizes_of(output);
  WmmaShape s;
  s.op = WmmaOp::kDq8caQ4gsw;
  s.N = utils::val_at(-1, out_sizes);
  s.M = utils::val_at(-2, out_sizes);
  s.K = utils::val_at(-1, graph->sizes_of(fp_input));
  s.group_size = graph->extract_scalar<int64_t>(resize_args.at(0));
  for (int64_t d = 0; d < graph->dim_of(output) - 2; d++) {
    s.batch *= utils::val_at(d, out_sizes);
  }
  s.gemv = is_gemv(graph, fp_input);
  s.has_bias = !graph->val_is_none(resize_args.at(2));
  s.half = graph->dtype_of(output) == vkapi::kHalf;
  s.input = to_wmma_storage(graph->storage_type_of(fp_input));
  s.output = to_wmma_storage(graph->storage_type_of(output));
  s.weight = to_wmma_storage(graph->storage_type_of(packed_weight));
  s.io_width_packed = graph->packed_dim_of(output) == WHCN::kWidthDim &&
      graph->packed_dim_of(fp_input) == WHCN::kWidthDim;
  s.int8_layout = graph->estimate_memory_layout_of(packed_int_input) ==
          utils::kPackedInt8_4W
      ? WmmaInt8Layout::kRowMajor
      : WmmaInt8Layout::k4H4W;
  return s;
}

// args: {input, input_scale, input_zp, weight, weight_sums, weight_scales,
//        group_size, bias, output}
WmmaShape dq8ca_build_time_shape(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  const ValueRef fp_input = args.at(0);
  const ValueRef output = args.at(8);
  const std::vector<int64_t> out_sizes = graph.sizes_of(output);
  WmmaShape s;
  s.op = WmmaOp::kDq8caQ4gsw;
  s.N = utils::val_at(-1, out_sizes);
  s.M = utils::val_at(-2, out_sizes);
  s.K = utils::val_at(-1, graph.sizes_of(fp_input));
  s.group_size = graph.extract_scalar<int64_t>(args.at(6));
  for (int64_t d = 0; d < graph.dim_of(output) - 2; d++) {
    s.batch *= utils::val_at(d, out_sizes);
  }
  s.gemv = is_gemv(&graph, fp_input);
  s.has_bias = !graph.val_is_none(args.at(7));
  s.half = graph.dtype_of(output) == vkapi::kHalf;
  s.input = to_wmma_storage(graph.storage_type_of(fp_input));
  s.output = to_wmma_storage(graph.storage_type_of(output));
  s.weight = to_wmma_storage(predicted_q4_weight_storage(graph, args.at(3)));
  s.io_width_packed = graph.packed_dim_of(output) == WHCN::kWidthDim &&
      graph.packed_dim_of(fp_input) == WHCN::kWidthDim;
  s.int8_layout = WmmaInt8Layout::kAny;
  return s;
}

vkapi::ShaderInfo pick_dq8ca_q4gsw_linear_wmma_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef input_zp = args.at(1).refs.at(4);
  const ValueRef packed_weight = args.at(1).refs.at(5);

  const CoopmatDeviceKey device = coopmat_device_key(graph);
  WmmaShape shape = dq8ca_runtime_shape(graph, args, resize_args);
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
    add_storage_type_suffix(kernel_name, graph->storage_type_of(output));
    add_storage_type_suffix(kernel_name, graph->storage_type_of(packed_weight));
    add_dtype_suffix(kernel_name, graph->dtype_of(output));
  } else {
    // The kernels for the 4h4w layout (as in pick_linear_dqa_qw_shader).
    kernel_name = "linear_dq8ca_q4gsw";
    kernel_name += is_gemv(graph, fp_input) ? "_coop" : "_tiled";
    add_storage_type_suffix(kernel_name, graph->storage_type_of(output));
    add_storage_type_suffix(kernel_name, graph->storage_type_of(packed_weight));
    add_dtype_suffix(kernel_name, graph->dtype_of(output));
    add_zp_dtype_mode_suffix(kernel_name, graph->dtype_of(input_zp));
  }
  return VK_KERNEL_FROM_STR(kernel_name);
}

// Row-major (utils::kPackedInt8_4W) counterpart of
// add_quantize_and_pack_4h4w_with_group_sums_node: same quantization, group
// sums and workgroup shape.
void add_quantize_and_pack_4w_with_group_sums_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef int_input_sums,
    const ValueRef packed_input_scales,
    const ValueRef packed_input_zps,
    const ValueRef packed_int_input,
    const ValueRef group_size) {
  const int32_t group_size_val = graph.extract_scalar<int32_t>(group_size);
  const int32_t blocks_per_group = utils::div_up(group_size_val, int32_t(4));

  std::string shader_name = "quantize_and_pack_4w_with_group_sums";
  shader_name += group_size_val >= 128 ? "_o2w32" : "_o4w16";
  add_storage_type_suffix(shader_name, graph.storage_type_of(packed_int_input));
  add_storage_type_suffix(shader_name, graph.storage_type_of(fp_input));
  add_dtype_suffix(shader_name, graph.dtype_of(fp_input));
  add_zp_dtype_mode_suffix(shader_name, graph.dtype_of(packed_input_zps));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(shader_name),
      pick_quantize_and_pack_4h4w_with_group_sums_gwg,
      pick_required_lwg,
      {{{packed_int_input, int_input_sums}, vkapi::kWrite},
       {{fp_input, packed_input_scales, packed_input_zps}, vkapi::kRead}},
      {graph.sizes_ubo(fp_input)},
      {},
      {blocks_per_group},
      {group_size}));
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
  const WmmaShape shape = dq8ca_build_time_shape(graph, args);
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
  const ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);
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

  if (rowmajor) {
    add_quantize_and_pack_4w_with_group_sums_node(
        graph,
        fp_input,
        int_input_sums,
        packed_input_scale,
        packed_input_zp,
        packed_int_input,
        group_size);
  } else {
    add_quantize_and_pack_4h4w_with_group_sums_node(
        graph,
        input_quant_config,
        fp_input,
        int_input_sums,
        packed_input_scale,
        packed_input_zp,
        packed_int_input,
        group_size);
  }

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
