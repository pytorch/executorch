/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/q4gsw/Q4gswLinearWmma.h>

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
// {{output}, {fp_input}, {packed_weight, packed_weight_scales, packed_bias}}
// resize_args: {group_size, weight_data, bias_data}
WmmaShape q4gsw_runtime_shape(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef packed_weight = args.at(2).refs.at(0);

  const std::vector<int64_t> out_sizes = graph->sizes_of(output);
  WmmaShape s;
  s.op = WmmaOp::kQ4gsw;
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
  return s;
}

// args: {input, weight, weight_scales, group_size, bias, output}
WmmaShape q4gsw_build_time_shape(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  const ValueRef fp_input = args.at(0);
  const ValueRef output = args.at(5);
  const std::vector<int64_t> out_sizes = graph.sizes_of(output);
  WmmaShape s;
  s.op = WmmaOp::kQ4gsw;
  s.N = utils::val_at(-1, out_sizes);
  s.M = utils::val_at(-2, out_sizes);
  s.K = utils::val_at(-1, graph.sizes_of(fp_input));
  s.group_size = graph.extract_scalar<int64_t>(args.at(3));
  for (int64_t d = 0; d < graph.dim_of(output) - 2; d++) {
    s.batch *= utils::val_at(d, out_sizes);
  }
  s.gemv = is_gemv(&graph, fp_input);
  s.has_bias = !graph.val_is_none(args.at(4));
  s.half = graph.dtype_of(output) == vkapi::kHalf;
  s.input = to_wmma_storage(graph.storage_type_of(fp_input));
  s.output = to_wmma_storage(graph.storage_type_of(output));
  s.weight = to_wmma_storage(predicted_q4_weight_storage(graph, args.at(1)));
  s.io_width_packed = graph.packed_dim_of(output) == WHCN::kWidthDim &&
      graph.packed_dim_of(fp_input) == WHCN::kWidthDim;
  return s;
}

vkapi::ShaderInfo pick_q4gsw_linear_wmma_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef packed_weight = args.at(2).refs.at(0);

  const std::optional<WmmaChoice> choice = select_wmma_kernel(
      coopmat_device_key(graph), q4gsw_runtime_shape(graph, args, resize_args));

  std::string kernel_name;
  if (choice.has_value()) {
    kernel_name = choice->kernel_base;
  } else {
    // Shapes no family row covers after a resize (decode, M below
    // kWmmaPartialTileM, buffer IO at unaligned M): the kernels that read
    // this packed weight layout.
    kernel_name =
        is_gemv(graph, fp_input) ? "linear_q4gsw_coop" : "linear_q4gsw_tiled";
  }
  add_storage_type_suffix(kernel_name, graph->storage_type_of(output));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph->dtype_of(output));
  return VK_KERNEL_FROM_STR(kernel_name);
}

} // namespace

bool try_add_q4gsw_linear_wmma_node(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  const ValueRef fp_input = args.at(0);
  const ValueRef weight_data = args.at(1);
  const ValueRef weight_scales_data = args.at(2);
  const ValueRef group_size = args.at(3);
  const ValueRef bias_data = args.at(4);
  const ValueRef output = args.at(5);

  // The family kernels are fp16 only.
  if (graph.dtype_of(fp_input) != vkapi::kHalf) {
    return false;
  }
  // Take the op over only if a row applies to its build-time shape; otherwise
  // it stays entirely on the existing path, including for later resizes.
  const std::optional<WmmaChoice> build_choice = select_wmma_kernel(
      coopmat_device_key(&graph), q4gsw_build_time_shape(graph, args));
  if (!build_choice.has_value()) {
    return false;
  }
  // A storage-image read of the input needs VK_IMAGE_LAYOUT_GENERAL
  // (VUID-vkCmdDispatch-imageLayout-00344), which read-write access selects.
  // The fallback kernels sample the input, which GENERAL also allows.
  const vkapi::MemoryAccessFlags input_access = build_choice->image_a &&
          graph.storage_type_of(fp_input) == utils::kTexture3D
      ? vkapi::kReadWrite
      : vkapi::kRead;

  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size);
  const QuantizationConfig weight_quant_config(4, kPerGroup, {group_size_val});

  const ValueRef packed_weight =
      prepack_quantized_linear_weight(graph, weight_quant_config, weight_data);
  const ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);

  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);
  ValueRef packed_bias = dummy_bias.vref;
  uint32_t apply_bias = 0;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
    apply_bias = 1;
  }

  const int32_t K4_per_group =
      utils::div_up(static_cast<int32_t>(group_size_val), int32_t(4));
  const int32_t num_groups = graph.size_at<int32_t>(-1, fp_input) /
      static_cast<int32_t>(group_size_val);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_q4gsw_linear_wmma_shader,
      wmma_linear_gwg,
      quantized_linear_lwg,
      {{output, vkapi::kWrite},
       {fp_input, input_access},
       {{packed_weight, packed_weight_scales, packed_bias}, vkapi::kRead}},
      {graph.sizes_ubo(output), graph.sizes_ubo(fp_input)},
      {},
      // The spec constants of the linear_q4gsw_{tiled,coop} kernels. The
      // output width N is a spec constant for the coopMatStore address math.
      {apply_bias,
       K4_per_group,
       num_groups,
       graph.size_at<int32_t>(-1, output)},
      {group_size, weight_data, bias_data},
      resize_linear_qw_node));
  return true;
}

} // namespace vkcompute
