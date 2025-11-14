/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

namespace vkcompute {

void resize_choose_qparams_per_row(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  ValueRef input_scales = args.at(0).refs.at(0);
  ValueRef input_zeros = args.at(0).refs.at(1);
  ValueRef input = args.at(1).refs.at(0);

  std::vector<int64_t> new_sizes = graph->sizes_of(input_scales);
  const size_t ndim = new_sizes.size();

  const int64_t input_height = graph->size_at<int64_t>(-2, input);
  new_sizes.at(ndim - 1) = input_height;

  graph->virtual_resize(input_scales, new_sizes);
  graph->virtual_resize(input_zeros, new_sizes);
}

vkapi::ShaderInfo pick_choose_qparams_per_row_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;

  const ValueRef input = args.at(1).refs.at(0);

  std::string kernel_name = "choose_qparams_per_row";
  add_storage_type_suffix(kernel_name, graph->storage_type_of(input));
  add_dtype_suffix(kernel_name, graph->dtype_of(input));

  return VK_KERNEL_FROM_STR(kernel_name);
}

utils::uvec3 pick_choose_qparams_per_row_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef input = args.at(1).refs.at(0);
  const uint32_t height = graph->size_at<uint32_t>(-2, input);
  return {1u, utils::div_up_4(height), 1u};
}

utils::uvec3 pick_choose_qparams_per_row_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;

  uint32_t outputs_per_wg = 1u;
  uint32_t workers_per_output = 64u;

  return {workers_per_output, outputs_per_wg, 1u};
}

void add_choose_qparams_per_row_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& quant_min,
    const ValueRef& quant_max,
    const ValueRef& input_scales,
    const ValueRef& input_zps) {
  int32_t quant_min_val = -128;
  int32_t quant_max_val = 127;

  // Int8 range by default
  if (graph.val_is_none(quant_min)) {
    quant_min_val = -128;
  } else {
    quant_min_val = graph.extract_scalar<int32_t>(quant_min);
  }

  // Int8 range by default
  if (graph.val_is_none(quant_min)) {
    quant_max_val = 127;
  } else {
    quant_max_val = graph.extract_scalar<int32_t>(quant_max);
  }

  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(input),
  };
  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&quant_min_val, sizeof(int32_t)),
      PushConstantDataInfo(&quant_max_val, sizeof(int32_t)),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_choose_qparams_per_row_shader,
      pick_choose_qparams_per_row_global_wg_size,
      pick_choose_qparams_per_row_local_wg_size,
      // Inputs and Outputs
      {{{input_scales, input_zps}, vkapi::kWrite}, {input, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_choose_qparams_per_row));
}

bool can_use_choose_qparams_per_row(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef block_size,
    const ValueRef input_zero_point) {
  if (!graph.is_vectorizable_contiguous_2d_matrix(input)) {
    return false;
  }

  std::vector<int64_t> input_sizes = graph.sizes_of(input);
  const IntListPtr block_size_vals = graph.get_int_list(block_size);
  const size_t ndim = block_size_vals->size();

  // Check for per y - dim quantization
  if (utils::val_at(-1, input_sizes) != utils::val_at(-1, *block_size_vals)) {
    return false;
  }

  for (int d = 0; d < ndim - 1; ++d) {
    if (block_size_vals->at(d) != 1) {
      return false;
    }
  }
  return true;
}

void choose_qparams_affine_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef mapping_type = args[arg_idx++];
  (void)mapping_type;
  const ValueRef block_size = args[arg_idx++];
  const ValueRef target_dtype = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef eps = args[arg_idx++];
  (void)eps;
  const ValueRef scale_dtype = args[arg_idx++];
  const ValueRef zero_point_dtype = args[arg_idx++];
  const ValueRef out_tuple_ref = args[arg_idx++];

  // Suppress unused variable warnings
  (void)target_dtype;
  (void)scale_dtype;
  (void)zero_point_dtype;

  ValueRef scale_out = kDummyValueRef;
  ValueRef zero_point_out = kDummyValueRef;

  {
    const ValueListPtr out_tuple = graph.get_value_list(out_tuple_ref);
    scale_out = out_tuple->at(0);
    zero_point_out = out_tuple->at(1);
  }

  // Use fast path if certain conditions are met
  if (can_use_choose_qparams_per_row(
          graph, input, block_size, zero_point_out)) {
    return add_choose_qparams_per_row_node(
        graph, input, quant_min, quant_max, scale_out, zero_point_out);
  }

  VK_THROW("Unsupported input case for choose_qparams_affine");
}

void choose_qparams_per_row(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef input_scales = args[arg_idx++];
  const ValueRef input_zps = args[arg_idx++];

  add_choose_qparams_per_row_node(
      graph, input, quant_min, quant_max, input_scales, input_zps);
}

REGISTER_OPERATORS {
  // Register the per-channel quantization operator
  VK_REGISTER_OP(etvk.choose_qparams_per_row.default, choose_qparams_per_row);

  // TorchAO affine choose_qparams operators
  VK_REGISTER_OP(
      torchao.choose_qparams_affine.default, choose_qparams_affine_impl);
}

} // namespace vkcompute
