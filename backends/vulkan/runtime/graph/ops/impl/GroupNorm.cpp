/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

utils::uvec3 group_norm_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;

  return {1, 64, 1};
}

void resize_group_norm_texture_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  // Extract tensor references from args
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const ValueRef mean = args.at(1).refs.at(3);
  const ValueRef rstd = args.at(1).refs.at(4);

  // Extract group from resize args
  const int64_t group_val = graph->extract_scalar<int64_t>(resize_args.at(0));

  // Get input tensor sizes using ComputeGraph APIs
  const std::vector<int64_t> in_sizes = graph->sizes_of(in);

  // Output tensor should have the same size as input
  graph->virtual_resize(out, in_sizes);

  // Mean and rstd tensors should have size {num_batches, num_groups}
  const int64_t N = in_sizes.at(0); // batch dimension
  const std::vector<int64_t> mean_rstd_sizes = {N, group_val};

  // Resize mean and rstd tensors
  graph->virtual_resize(mean, mean_rstd_sizes);
  graph->virtual_resize(rstd, mean_rstd_sizes);
}

void add_native_group_norm_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias_data,
    const ValueRef N,
    const ValueRef C,
    const ValueRef HxW,
    const ValueRef group,
    const ValueRef eps,
    const ValueRef out,
    const ValueRef mean,
    const ValueRef rstd) {
  (void)N;
  (void)C;
  (void)HxW;

  const ValueRef arg_weight = prepack_standard(
      graph,
      weight_data,
      graph.storage_type_of(in),
      utils::kWidthPacked,
      false);
  const ValueRef arg_bias = prepack_standard(
      graph, bias_data, graph.storage_type_of(in), utils::kWidthPacked, false);

  const int64_t group_val = graph.extract_scalar<int64_t>(group);
  const float epsilon = graph.extract_scalar<float>(eps);

  const std::vector<int64_t> in_sizes = graph.sizes_of(in);

  std::string kernel_name("group_norm_reduce_texture");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  const struct {
    int32_t group;
    float epsilon;
  } params_uniform = {static_cast<int32_t>(group_val), epsilon};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      group_norm_local_wg_size,
      // Inputs and Outputs
      {{{mean, rstd}, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      {
          graph.strides_ubo(mean),
          graph.numel_ubo(mean),
          graph.logical_limits_ubo(in),
          graph.sizes_ubo(in),
      },
      // Push Constants
      {
          PushConstantDataInfo(&params_uniform, sizeof(params_uniform)),
      },
      // Specialization Constants
      {
          graph.hashed_layout_of(mean),
      },
      // Resize Args
      {group},
      // Resizing Logic
      nullptr));

  // Compute element-wise normalization, now that mean and rstd have been
  // computed.
  std::string norm_kernel_name("group_norm_texture");
  norm_kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(norm_kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(norm_kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite},
       {{in, arg_weight, arg_bias, mean, rstd}, vkapi::kRead}},
      // Shader params buffers
      {
          graph.logical_limits_ubo(out),
          graph.sizes_ubo(out),
          graph.logical_limits_ubo(arg_weight),
          graph.strides_ubo(mean),
      },
      // Push Constants
      {
          PushConstantDataInfo(&params_uniform, sizeof(params_uniform)),
      },
      // Specialization Constants
      {
          graph.hashed_layout_of(in),
      },
      // Resize Args
      {group},
      // Resizing Logic
      resize_group_norm_texture_node));
}

void native_group_norm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // Assign each element of the args vector to const ValueRef variables
  const ValueRef in = args.at(0);
  const ValueRef weight_data = args.at(1);
  const ValueRef bias_data = args.at(2);
  const ValueRef N = args.at(3);
  const ValueRef C = args.at(4);
  const ValueRef HxW = args.at(5);
  const ValueRef group = args.at(6);
  const ValueRef eps = args.at(7);
  const ValueRef out_tuple_ref = args.at(8);

  ValueRef out = kDummyValueRef;
  ValueRef mean = kDummyValueRef;
  ValueRef rstd = kDummyValueRef;

  {
    const ValueListPtr out_tuple = graph.get_value_list(out_tuple_ref);
    out = out_tuple->at(0);
    mean = out_tuple->at(1);
    rstd = out_tuple->at(2);
  }

  VK_CHECK_COND(graph.val_is_tref(weight_data));
  VK_CHECK_COND(graph.val_is_tref(bias_data));

  // Check expected storage types and memory layouts for tensor variables
  VK_CHECK_COND(graph.is_standard_channels_packed_texture_tensor(in));
  VK_CHECK_COND(graph.is_standard_channels_packed_texture_tensor(out));

  VK_CHECK_COND(graph.is_contiguous_buffer_tensor(mean));
  VK_CHECK_COND(graph.is_contiguous_buffer_tensor(rstd));

  return add_native_group_norm_node(
      graph,
      in,
      weight_data,
      bias_data,
      N,
      C,
      HxW,
      group,
      eps,
      out,
      mean,
      rstd);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.native_group_norm.default, native_group_norm);
}

} // namespace vkcompute
