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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_linear_qga4w_args(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef group_size,
    const ValueRef scales_and_zeros,
    const ValueRef out) {
  VK_CHECK_COND(graph.val_is_tensor(mat1));
  VK_CHECK_COND(graph.val_is_tref(mat2_data));
  VK_CHECK_COND(graph.val_is_tref(scales_and_zeros));

  VK_CHECK_COND(graph.dim_of(mat1) <= 3);
  VK_CHECK_COND(graph.dim_of(mat2_data) == 2);
  VK_CHECK_COND(graph.dim_of(scales_and_zeros) == 3);

  VK_CHECK_COND(graph.size_at<int>(-3, mat1) == 1);
  const int K = graph.size_at<int>(-1, mat1);
  VK_CHECK_COND(graph.size_at<int>(-1, mat2_data) * 2 == K);

  const int group_size_val = graph.extract_scalar<int>(group_size);
  VK_CHECK_COND(K % group_size_val == 0);
  // Due to the way weight packing works, group size needs to be a multiple of 8
  VK_CHECK_COND(group_size_val % 8 == 0);

  VK_CHECK_COND(graph.has_standard_axis_map(mat1));
  VK_CHECK_COND(graph.has_standard_axis_map(out));
}

void resize_linear_qga4w_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  ValueRef out = args.at(0).refs.at(0);
  ValueRef mat1 = args.at(1).refs.at(0);
  ValueRef mat2_data = extra_args.at(0);

  std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1);
  std::vector<int64_t> mat2_sizes = graph->sizes_of(mat2_data);

  const int64_t out_cols = utils::val_at(-2, mat1_sizes);
  const int64_t out_rows = utils::val_at(-2, mat2_sizes);

  std::vector<int64_t> new_out_sizes(3);
  if (mat1_sizes.size() == 2) {
    new_out_sizes.resize(2);
    new_out_sizes.at(0) = out_cols;
    new_out_sizes.at(1) = out_rows;
  } else {
    new_out_sizes.at(0) = mat1_sizes.at(0);
    new_out_sizes.at(1) = out_cols;
    new_out_sizes.at(2) = out_rows;
  }

  graph->virtual_resize(out, new_out_sizes);
}

/**
 * Determines if the cooperative algorithm should be used based on input tensor
 * dimensions. Apply the coop algorithm for gemv cases, i.e. mat1 is avector as
 * as opposed to a matrix.
 */
bool should_use_coop_algorithm(ComputeGraph* graph, const ValueRef& mat1) {
  return graph->size_at<uint32_t>(-2, mat1) == 1;
}

vkapi::ShaderInfo pick_linear_qga4w_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);
  const ValueRef mat2 = args.at(1).refs.at(1);

  const bool use_coop_algorithm = should_use_coop_algorithm(graph, mat1);

  std::string kernel_name = "linear_qga4w";
  if (use_coop_algorithm) {
    kernel_name += "_coop";
  } else {
    kernel_name += "_tiled";
  }
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(mat1));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(mat2));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));

  return VK_KERNEL_FROM_STR(kernel_name);
}

utils::uvec3 linear_qga4w_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  const bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  if (!use_coop_algorithm) {
    // Constructing the global workgroup size for the tiled algorithm
    utils::uvec3 global_wg_size = graph->logical_limits_of(out);
    // Each shader thread computes a 4 high x 8 wide tile of the output matrix,
    // which is equivalent to 4 x 2 texels. Since the output tensor must be
    // width packed, div-up the "texel-width" of the output by 2 and the height
    // of the output tensor by 4 to obtain the number of tiles that need to be
    // computed.
    global_wg_size[0] = utils::div_up(global_wg_size[0], uint32_t(2));
    global_wg_size[1] = utils::div_up(global_wg_size[1], uint32_t(4));
    return global_wg_size;
  }

  uint32_t output_channels = graph->size_at<uint32_t>(-1, out);
  uint32_t batch_size = graph->size_at<uint32_t>(-2, out);

  // Constructing the global workgroup size of the co-operative algorithm. The
  // local work group size is 64, and each local work group co-operates to
  // compute 8 output channels of the output. Therefore, a total of
  // (output_channels / 8 x 64) threads should be launched, assuming a batch
  // size of 1.
  return {64, utils::div_up(output_channels, 8u), batch_size};
}

utils::uvec3 linear_qga4w_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)args;
  (void)resize_args;
  const bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  if (use_coop_algorithm) {
    return {64, 1, 1};
  } else {
    return pick_hw_square_wg_size(
        graph, shader, global_workgroup_size, args, resize_args);
  }
}

void add_linear_qga4w_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef group_size,
    const ValueRef scales_and_zeros_data,
    const ValueRef out) {
  check_linear_qga4w_args(
      graph, mat1, mat2_data, group_size, scales_and_zeros_data, out);

  const uint32_t group_size_val = graph.extract_scalar<uint32_t>(group_size);

  ValueRef mat2 =
      prepack_int4_linear_weight_transposed_block_4x8(graph, mat2_data);

  ValueRef scales_and_zeros = prepack_standard(
      graph, scales_and_zeros_data, utils::kBuffer, utils::kWidthPacked);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_qga4w_shader,
      linear_qga4w_global_wg_size,
      linear_qga4w_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1, mat2, scales_and_zeros}, vkapi::kRead}},
      // Shader params buffers
      {},
      // Push Constants
      {graph.sizes_pc_of(out),
       graph.sizes_pc_of(mat1),
       graph.sizes_pc_of(mat2)},
      // Specialization Constants
      {SV(group_size_val)},
      // Resize Args
      {mat2_data},
      // Resizing Logic
      resize_linear_qga4w_node));
}

void linear_weight_int4(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  return add_linear_qga4w_node(
      graph,
      args[0], // mat1
      args[1], // mat2
      args[2], // group_size
      args[3], // scales_and_zeros
      // There is an unused variable inner_k_tiles which is used to call
      // _convert_weight_to_int4pack in the AOT custom op, which is why the 4th
      // argument is skipped.
      args[5] // out
  );
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_weight_int4.default, linear_weight_int4);
}

} // namespace vkcompute
