/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

using namespace utils;

utils::uvec3 fused_ce_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef loss_partial = args.at(0).refs.at(1);
  return {
      1u, utils::safe_downcast<uint32_t>(graph->numel_of(loss_partial)), 1u};
}

utils::uvec3 fused_ce_local_wg_size(
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
  return {64u, 1u, 1u};
}

utils::uvec3 fused_ce_sum_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)args;
  (void)resize_args;
  return {1u, 1u, 1u};
}

utils::uvec3 fused_ce_sum_local_wg_size(
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
  return {64u, 1u, 1u};
}

void resize_fused_ce_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef dlogits = args.at(0).refs.at(0);
  const ValueRef loss_partial = args.at(0).refs.at(1);
  const ValueRef logits = args.at(1).refs.at(0);

  const std::vector<int64_t> logits_sizes = graph->sizes_of(logits);
  graph->virtual_resize(dlogits, logits_sizes);
  graph->virtual_resize(loss_partial, {logits_sizes.at(0)});
}

void resize_fused_ce_sum_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef loss = args.at(0).refs.at(0);
  // Rank-0 scalar, matching fused_ce_meta's logits.new_empty([]).
  graph->virtual_resize(loss, {});
}

void fused_ce(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef logits = args[arg_idx++];
  const ValueRef labels = args[arg_idx++];
  const ValueRef n_valid_ref = args[arg_idx++];
  const ValueListPtr out_tuple = graph.get_value_list(args[arg_idx++]);
  const ValueRef loss = out_tuple->at(0);
  const ValueRef dlogits = out_tuple->at(1);

  VK_CHECK_COND(
      graph.is_buffer_storage(logits) && graph.is_buffer_storage(dlogits),
      "fused_ce: logits and dlogits must use buffer storage");
  VK_CHECK_COND(
      graph.dim_of(logits) == 2, "fused_ce: logits must be 2D [n_rows, vocab]");
  VK_CHECK_COND(
      graph.sizes_of(dlogits) == graph.sizes_of(logits),
      "fused_ce: dlogits must match logits shape");
  VK_CHECK_COND(
      graph.dtype_of(labels) == vkapi::kInt, "fused_ce: labels must be int32");
  VK_CHECK_COND(graph.dim_of(labels) == 1, "fused_ce: labels must be 1D [N]");
  VK_CHECK_COND(
      graph.size_at<int64_t>(0, labels) == graph.size_at<int64_t>(0, logits),
      "fused_ce: labels length must equal number of rows");
  VK_CHECK_COND(graph.numel_of(loss) == 1, "fused_ce: loss must be a scalar");

  const int32_t n_rows = graph.size_at<int32_t>(0, logits);
  const int32_t vocab = graph.size_at<int32_t>(1, logits);
  const float n_valid = graph.extract_scalar<float>(n_valid_ref);

  // One workgroup per row; the Vulkan spec guarantees maxComputeWorkGroupCount
  // >= 65535 per dimension.
  VK_CHECK_COND(
      n_rows <= 65535, "fused_ce: n_rows exceeds max workgroup count");

  // Per-row loss contributions, reduced to the scalar loss by the sum node.
  TmpTensor loss_partial(
      &graph, {n_rows}, graph.dtype_of(logits), utils::kBuffer);

  std::string kernel_name = "fused_ce";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(dlogits));
  add_dtype_suffix(kernel_name, graph.dtype_of(dlogits));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      fused_ce_global_wg_size,
      fused_ce_local_wg_size,
      // Inputs and Outputs
      {{{dlogits, loss_partial}, vkapi::kWrite},
       {{logits, labels}, vkapi::kRead}},
      // Shader params buffers
      {graph.create_params_buffer(vocab),
       graph.create_params_buffer(n_rows),
       graph.create_params_buffer(n_valid)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_fused_ce_node));

  std::string sum_kernel_name = "fused_ce_sum";
  sum_kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(sum_kernel_name, graph.storage_type_of(loss));
  add_dtype_suffix(sum_kernel_name, graph.dtype_of(loss));

  // Separate dispatch node so the loss_partial write (above) is ordered before
  // this read: the graph inserts a per-tensor pipeline barrier between them.
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(sum_kernel_name),
      fused_ce_sum_global_wg_size,
      fused_ce_sum_local_wg_size,
      // Inputs and Outputs
      {{{loss}, vkapi::kWrite}, {{loss_partial}, vkapi::kRead}},
      // Shader params buffers
      {graph.create_params_buffer(n_rows)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_fused_ce_sum_node));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.fused_ce.default, fused_ce);
}

} // namespace vkcompute
