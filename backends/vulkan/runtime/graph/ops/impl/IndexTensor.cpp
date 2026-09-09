/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_index_tensor_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);
  const ValueRef index = args.at(1).refs.at(1);

  int64_t index_dim = -1;
  {
    const ValueListPtr indices = graph->get_value_list(resize_args.at(0));
    for (size_t dim = 0; dim < indices->size(); ++dim) {
      if (!graph->val_is_none(indices->at(dim))) {
        index_dim = utils::safe_downcast<int64_t>(dim);
        break;
      }
    }
  }
  VK_CHECK_COND(index_dim >= 0, "index.Tensor: an index tensor is required");

  const std::vector<int64_t> self_sizes = graph->sizes_of(self);
  const std::vector<int64_t> index_sizes = graph->sizes_of(index);
  std::vector<int64_t> out_sizes;
  out_sizes.reserve(self_sizes.size() + index_sizes.size() - 1);
  out_sizes.insert(
      out_sizes.end(), self_sizes.begin(), self_sizes.begin() + index_dim);
  out_sizes.insert(out_sizes.end(), index_sizes.begin(), index_sizes.end());
  out_sizes.insert(
      out_sizes.end(), self_sizes.begin() + index_dim + 1, self_sizes.end());

  graph->virtual_resize(out, out_sizes);
}

void add_index_tensor_node(
    ComputeGraph& graph,
    const ValueRef self,
    const ValueRef index,
    const int64_t index_dim,
    const ValueRef indices_list_ref,
    const ValueRef out) {
  std::string kernel_name = "index_tensor";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  if (graph.is_buffer_storage(out)) {
    add_storage_type_suffix(kernel_name, graph.storage_type_of(index));
  }
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_ubos = {
      graph.meta_ubo(out), graph.meta_ubo(self), graph.meta_ubo(index)};
  const utils::ivec2 index_params = {
      utils::safe_downcast<int32_t>(graph.dim_of(self) - 1 - index_dim),
      utils::safe_downcast<int32_t>(graph.dim_of(index))};
  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&index_params, sizeof(index_params))};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{self, index}, vkapi::kRead}},
      // Shader params buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(self),
       graph.hashed_layout_of(index)},
      // Resize Args
      {indices_list_ref},
      // Resizing Logic
      resize_index_tensor_node));
}

void index_tensor(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef self = args[0];
  ValueRef indices_list_ref = args[1];
  ValueRef out = args[2];

  ValueRef index = -1;
  int64_t index_dim = -1;
  {
    const ValueListPtr indices_list = graph.get_value_list(indices_list_ref);
    for (size_t dim = 0; dim < indices_list->size(); ++dim) {
      const ValueRef candidate = indices_list->at(dim);
      if (graph.val_is_none(candidate)) {
        continue;
      }
      VK_CHECK_COND(
          index_dim < 0, "index.Tensor: only one index tensor is supported");
      index = candidate;
      index_dim = utils::safe_downcast<int64_t>(dim);
    }
  }
  VK_CHECK_COND(index_dim >= 0, "index.Tensor: an index tensor is required");
  VK_CHECK_COND(
      index_dim < graph.dim_of(self),
      "index.Tensor: index dimension is invalid");

  add_index_tensor_node(graph, self, index, index_dim, indices_list_ref, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.index.Tensor, index_tensor);
}

} // namespace vkcompute
