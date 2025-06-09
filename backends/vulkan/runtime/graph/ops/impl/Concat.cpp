/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Clone.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_concat_node(
    ComputeGraph& graph,
    const ValueRef tensors_ref,
    const ValueRef dim_ref,
    const ValueRef out) {
  std::vector<ValueRef> in_value_refs;

  {
    const ValueListPtr tensors = graph.get_value_list(tensors_ref);

    VK_CHECK_COND(
        tensors->size() <= 3,
        "Currently only concatenation of <= 3 tensors is supported");

    for (const ValueRef in : *tensors) {
      in_value_refs.push_back(in);
    }
  }

  const int64_t dim = graph.extract_scalar<int64_t>(dim_ref);

  const int64_t ndim = graph.dim_of(in_value_refs.at(0));
  int64_t normalized_dim = dim;
  if (normalized_dim < 0) {
    normalized_dim += ndim;
  }

  const int64_t dim_whcn = nchw_dim_to_whcn_dim(normalized_dim, ndim);
  const ValueRef dim_whcn_ref = graph.get_or_add_value_for_int(dim_whcn);

  vkapi::ParamsBindList param_buffers = {
      graph.get_or_create_int_param_buffer(dim_whcn_ref, 0)};

  std::vector<PushConstantDataInfo> push_constants;
  vkapi::SpecVarList spec_vars;

  if (graph.is_buffer_storage(out)) {
    param_buffers.append(graph.sizes_ubo(out));
    param_buffers.append(graph.strides_ubo(out));

    for (const ValueRef in_ref : in_value_refs) {
      param_buffers.append(graph.sizes_ubo(in_ref));
      param_buffers.append(graph.strides_ubo(in_ref));
    }

    param_buffers.append(graph.numel_ubo(out));

    spec_vars = {graph.packed_dim_of(out)};
  } else {
    push_constants = {graph.sizes_pc_of(out)};

    spec_vars = {graph.hashed_layout_of(out)};

    for (const ValueRef in_ref : in_value_refs) {
      push_constants.push_back(graph.sizes_pc_of(in_ref));
      spec_vars.append(graph.hashed_layout_of(in_ref));
    }
  }

  std::string kernel_name = "concat";
  if (in_value_refs.size() == 1) {
    kernel_name += "_1";
  } else if (in_value_refs.size() == 2) {
    kernel_name += "_2";
  } else if (in_value_refs.size() == 3) {
    kernel_name += "_3";
  }
  if (graph.is_buffer_storage(out)) {
    kernel_name += "_buffer";
  } else {
    kernel_name += "_texture3d";
  }

  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in_value_refs, vkapi::kRead}},
      // Parameter buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      spec_vars,
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

void cat_tensor(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // Extract arguments
  const ValueRef tensors_ref = args.at(0);
  const ValueRef dim_ref = args.at(1);
  const ValueRef out = args.at(2);

  // Add concat node
  add_concat_node(graph, tensors_ref, dim_ref, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.cat.default, cat_tensor);
}

} // namespace vkcompute
