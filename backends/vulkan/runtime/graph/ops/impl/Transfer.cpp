/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Transfer.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

/**
 * Adds a transfer copy operation node to the compute graph.
 * This function handles both SELECT and SLICE operations based on the
 * transfer_type parameter.
 */
void add_transfer_copy_node(
    ComputeGraph& graph,
    TransferType transfer_type,
    const ValueRef in,
    const ValueRef dim_ref,
    const ValueRef index_or_start_ref,
    const ValueRef end_ref,
    const ValueRef step_ref,
    const ValueRef out,
    const std::vector<ValueRef>& resize_args,
    const ExecuteNode::ResizeFunction& resize_fn) {
  int64_t ndim = graph.dim_of(in);
  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);

  if (dim < 0) {
    dim += ndim;
  }

  int64_t dim_whcn = nchw_dim_to_whcn_dim(dim, ndim);

  vkapi::ParamsBindList param_buffers;
  if (transfer_type == TransferType::SELECT) {
    param_buffers = {
        graph.get_or_create_int_param_buffer(index_or_start_ref, 0)};
  } else { // TransferType::SLICE
    param_buffers = {
        graph.get_or_create_int_param_buffer(index_or_start_ref, 0),
        graph.get_or_create_int_param_buffer(step_ref, 1)};
  }

  const struct TransferParams {
    const int32_t dim;
  } transfer_params{static_cast<int32_t>(dim_whcn)};

  std::vector<PushConstantDataInfo> push_constants;
  vkapi::SpecVarList spec_vars;

  if (graph.is_buffer_storage(out)) {
    push_constants = {
        graph.sizes_pc_of(in),
        graph.strides_pc_of(out),
        graph.strides_pc_of(in),
        graph.numel_pc_of(out),
        PushConstantDataInfo(&transfer_params, sizeof(transfer_params))};

    spec_vars = {
        graph.packed_dim_of(out),
        graph.packed_dim_of(in),
    };
  } else {
    push_constants = {
        graph.sizes_pc_of(out),
        graph.sizes_pc_of(in),
        PushConstantDataInfo(&transfer_params, sizeof(transfer_params))};

    spec_vars = {
        graph.hashed_layout_of(out),
        graph.hashed_layout_of(in),
    };
  }

  // Determine the shader directly
  std::string kernel_name;
  if (transfer_type == TransferType::SELECT) {
    kernel_name = "select";
  } else { // TransferType::SLICE
    kernel_name = "slice";
  }
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  // Create and add the dispatch node
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Parameter buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      spec_vars,
      // Resize Args
      resize_args,
      // Resizing Logic
      resize_fn));
}

} // namespace vkcompute
