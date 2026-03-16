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

  struct TransferParams {
    int32_t dim;
    int32_t index_or_start_ref;
    int32_t step_ref;
  } transfer_params{static_cast<int32_t>(dim_whcn), 0, 0};

  const bool param_is_scalar = graph.is_scalar_or_none(index_or_start_ref) &&
      (transfer_type == TransferType::SELECT ||
       graph.is_scalar_or_none(step_ref));

  vkapi::ParamsBindList param_ubos = {graph.meta_ubo(out), graph.meta_ubo(in)};

  if (!param_is_scalar) {
    if (transfer_type == TransferType::SELECT) {
      param_ubos.append(
          graph.get_or_create_int_param_buffer(index_or_start_ref, 0));
    } else { // TransferType::SLICE
      param_ubos.append(
          graph.get_or_create_int_param_buffer(index_or_start_ref, 0));
      param_ubos.append(graph.get_or_create_int_param_buffer(step_ref, 1));
    }
  } else {
    transfer_params.index_or_start_ref =
        graph.extract_scalar_or<int32_t>(index_or_start_ref, 0);
    if (transfer_type != TransferType::SELECT) {
      transfer_params.step_ref = graph.extract_scalar_or<int32_t>(step_ref, 1);
    }
  }

  std::vector<PushConstantDataInfo> push_constants;
  if (param_is_scalar) {
    push_constants.emplace_back(&transfer_params, sizeof(transfer_params));
  } else {
    push_constants.emplace_back(
        &transfer_params.dim, sizeof(transfer_params.dim));
  }

  // Determine the shader directly
  std::string kernel_name;
  if (transfer_type == TransferType::SELECT) {
    kernel_name = "select";
  } else { // TransferType::SLICE
    kernel_name = "slice";
  }
  if (!param_is_scalar) {
    kernel_name += "_ubo";
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
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize Args
      resize_args,
      // Resizing Logic
      resize_fn));
}

} // namespace vkcompute
