/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

namespace at {
namespace native {
namespace vulkan {

ComputeGraph::ComputeGraph(GraphConfig config)
    : config_{config},
      context_{new api::Context(
          api::runtime()->default_adapter_i(),
          config_.contextConfig)},
      shared_objects_{},
      values_{},
      prepack_nodes_{},
      execute_nodes_{},
      inputs_{},
      outputs_{} {
  context_->set_cmd(/*reusable = */ true);
}

ComputeGraph::~ComputeGraph() {
  values_.clear();

  prepack_nodes_.clear();
  execute_nodes_.clear();

  context_->flush();
}

ValueRef ComputeGraph::add_tensor(
    const std::vector<int64_t>& sizes,
    const api::ScalarType dtype,
    const int64_t shared_object_idx) {
  bool allocate_memory = shared_object_idx < 0;

  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(vTensor(
      context(),
      sizes,
      dtype,
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
      allocate_memory));

  if (!allocate_memory) {
    get_shared_object(shared_object_idx).add_user(this, idx);
  }
  return idx;
}

ValueRef ComputeGraph::add_tensorref(
    const std::vector<int64_t>& sizes,
    const api::ScalarType dtype,
    const void* const data) {
  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(TensorRef(sizes, dtype, data));
  return idx;
}

ValueRef ComputeGraph::add_staging(
    const api::ScalarType dtype,
    const size_t numel) {
  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(api::StorageBuffer(context(), dtype, numel));
  return idx;
}

ValueRef ComputeGraph::set_input_tensor(
    const ValueRef idx,
    const bool use_staging) {
  if (use_staging) {
    vTensor& tensor = get_val(idx).toTensor();
    ValueRef staging_idx = add_staging(tensor.dtype(), tensor.gpu_numel());
    execute_nodes_.emplace_back(new StagingNode(staging_idx, idx));
    inputs_.push_back(staging_idx);
    return staging_idx;
  }
  inputs_.push_back(idx);
  return idx;
}

ValueRef ComputeGraph::set_output_tensor(
    const ValueRef idx,
    const bool use_staging) {
  if (use_staging) {
    vTensor& tensor = get_val(idx).toTensor();
    ValueRef staging_idx = add_staging(tensor.dtype(), tensor.gpu_numel());
    execute_nodes_.emplace_back(new StagingNode(idx, staging_idx));
    outputs_.push_back(staging_idx);
    return staging_idx;
  }
  outputs_.push_back(idx);
  return idx;
}

SharedObject& ComputeGraph::get_shared_object(const int64_t idx) {
  if (idx >= shared_objects_.size()) {
    shared_objects_.resize(static_cast<size_t>(idx + 1));
  }
  return shared_objects_.at(idx);
}

void ComputeGraph::copy_into_staging(
    const ValueRef idx,
    const void* data,
    const size_t numel) {
  Value& in_val = get_val(idx);
  api::StorageBuffer& staging = in_val.toStaging();
  size_t nbytes = numel * api::element_size(staging.dtype());
  copy_ptr_to_staging(data, staging, nbytes);
}

void ComputeGraph::copy_from_staging(
    const ValueRef idx,
    void* data,
    const size_t numel) {
  Value& out_val = get_val(idx);
  api::StorageBuffer& staging = out_val.toStaging();
  size_t nbytes = numel * api::element_size(staging.dtype());
  copy_staging_to_ptr(staging, data, nbytes);
}

void ComputeGraph::encode_prepack() {
  for (std::unique_ptr<PrepackNode>& node : prepack_nodes_) {
    node->encode(this);
  }
}

void ComputeGraph::prepack() const {
  // Submit and execute the command buffer
  api::VulkanFence fence = context_->fences().get_fence();
  context_->submit_cmd_to_gpu(fence.get_submit_handle(), /*final_use = */ true);
  fence.wait();

  context_->flush();
}

void ComputeGraph::encode_execute() {
  context_->flush();
  context_->set_cmd(/*reusable = */ true);

  for (SharedObject& shared_object : shared_objects_) {
    shared_object.allocate(this);
    shared_object.bind_users(this);
  }

  for (std::unique_ptr<ExecuteNode>& node : execute_nodes_) {
    node->encode(this);
  }
}

void ComputeGraph::execute() const {
  api::VulkanFence fence = context_->fences().get_fence();
  context_->submit_cmd_to_gpu(fence.get_submit_handle());
  fence.wait();
}

} // namespace vulkan
} // namespace native
} // namespace at
