/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY
// facebook-security-vulnerable-integer-sign-conversion

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

namespace vkcompute {

//
// VTensorPtr
//

#define VALUE_PTR_CLASS_IMPL(classname, ctype, type_name)                 \
  classname::classname(ComputeGraph* const graph, const ValueRef idx)     \
      : graph_(graph), ptr_(&(graph_->values_.at(idx).to##type_name())) { \
    graph_->values_in_use_++;                                             \
  }                                                                       \
  ctype* classname::operator->() const {                                  \
    return ptr_;                                                          \
  }                                                                       \
  ctype& classname::operator*() const {                                   \
    return *ptr_;                                                         \
  }                                                                       \
  classname::~classname() {                                               \
    graph_->values_in_use_--;                                             \
  }

VALUE_PTR_CLASS_IMPL(vTensorPtr, vTensor, Tensor)
VALUE_PTR_CLASS_IMPL(TensorRefPtr, TensorRef, TensorRef)
VALUE_PTR_CLASS_IMPL(StagingPtr, api::StorageBuffer, Staging)
VALUE_PTR_CLASS_IMPL(IntListPtr, std::vector<int64_t>, IntList)
VALUE_PTR_CLASS_IMPL(DoubleListPtr, std::vector<double>, DoubleList)
VALUE_PTR_CLASS_IMPL(BoolListPtr, std::vector<bool>, BoolList)
VALUE_PTR_CLASS_IMPL(ValueListPtr, std::vector<ValueRef>, ValueList)

#undef VALUE_PTR_CLASS_IMPL

//
// ComputeGraph
//

ComputeGraph::ComputeGraph(GraphConfig config)
    : config_{config},
      prepack_descriptor_counts_{},
      execute_descriptor_counts_{},
      context_{new api::Context(
          api::runtime()->default_adapter_i(),
          config_.context_config)},
      shared_objects_{},
      values_{},
      param_ubos_{},
      prepack_nodes_{},
      execute_nodes_{},
      inputs_{},
      outputs_{} {
  // Ensure that descriptor counts are initialized to 0
  prepack_descriptor_counts_.descriptor_pool_max_sets = 0;
  prepack_descriptor_counts_.descriptor_uniform_buffer_count = 0;
  prepack_descriptor_counts_.descriptor_storage_buffer_count = 0;
  prepack_descriptor_counts_.descriptor_combined_sampler_count = 0;
  prepack_descriptor_counts_.descriptor_storage_image_count = 0;

  execute_descriptor_counts_.descriptor_pool_max_sets = 0;
  execute_descriptor_counts_.descriptor_uniform_buffer_count = 0;
  execute_descriptor_counts_.descriptor_storage_buffer_count = 0;
  execute_descriptor_counts_.descriptor_combined_sampler_count = 0;
  execute_descriptor_counts_.descriptor_storage_image_count = 0;

  context_->set_cmd(/*reusable = */ true);
}

ComputeGraph::~ComputeGraph() {
  values_.clear();

  prepack_nodes_.clear();
  execute_nodes_.clear();

  context_->flush();
}

api::StorageType ComputeGraph::suggested_storage_type() {
  if (config_.enable_storage_type_override) {
    return config_.storage_type_override;
  }
  return api::kTexture3D;
}

api::GPUMemoryLayout ComputeGraph::suggested_memory_layout(
    const std::vector<int64_t>& sizes) {
  if (config_.enable_memory_layout_override) {
    return config_.memory_layout_override;
  }
  if (sizes.size() < 3) {
    return api::kWidthPacked;
  }
  // For 3 dimensional tensors that only have a channels dimension of 1, still
  // prefer width packed.
  if (api::utils::val_at(-3, sizes) == 1) {
    return api::kWidthPacked;
  }
  return api::kChannelsPacked;
}

void ComputeGraph::check_no_active_value_ptrs() {
  VK_CHECK_COND(
      values_in_use_ == 0,
      "Make sure that there are no pointers stored from the return values of "
      "`ComputeGraph::get_*()` functions in scope before adding Values to the "
      "graph. Modifying the graph's values may cause existing pointers to be "
      "invalidated.");
}

std::vector<int64_t> ComputeGraph::sizes_of(const ValueRef idx) const {
  const Value& val = values_.at(idx);
  if (val.isTensor()) {
    return val.toConstTensor().sizes();
  } else if (val.isTensorRef()) {
    return val.toConstTensorRef().sizes;
  }
  VK_THROW("Could not get sizes of value with type ", val.type());
}

api::ScalarType ComputeGraph::dtype_of(const ValueRef idx) const {
  const Value& val = values_.at(idx);
  if (val.isTensor()) {
    return val.toConstTensor().dtype();
  } else if (val.isTensorRef()) {
    return val.toConstTensorRef().dtype;
  }
  VK_THROW("Could not get dtype of value with type ", val.type());
}

ValueRef ComputeGraph::add_tensor(
    const std::vector<int64_t>& sizes,
    const api::ScalarType dtype,
    const api::StorageType storage_type,
    const api::GPUMemoryLayout memory_layout,
    const int64_t shared_object_idx) {
  bool allocate_memory = shared_object_idx < 0;

  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(vTensor(
      context(), sizes, dtype, storage_type, memory_layout, allocate_memory));

  if (!allocate_memory) {
    get_shared_object(shared_object_idx).add_user(this, idx);
  }
  return idx;
}

ValueRef ComputeGraph::add_tensor(
    const std::vector<int64_t>& sizes,
    const api::ScalarType dtype,
    const api::StorageType storage_type,
    const int64_t shared_object_idx) {
  return add_tensor(
      sizes,
      dtype,
      storage_type,
      suggested_memory_layout(sizes),
      shared_object_idx);
}

ValueRef ComputeGraph::add_tensor(
    const std::vector<int64_t>& sizes,
    const api::ScalarType dtype,
    const api::GPUMemoryLayout memory_layout,
    const int64_t shared_object_idx) {
  return add_tensor(
      sizes, dtype, suggested_storage_type(), memory_layout, shared_object_idx);
}

ValueRef ComputeGraph::add_tensor_like(
    const ValueRef idx,
    const api::StorageType storage_type,
    const api::GPUMemoryLayout memory_layout) {
  return add_tensor(sizes_of(idx), dtype_of(idx), storage_type, memory_layout);
}

ValueRef ComputeGraph::add_tensor_like(
    const ValueRef idx,
    const api::GPUMemoryLayout memory_layout) {
  return add_tensor(sizes_of(idx), dtype_of(idx), memory_layout);
}

ValueRef ComputeGraph::add_tensor(
    const std::vector<int64_t>& sizes,
    const api::ScalarType dtype,
    const int64_t shared_object_idx) {
  return add_tensor(
      sizes, dtype, suggested_memory_layout(sizes), shared_object_idx);
}

ValueRef ComputeGraph::add_tensorref(
    const std::vector<int64_t>& sizes,
    const api::ScalarType dtype,
    const void* const data) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(TensorRef(sizes, dtype, data));
  return idx;
}

ValueRef ComputeGraph::add_staging(
    const api::ScalarType dtype,
    const size_t numel) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(api::StorageBuffer(context(), dtype, numel));
  return idx;
}

ValueRef ComputeGraph::add_none() {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back();
  return idx;
}

ValueRef ComputeGraph::add_value_list(std::vector<ValueRef>&& value) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(std::move(value));
  return idx;
}

ValueRef ComputeGraph::add_string(std::string&& str) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(std::move(str));
  return idx;
}

ValueRef ComputeGraph::set_input_tensor(
    const ValueRef idx,
    const bool use_staging) {
  if (use_staging) {
    api::ScalarType dtype = get_tensor(idx)->dtype();
    size_t gpu_numel = get_tensor(idx)->gpu_numel();
    ValueRef staging_idx = add_staging(dtype, gpu_numel);
    add_staging_to_tensor_node(*this, staging_idx, idx);
    inputs_.push_back({idx, staging_idx});
    return staging_idx;
  }
  inputs_.push_back({idx, kDummyValueRef});
  return idx;
}

ValueRef ComputeGraph::set_output_tensor(
    const ValueRef idx,
    const bool use_staging) {
  if (use_staging) {
    api::ScalarType dtype = get_tensor(idx)->dtype();
    size_t gpu_numel = get_tensor(idx)->gpu_numel();
    ValueRef staging_idx = add_staging(dtype, gpu_numel);
    // We only run this when the tensor is non-empty.  When the underlying
    // tensor is empty (e.g. gpu_numel == 0), we do not allocate a VkImage to
    // tensor, we will not be able to bind the node for execution.
    if (gpu_numel > 0) {
      add_tensor_to_staging_node(*this, idx, staging_idx);
    }
    outputs_.push_back({idx, staging_idx});
    return staging_idx;
  }
  outputs_.push_back({idx, kDummyValueRef});
  return idx;
}

SharedObject& ComputeGraph::get_shared_object(const int64_t idx) {
  if (idx >= shared_objects_.size()) {
    shared_objects_.resize(static_cast<size_t>(idx + 1));
  }
  return shared_objects_.at(idx);
}

void ComputeGraph::update_descriptor_counts(
    const api::ShaderInfo& shader_info,
    bool execute) {
  api::DescriptorPoolConfig* config =
      execute ? &execute_descriptor_counts_ : &prepack_descriptor_counts_;

  config->descriptor_pool_max_sets += 1;
  for (const VkDescriptorType arg_type : shader_info.kernel_layout) {
    switch (arg_type) {
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        config->descriptor_uniform_buffer_count += 1;
        break;
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        config->descriptor_storage_buffer_count += 1;
        break;
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        config->descriptor_combined_sampler_count += 1;
        break;
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        config->descriptor_storage_image_count += 1;
        break;
      default:
        VK_THROW("Unsupported descriptor type!");
    }
  }
}

api::utils::uvec3 ComputeGraph::create_global_wg_size(const ValueRef idx) {
  if (is_buffer_storage(idx)) {
    return {uint32_t(texel_numel_of(idx)), 1u, 1u};
  }
  return image_extents_of(idx);
}

api::utils::uvec3 ComputeGraph::create_local_wg_size(const ValueRef idx) {
  if (is_buffer_storage(idx)) {
    return {64u, 1u, 1u};
  }

  const api::utils::uvec3 image_extents = image_extents_of(idx);
  api::utils::uvec3 local_group_size = {4, 4, 4};

  if (image_extents.data[2u] == 1) {
    if (image_extents.data[1u] == 1) {
      local_group_size.data[0u] = 64;
      local_group_size.data[1u] = 1;
      local_group_size.data[2u] = 1;
    } else if (image_extents.data[1u] < 8) {
      local_group_size.data[0u] = 16;
      local_group_size.data[1u] = 4;
      local_group_size.data[2u] = 1;
    } else {
      local_group_size.data[0u] = 8;
      local_group_size.data[1u] = 8;
      local_group_size.data[2u] = 1;
    }
  }
  return local_group_size;
}

void ComputeGraph::copy_into_staging(
    const ValueRef idx,
    const void* data,
    const size_t numel) {
  StagingPtr staging = get_staging(idx);
  size_t nbytes = numel * api::element_size(staging->dtype());
  copy_ptr_to_staging(data, *staging, nbytes);
}

void ComputeGraph::copy_from_staging(
    const ValueRef idx,
    void* data,
    const size_t numel) {
  StagingPtr staging = get_staging(idx);
  size_t nbytes = numel * api::element_size(staging->dtype());
  copy_staging_to_ptr(*staging, data, nbytes);
}

void ComputeGraph::prepare() {
#define MERGE_FIELD(field)                    \
  static_cast<uint32_t>(std::ceil(            \
      std::max(                               \
          execute_descriptor_counts_.field,   \
          prepack_descriptor_counts_.field) * \
      config_.descriptor_pool_safety_factor))

  uint32_t max_sets = MERGE_FIELD(descriptor_pool_max_sets);
  api::DescriptorPoolConfig config{
      max_sets,
      std::max(MERGE_FIELD(descriptor_uniform_buffer_count), max_sets),
      std::max(MERGE_FIELD(descriptor_storage_buffer_count), max_sets),
      std::max(MERGE_FIELD(descriptor_combined_sampler_count), max_sets),
      std::max(MERGE_FIELD(descriptor_storage_image_count), max_sets),
      1u,
  };

  if (!context_->descriptor_pool()) {
    context_->descriptor_pool().init(config);
  }
#undef MERGE_FIELD
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

void ComputeGraph::resize_input(
    const int64_t idx,
    const std::vector<int64_t>& new_sizes) {
  IOValueRef io_val = inputs_.at(idx);
  get_tensor(io_val.value)->virtual_resize(new_sizes);
}

void ComputeGraph::propagate_resize() {
  for (std::unique_ptr<ExecuteNode>& node : execute_nodes_) {
    node->trigger_resize(this);
  }
}

} // namespace vkcompute
