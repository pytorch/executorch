/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/BindingUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

namespace vkcompute {

api::ShaderInfo get_noop_shader(ComputeGraph& graph, const ValueRef packed) {
  std::string noop_shader_name("no_op");
  vTensorPtr t_packed = graph.get_tensor(packed);
  add_dtype_suffix(noop_shader_name, *t_packed);
  add_storage_type_suffix(noop_shader_name, *t_packed);
  return VK_KERNEL_FROM_STR(noop_shader_name);
}

PrepackNode::PrepackNode(
    ComputeGraph& graph,
    const api::ShaderInfo& shader,
    const api::utils::uvec3& global_workgroup_size,
    const api::utils::uvec3& local_workgroup_size,
    const ValueRef tref,
    const ValueRef packed,
    const api::ParamsBindList& params,
    const api::SpecVarList& spec_vars)
    : shader_(shader),
      noop_shader_(get_noop_shader(graph, packed)),
      global_workgroup_size_(global_workgroup_size),
      local_workgroup_size_(local_workgroup_size),
      tref_(tref),
      packed_(packed),
      params_(params),
      spec_vars_(spec_vars) {
  graph.update_descriptor_counts(shader, /*execute = */ false);
  graph.update_descriptor_counts(noop_shader_, /*execute = */ false);
}

api::StorageBuffer PrepackNode::create_staging_buffer(ComputeGraph* graph) {
  vTensorPtr packed = graph->get_tensor(packed_);

  // If no TensorRef is provided, create a staging buffer of zeros according to
  // the vTensor metadata.
  if (graph->val_is_none(tref_)) {
    size_t numel = api::utils::multiply_integers(packed->sizes());
    api::StorageBuffer staging(graph->context(), packed->dtype(), numel);
    size_t nbytes = numel * api::element_size(packed->dtype());
    set_staging_zeros(staging, nbytes);
    return staging;
  }

  TensorRefPtr tref = graph->get_tref(tref_);
  size_t numel = api::utils::multiply_integers(tref->sizes);
  api::StorageBuffer staging(graph->context(), tref->dtype, numel);
  size_t nbytes = numel * api::element_size(tref->dtype);
  copy_ptr_to_staging(tref->data, staging, nbytes);
  return staging;
}

void PrepackNode::encode(ComputeGraph* graph) {
  api::Context* const context = graph->context();

  vTensorPtr packed = graph->get_tensor(packed_);
  api::StorageBuffer staging = create_staging_buffer(graph);

  std::unique_lock<std::mutex> cmd_lock = context->dispatch_lock();

  {
    api::PipelineBarrier pipeline_barrier{};
    api::DescriptorSet descriptor_set =
        context->get_descriptor_set(shader_, local_workgroup_size_, spec_vars_);

    uint32_t idx = 0;
    bind_tensor_to_descriptor_set(
        *packed,
        pipeline_barrier,
        api::MemoryAccessType::WRITE,
        descriptor_set,
        idx++);
    bind_staging_to_descriptor_set(staging, descriptor_set, idx++);
    bind_params_to_descriptor_set(params_, descriptor_set, idx);

    context->register_shader_dispatch(
        descriptor_set, pipeline_barrier, shader_, global_workgroup_size_);
  }

  // Submit a compute shader that performs a no-op with the packed tensor in
  // order to trigger an image layout transition from GENERAL to
  // READ_ONLY_OPTIMAL. This ensures that future uses of the tensor will be
  // bound with the correct image layout.
  {
    api::PipelineBarrier pipeline_barrier{};
    api::DescriptorSet descriptor_set =
        context->get_descriptor_set(noop_shader_, {1, 1, 1});

    bind_tensor_to_descriptor_set(
        *packed,
        pipeline_barrier,
        api::MemoryAccessType::READ,
        descriptor_set,
        0);

    context->register_shader_dispatch(
        descriptor_set, pipeline_barrier, noop_shader_, {1, 1, 1});
  }
}

} // namespace vkcompute
