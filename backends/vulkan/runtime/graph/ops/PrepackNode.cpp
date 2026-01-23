/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/DispatchNode.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/BindingUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

namespace vkcompute {

vkapi::ShaderInfo get_noop_shader(ComputeGraph& graph, const ValueRef packed) {
  std::string noop_shader_name("no_op");
  add_dtype_suffix(noop_shader_name, graph.dtype_of(packed));
  add_storage_type_suffix(noop_shader_name, graph.storage_type_of(packed));
  return VK_KERNEL_FROM_STR(noop_shader_name);
}

PrepackNode::PrepackNode(
    ComputeGraph& graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const utils::uvec3& local_workgroup_size,
    const ValueRef tref,
    const ValueRef packed,
    const vkapi::ParamsBindList& params,
    const vkapi::SpecVarList& spec_vars,
    const std::vector<PushConstantDataInfo>& push_constants)
    : shader_(shader),
      noop_shader_(get_noop_shader(graph, packed)),
      global_workgroup_size_(global_workgroup_size),
      local_workgroup_size_(local_workgroup_size),
      tref_(tref),
      packed_(packed),
      params_(params),
      spec_vars_(spec_vars),
      push_constants_(push_constants) {
  graph.update_descriptor_counts(shader, /*execute = */ false);
  graph.update_descriptor_counts(noop_shader_, /*execute = */ false);
}

api::StagingBuffer PrepackNode::create_staging_buffer(ComputeGraph* graph) {
  // If no TensorRef is provided, create a staging buffer of zeros based on the
  // Tensor metadata.
  if (graph->val_is_none(tref_)) {
    const std::vector<int64_t> packed_sizes = graph->sizes_of(packed_);
    size_t numel = utils::multiply_integers(packed_sizes);
    api::StagingBuffer staging(
        graph->context(),
        graph->dtype_of(packed_),
        numel,
        vkapi::CopyDirection::HOST_TO_DEVICE);
    staging.set_staging_zeros();
    return staging;
  }

  TensorRefPtr tref = graph->get_tref(tref_);
  size_t numel = utils::multiply_integers(tref->sizes);
  api::StagingBuffer staging(
      graph->context(),
      tref->dtype,
      numel,
      vkapi::CopyDirection::HOST_TO_DEVICE);
  graph->update_staging_nbytes_in_cmd(staging.buffer().mem_size_as_size_t());
  size_t nbytes = numel * vkapi::element_size(tref->dtype);

  // In some cases the staging dtype will diverge from the TensorRef dtype. The
  // most common case for this is when the tensor data is float16, but the GPU
  // does not support 16-bit storage buffers. In these cases, the tensor data
  // is manually casted to the staging dtype.
  vkapi::ScalarType staging_dtype = staging.dtype();
  vkapi::ScalarType tref_dtype = tref->dtype;
  if (staging_dtype == tref_dtype) {
    staging.copy_from(tref->data, nbytes);
  } else {
    // Hard-coded type conversion cases
    if (tref_dtype == vkapi::kHalf && staging_dtype == vkapi::kFloat) {
      const int16_t* casted_data = reinterpret_cast<const int16_t*>(tref->data);
      staging.cast_half_to_float_and_copy_from(casted_data, numel);
    } else if (tref_dtype == vkapi::kLong && staging_dtype == vkapi::kInt) {
      const int64_t* casted_data = reinterpret_cast<const int64_t*>(tref->data);
      staging.cast_and_copy_from<int64_t, int32_t>(casted_data, numel);
    } else if (tref_dtype == vkapi::kDouble && staging_dtype == vkapi::kFloat) {
      const double* casted_data = reinterpret_cast<const double*>(tref->data);
      staging.cast_and_copy_from<double, float>(casted_data, numel);
    } else {
      VK_THROW(
          "Unsupported type conversion from ",
          tref_dtype,
          " to staging dtype ",
          staging_dtype);
    }
  }

  // Once the staging buffer is copied, if the TensorRef owns a FreeableBuffer,
  // it can be freed.
  tref->free_buffer();
  return staging;
}

void PrepackNode::prepare_pipelines(ComputeGraph* graph) {
  graph->register_pipeline_to_create(
      shader_, local_workgroup_size_, spec_vars_, push_constants_);
  graph->register_pipeline_to_create(
      noop_shader_, utils::WorkgroupSize(1, 1, 1), {}, {});
}

void PrepackNode::encode(ComputeGraph* graph) {
  api::Context* const context = graph->context();

  context->check_device_capabilities(shader_);

  api::StagingBuffer staging = create_staging_buffer(graph);

  std::unique_lock<std::mutex> cmd_lock = context->dispatch_lock();

  std::array<uint8_t, kMaxPushConstantSize> push_constants_data;
  uint32_t push_constants_offset = 0;

  for (const auto& push_constant : push_constants_) {
    push_constants_offset += push_constant.write(
        push_constants_data.data(),
        push_constants_offset,
        kMaxPushConstantSize);
  }

  {
    // If the vTensor is not yet bound to a memory allocation, create a new one
    // and aquire it.
    graph->create_dedicated_allocation_for(packed_);

    vkapi::PipelineBarrier pipeline_barrier{};
    vkapi::DescriptorSet descriptor_set = context->get_descriptor_set(
        shader_, local_workgroup_size_, spec_vars_, push_constants_offset);

    uint32_t idx = 0;
    graph->bind_tensor_to_descriptor_set(
        packed_,
        pipeline_barrier,
        vkapi::MemoryAccessType::WRITE,
        descriptor_set,
        idx++);
    bind_staging_to_descriptor_set(staging, descriptor_set, idx++);
    bind_params_to_descriptor_set(params_, descriptor_set, idx);

    context->register_shader_dispatch(
        descriptor_set,
        pipeline_barrier,
        shader_,
        global_workgroup_size_,
        push_constants_data.data(),
        push_constants_offset);
  }

  // Submit a compute shader that performs a no-op with the packed tensor in
  // order to trigger an image layout transition from GENERAL to
  // READ_ONLY_OPTIMAL. This ensures that future uses of the tensor will be
  // bound with the correct image layout.
  {
    vkapi::PipelineBarrier pipeline_barrier{};
    vkapi::DescriptorSet descriptor_set = context->get_descriptor_set(
        noop_shader_, utils::WorkgroupSize(1, 1, 1));

    graph->bind_tensor_to_descriptor_set(
        packed_,
        pipeline_barrier,
        vkapi::MemoryAccessType::READ,
        descriptor_set,
        0);

    context->register_shader_dispatch(
        descriptor_set, pipeline_barrier, noop_shader_, {1, 1, 1});
  }
}

} // namespace vkcompute
