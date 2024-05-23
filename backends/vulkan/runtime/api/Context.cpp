/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/Context.h>

#include <cstring>
#include <memory>
#include <sstream>

#ifndef VULKAN_DESCRIPTOR_POOL_SIZE
#define VULKAN_DESCRIPTOR_POOL_SIZE 1024u
#endif

#ifndef VULKAN_QUERY_POOL_SIZE
#define VULKAN_QUERY_POOL_SIZE 4096u
#endif

namespace vkcompute {
namespace api {

Context::Context(size_t adapter_i, const ContextConfig& config)
    : config_(config),
      // Important handles
      adapter_p_(runtime()->get_adapter_p(adapter_i)),
      device_(adapter_p_->device_handle()),
      queue_(adapter_p_->request_queue()),
      // Resource pools
      command_pool_(device_, queue_.family_index, config_.cmd_pool_config),
      descriptor_pool_(device_, config_.descriptor_pool_config),
      fences_(device_),
// Diagnostics
#ifdef USE_VULKAN_GPU_DIAGNOSTICS
      querypool_(config_.query_pool_config, adapter_p_),
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */
      // Command buffer submission
      cmd_mutex_{},
      cmd_(VK_NULL_HANDLE, 0u),
      submit_count_{0u},
      // Memory Management
      buffer_clearlist_mutex_{},
      buffers_to_clear_{},
      image_clearlist_mutex_{},
      images_to_clear_{} {
}

Context::~Context() {
  try {
    flush();
    // Let the device know the context is done with the queue
    adapter_p_->return_queue(queue_);
  } catch (...) {
  }
}

DescriptorSet Context::get_descriptor_set(
    const ShaderInfo& shader_descriptor,
    const utils::uvec3& local_workgroup_size,
    const SpecVarList& additional_constants) {
  VkDescriptorSetLayout shader_layout =
      shader_layout_cache().retrieve(shader_descriptor.kernel_layout);

  VkPipelineLayout pipeline_layout =
      pipeline_layout_cache().retrieve(shader_layout);

  SpecVarList spec_constants = {
      SV(local_workgroup_size.data[0u]),
      SV(local_workgroup_size.data[1u]),
      SV(local_workgroup_size.data[2u])};

  spec_constants.append(additional_constants);

  VkPipeline pipeline = pipeline_cache().retrieve(
      {pipeline_layout_cache().retrieve(shader_layout),
       shader_cache().retrieve(shader_descriptor),
       spec_constants});

  cmd_.bind_pipeline(pipeline, pipeline_layout, local_workgroup_size);

  return descriptor_pool().get_descriptor_set(
      shader_layout, shader_descriptor.kernel_layout);
}

void Context::register_shader_dispatch(
    const DescriptorSet& descriptors,
    PipelineBarrier& pipeline_barrier,
    const ShaderInfo& shader_descriptor,
    const utils::uvec3& global_workgroup_size) {
  // Adjust the global workgroup size based on the output tile size
  uint32_t global_wg_w = utils::div_up(
      global_workgroup_size.data[0u], shader_descriptor.out_tile_size.data[0u]);
  uint32_t global_wg_h = utils::div_up(
      global_workgroup_size.data[1u], shader_descriptor.out_tile_size.data[1u]);
  uint32_t global_wg_d = utils::div_up(
      global_workgroup_size.data[2u], shader_descriptor.out_tile_size.data[2u]);

  // Submitting a global work group size of 0 is undefined behaviour. If this is
  // detected then submit a single workgroup instead.
  if (global_wg_w == 0u || global_wg_h == 0u || global_wg_d == 0u) {
    global_wg_w = 1u;
    global_wg_h = 1u;
    global_wg_d = 1u;
  }

  const utils::uvec3 effective_global_wg = {
      global_wg_w,
      global_wg_h,
      global_wg_d,
  };

  cmd_.bind_descriptors(descriptors.get_bind_handle());
  cmd_.insert_barrier(pipeline_barrier);

  cmd_.dispatch(effective_global_wg);
}

void Context::submit_cmd_to_gpu(VkFence fence_handle, const bool final_use) {
  if (cmd_) {
    cmd_.end();
    adapter_p_->submit_cmd(
        queue_, cmd_.get_submit_handle(final_use), fence_handle);

    submit_count_ = 0u;
  }
}

void Context::flush() {
  VK_CHECK(vkQueueWaitIdle(queue()));

  command_pool_.flush();
  descriptor_pool_.flush();

  // If there is an existing command buffer, invalidate it
  if (cmd_) {
    cmd_.invalidate();
  }

  std::lock_guard<std::mutex> bufferlist_lock(buffer_clearlist_mutex_);
  std::lock_guard<std::mutex> imagelist_lock(image_clearlist_mutex_);
  buffers_to_clear_.clear();
  images_to_clear_.clear();
}

bool available() {
  return context();
}

Context* context() {
  static const std::unique_ptr<Context> context([]() -> Context* {
    try {
      const uint32_t cmd_submit_frequency = 16u;

      const CommandPoolConfig cmd_config{
          32u, // cmdPoolInitialSize
          8u, // cmdPoolBatchSize
      };

      const DescriptorPoolConfig descriptor_pool_config{
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorPoolMaxSets
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorUniformBufferCount
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorStorageBufferCount
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorCombinedSamplerCount
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorStorageImageCount
          32u, // descriptorPileSizes
      };

      const QueryPoolConfig query_pool_config{
          VULKAN_QUERY_POOL_SIZE, // maxQueryCount
          256u, // initialReserveSize
      };

      const ContextConfig config{
          cmd_submit_frequency,
          cmd_config,
          descriptor_pool_config,
          query_pool_config,
      };

      return new Context(runtime()->default_adapter_i(), config);
    } catch (...) {
    }

    return nullptr;
  }());

  return context.get();
}

//
// UniformParamsBuffer
//

namespace {

void memcpy_to_buffer(const VulkanBuffer& src, VulkanBuffer& dst) {
  MemoryMap dst_mapping(dst, MemoryAccessType::WRITE);

  MemoryMap src_mapping(src, MemoryAccessType::READ);
  src_mapping.invalidate();

  void* dst_ptr = dst_mapping.template data<void>();
  void* src_ptr = src_mapping.template data<void>();

  // @lint-ignore CLANGTIDY facebook-security-vulnerable-memcpy
  memcpy(dst_ptr, src_ptr, src.mem_size());
}

} // namespace

UniformParamsBuffer::UniformParamsBuffer(const UniformParamsBuffer& other)
    : context_p_(other.context_p_), vulkan_buffer_{} {
  if (other.vulkan_buffer_) {
    vulkan_buffer_ = context_p_->adapter_ptr()->vma().create_uniform_buffer(
        other.vulkan_buffer_.mem_size());

    memcpy_to_buffer(other.vulkan_buffer_, vulkan_buffer_);
  }
}

UniformParamsBuffer& UniformParamsBuffer::operator=(
    const UniformParamsBuffer& other) {
  if (&other != this) {
    context_p_ = other.context_p_;

    // Move vulkan_buffer_ to another VulkanBuffer for cleanup
    if (vulkan_buffer_) {
      VulkanBuffer temp_buffer(std::move(vulkan_buffer_));
      context_p_->register_buffer_cleanup(temp_buffer);
    }
    // vulkan_buffer_ should now be empty

    if (other.vulkan_buffer_) {
      vulkan_buffer_ = context_p_->adapter_ptr()->vma().create_uniform_buffer(
          other.vulkan_buffer_.mem_size());

      memcpy_to_buffer(other.vulkan_buffer_, vulkan_buffer_);
    }
  }

  return *this;
}

ParamsBindList::ParamsBindList(
    std::initializer_list<const BufferBindInfo> init_list) {
  bind_infos.resize(init_list.size());
  std::copy(init_list.begin(), init_list.end(), bind_infos.begin());
}

void ParamsBindList::append(const ParamsBindList& other) {
  bind_infos.insert(
      bind_infos.end(), other.bind_infos.begin(), other.bind_infos.end());
}

} // namespace api
} // namespace vkcompute
