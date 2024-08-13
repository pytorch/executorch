/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/Context.h>

#include <executorch/backends/vulkan/runtime/vk_api/VkUtils.h>

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
      adapter_p_(vkapi::runtime()->get_adapter_p(adapter_i)),
      device_(adapter_p_->device_handle()),
      queue_(adapter_p_->request_queue()),
      // Resource pools
      command_pool_(device_, queue_.family_index, config_.cmd_pool_config),
      descriptor_pool_(device_, config_.descriptor_pool_config),
      fences_(device_),
      // Profiling
      querypool_(config_.query_pool_config, nullptr),
      // Command buffer submission
      cmd_mutex_{},
      cmd_(VK_NULL_HANDLE, 0u),
      submit_count_{0u},
      // Memory Management
      buffer_clearlist_mutex_{},
      buffers_to_clear_{},
      image_clearlist_mutex_{},
      images_to_clear_{} {}

Context::~Context() {
  try {
    flush();
    // Let the device know the context is done with the queue
    adapter_p_->return_queue(queue_);
  } catch (...) {
  }
}

void Context::initialize_querypool() {
  querypool_.initialize(adapter_p_);
}

void Context::cmd_reset_querypool() {
  if (querypool_) {
    set_cmd();
    querypool_.reset_querypool(cmd_);
  }
}

void Context::report_shader_dispatch_start(
    const std::string& shader_name,
    const utils::uvec3& global_wg_size,
    const utils::uvec3& local_wg_size,
    const uint32_t dispatch_id) {
  if (querypool_) {
    querypool_.shader_profile_begin(
        cmd_,
        dispatch_id,
        shader_name,
        vkapi::create_extent3d(global_wg_size),
        vkapi::create_extent3d(local_wg_size));
  }
}

void Context::report_shader_dispatch_end() {
  if (querypool_) {
    querypool_.shader_profile_end(cmd_);
  }
}

vkapi::DescriptorSet Context::get_descriptor_set(
    const vkapi::ShaderInfo& shader_descriptor,
    const utils::uvec3& local_workgroup_size,
    const vkapi::SpecVarList& additional_constants) {
  VkDescriptorSetLayout shader_layout =
      shader_layout_cache().retrieve(shader_descriptor.kernel_layout);

  VkPipelineLayout pipeline_layout =
      pipeline_layout_cache().retrieve(shader_layout);

  vkapi::SpecVarList spec_constants = {
      SV(local_workgroup_size[0u]),
      SV(local_workgroup_size[1u]),
      SV(local_workgroup_size[2u])};

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
    const vkapi::DescriptorSet& descriptors,
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::ShaderInfo& shader_descriptor,
    const utils::uvec3& global_workgroup_size) {
  // Adjust the global workgroup size based on the output tile size
  uint32_t global_wg_w = utils::div_up(
      global_workgroup_size[0u], shader_descriptor.out_tile_size[0u]);
  uint32_t global_wg_h = utils::div_up(
      global_workgroup_size[1u], shader_descriptor.out_tile_size[1u]);
  uint32_t global_wg_d = utils::div_up(
      global_workgroup_size[2u], shader_descriptor.out_tile_size[2u]);

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

      const vkapi::CommandPoolConfig cmd_config{
          32u, // cmdPoolInitialSize
          8u, // cmdPoolBatchSize
      };

      const vkapi::DescriptorPoolConfig descriptor_pool_config{
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorPoolMaxSets
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorUniformBufferCount
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorStorageBufferCount
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorCombinedSamplerCount
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorStorageImageCount
          32u, // descriptorPileSizes
      };

      const vkapi::QueryPoolConfig query_pool_config{
          VULKAN_QUERY_POOL_SIZE, // maxQueryCount
          256u, // initialReserveSize
      };

      const ContextConfig config{
          cmd_submit_frequency,
          cmd_config,
          descriptor_pool_config,
          query_pool_config,
      };

      return new Context(vkapi::runtime()->default_adapter_i(), config);
    } catch (...) {
    }

    return nullptr;
  }());

  return context.get();
}

} // namespace api
} // namespace vkcompute
