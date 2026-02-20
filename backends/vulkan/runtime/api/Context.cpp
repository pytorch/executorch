/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/Context.h>

#ifdef VULKAN_DEBUG
#include <iomanip>
#include <iostream>
#endif // VULKAN_DEBUG

#ifndef VULKAN_DESCRIPTOR_POOL_SIZE
#define VULKAN_DESCRIPTOR_POOL_SIZE 1024u
#endif

#ifndef VULKAN_QUERY_POOL_SIZE
#define VULKAN_QUERY_POOL_SIZE 4096u
#endif

namespace vkcompute {
namespace api {

Context::Context(vkapi::Adapter* adapter, const ContextConfig& config)
    : config_(config),
      // Important handles
      adapter_p_(adapter),
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
      images_to_clear_{},
      preferred_image_tiling_{VK_IMAGE_TILING_OPTIMAL} {
  if (adapter_p_->linear_tiling_3d_enabled()) {
    preferred_image_tiling_ = VK_IMAGE_TILING_LINEAR;
  }
}

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
    const utils::WorkgroupSize& local_wg_size,
    const uint32_t dispatch_id) {
  if (querypool_) {
    querypool_.shader_profile_begin(
        cmd_,
        dispatch_id,
        shader_name,
        vkapi::create_extent3d(global_wg_size),
        vkapi::create_extent3d((utils::uvec3)local_wg_size));
  }
}

void Context::report_shader_dispatch_end() {
  if (querypool_) {
    querypool_.shader_profile_end(cmd_);
  }
}

void Context::check_device_capabilities(const vkapi::ShaderInfo& shader) {
  if (shader.requires_shader_int16) {
    if (!adapter_p_->supports_int16_shader_types()) {
      throw vkapi::ShaderNotSupportedError(
          shader.kernel_name, vkapi::VulkanExtension::SHADER_INT16);
    }
  }
  if (shader.requires_16bit_storage) {
    if (!adapter_p_->supports_16bit_storage_buffers()) {
      throw vkapi::ShaderNotSupportedError(
          shader.kernel_name, vkapi::VulkanExtension::INT16_STORAGE);
    }
  }
  if (shader.requires_8bit_storage) {
    if (!adapter_p_->supports_8bit_storage_buffers()) {
      throw vkapi::ShaderNotSupportedError(
          shader.kernel_name, vkapi::VulkanExtension::INT8_STORAGE);
    }
  }
  if (shader.requires_integer_dot_product) {
    if (!adapter_p_->supports_int8_dot_product()) {
      throw vkapi::ShaderNotSupportedError(
          shader.kernel_name, vkapi::VulkanExtension::INTEGER_DOT_PRODUCT);
    }
  }
  if (shader.requires_shader_int64) {
    if (!adapter_p_->supports_int64_shader_types()) {
      throw vkapi::ShaderNotSupportedError(
          shader.kernel_name, vkapi::VulkanExtension::SHADER_INT64);
    }
  }
  if (shader.requires_shader_float64) {
    if (!adapter_p_->supports_float64_shader_types()) {
      throw vkapi::ShaderNotSupportedError(
          shader.kernel_name, vkapi::VulkanExtension::SHADER_FLOAT64);
    }
  }
}

vkapi::DescriptorSet Context::get_descriptor_set(
    const vkapi::ShaderInfo& shader_descriptor,
    const utils::WorkgroupSize& local_workgroup_size,
    const vkapi::SpecVarList& additional_constants,
    const uint32_t push_constants_size) {
  VkDescriptorSetLayout shader_layout =
      shader_layout_cache().retrieve(shader_descriptor.kernel_layout);

  VkPipelineLayout pipeline_layout =
      pipeline_layout_cache().retrieve(shader_layout, push_constants_size);

  vkapi::SpecVarList spec_constants = {
      SV(local_workgroup_size[0u]),
      SV(local_workgroup_size[1u]),
      SV(local_workgroup_size[2u])};

  spec_constants.append(additional_constants);

  VkPipeline pipeline = pipeline_cache().retrieve(
      {pipeline_layout_cache().retrieve(shader_layout, push_constants_size),
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
    const utils::uvec3& global_workgroup_size,
    const void* push_constants_data,
    const uint32_t push_constants_size) {
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

  if (push_constants_size > 0 && push_constants_data != nullptr) {
    const VkDescriptorSetLayout shader_layout =
        shader_layout_cache().retrieve(shader_descriptor.kernel_layout);
    const VkPipelineLayout pipeline_layout =
        pipeline_layout_cache().retrieve(shader_layout, push_constants_size);
    cmd_.set_push_constants(
        pipeline_layout, push_constants_data, push_constants_size);
  }

  cmd_.dispatch(effective_global_wg);
}

void Context::register_blit(
    vkapi::PipelineBarrier& pipeline_barrier,
    vkapi::VulkanImage& src,
    vkapi::VulkanImage& dst) {
  cmd_.insert_barrier(pipeline_barrier);
  cmd_.blit(src, dst);
}

void Context::submit_cmd_to_gpu(VkFence fence_handle, const bool final_use) {
  if (cmd_) {
    cmd_.end();
    adapter_p_->submit_cmd(
        queue_,
        cmd_.get_submit_handle(final_use),
        fence_handle,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE);

    submit_count_ = 0u;
  }
}

void Context::wait_for_queue() {
  VK_CHECK(vkQueueWaitIdle(queue().handle));
}

void Context::clear_resources() {
  command_pool_.flush();
  descriptor_pool_.flush();

  if (cmd_) {
    cmd_.invalidate();
  }

  std::lock_guard<std::mutex> bufferlist_lock(buffer_clearlist_mutex_);
  std::lock_guard<std::mutex> imagelist_lock(image_clearlist_mutex_);
  buffers_to_clear_.clear();
  images_to_clear_.clear();
}

void Context::flush() {
  wait_for_queue();
  clear_resources();
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

      return new Context(vkapi::runtime()->get_adapter_p(), config);
    } catch (...) {
    }

    return nullptr;
  }());

  return context.get();
}

#if defined(VK_KHR_pipeline_executable_properties) && \
    defined(ETVK_INSPECT_PIPELINES)

VkPipeline Context::get_shader_pipeline(
    const vkapi::ShaderInfo& shader,
    const vkapi::SpecVarList& additional_constants) {
  const uint32_t push_constants_size = 128u;

  VkDescriptorSetLayout shader_layout =
      shader_layout_cache().retrieve(shader.kernel_layout);
  VkPipelineLayout pipeline_layout =
      pipeline_layout_cache().retrieve(shader_layout, push_constants_size);

  const utils::WorkgroupSize local_workgroup_size(4u, 4u, 1u);
  vkapi::SpecVarList spec_constants = {
      SV(local_workgroup_size[0u]),
      SV(local_workgroup_size[1u]),
      SV(local_workgroup_size[2u])};

  spec_constants.append(additional_constants);

  VkPipeline pipeline = pipeline_cache().retrieve(
      {pipeline_layout, shader_cache().retrieve(shader), spec_constants});

  return pipeline;
}

std::vector<VkPipelineExecutablePropertiesKHR>
Context::get_pipeline_executable_props(const VkPipeline pipeline) {
  VkPipelineInfoKHR pipeline_info{
      VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,
      nullptr,
      pipeline,
  };

  uint32_t shader_props_count = 0u;
  vkGetPipelineExecutablePropertiesKHR(
      device(), &pipeline_info, &shader_props_count, nullptr);

  std::vector<VkPipelineExecutablePropertiesKHR> pipeline_props(
      shader_props_count);
  for (int i = 0; i < shader_props_count; i++) {
    pipeline_props.at(i).sType =
        VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    pipeline_props.at(i).pNext = nullptr;
  }
  vkGetPipelineExecutablePropertiesKHR(
      device(), &pipeline_info, &shader_props_count, pipeline_props.data());

  return pipeline_props;
}

std::tuple<
    std::vector<VkPipelineExecutableInternalRepresentationKHR>,
    std::vector<std::vector<char>>>
Context::get_shader_executable_irs(
    const VkPipeline pipeline,
    const uint32_t pipeline_exec_idx) {
  VkPipelineExecutableInfoKHR exec_info{
      VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
      nullptr,
      pipeline,
      pipeline_exec_idx,
  };

  uint32_t ir_count;
  VK_CHECK(vkGetPipelineExecutableInternalRepresentationsKHR(
      device(), &exec_info, &ir_count, nullptr));

  std::vector<VkPipelineExecutableInternalRepresentationKHR> irs(ir_count);
  for (int i = 0; i < ir_count; i++) {
    irs.at(i).sType =
        VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
    irs.at(i).pNext = nullptr;
    irs.at(i).pData = nullptr;
  }
  VK_CHECK(vkGetPipelineExecutableInternalRepresentationsKHR(
      device(), &exec_info, &ir_count, irs.data()));

  std::vector<std::vector<char>> irs_data(ir_count);
  for (int i = 0; i < ir_count; i++) {
    irs_data.at(i).resize(irs.at(i).dataSize);
    irs.at(i).pData = irs_data.at(i).data();
  }
  VK_CHECK(vkGetPipelineExecutableInternalRepresentationsKHR(
      device(), &exec_info, &ir_count, irs.data()));

  return std::make_tuple(irs, irs_data);
}

std::vector<VkPipelineExecutableStatisticKHR>
Context::get_shader_executable_stats(
    const VkPipeline pipeline,
    const uint32_t pipeline_exec_idx) {
  VkPipelineExecutableInfoKHR exec_info{
      VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
      nullptr,
      pipeline,
      pipeline_exec_idx,
  };

  uint32_t stats_count;
  VK_CHECK(vkGetPipelineExecutableStatisticsKHR(
      device(), &exec_info, &stats_count, NULL));

  std::vector<VkPipelineExecutableStatisticKHR> shader_stats(stats_count);
  for (int i = 0; i < stats_count; i++) {
    shader_stats.at(i).sType =
        VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
    shader_stats.at(i).pNext = nullptr;
  }
  vkGetPipelineExecutableStatisticsKHR(
      device(), &exec_info, &stats_count, shader_stats.data());

  return shader_stats;
}

std::ostream& operator<<(
    std::ostream& os,
    const VkPipelineExecutablePropertiesKHR& props) {
  os << std::left << std::setw(10) << "name: " << props.name << std::endl;
  os << std::left << std::setw(10) << "descr: " << props.description
     << std::endl;
  os << std::left << std::setw(10) << "subgroup: " << props.subgroupSize
     << std::endl;

  return os;
}

std::ostream& operator<<(
    std::ostream& os,
    const VkPipelineExecutableInternalRepresentationKHR& ir) {
  os << std::left << std::setw(10) << "descr: " << ir.description << std::endl;
  os << std::left << std::setw(10) << "isText: " << ir.isText << std::endl;
  os << std::left << std::setw(10) << "size: " << ir.dataSize << std::endl;
  if (ir.isText) {
    os << "text:" << std::endl;
    char* str = (char*)ir.pData;
    os << str << std::endl;
  }
  return os;
}

std::ostream& operator<<(
    std::ostream& os,
    VkPipelineExecutableStatisticKHR& stat) {
  os << stat.name << ": ";
  switch (stat.format) {
    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
      os << (stat.value.b32 ? "true" : "false") << std::endl;
      break;
    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
      os << stat.value.i64 << std::endl;
      break;
    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
      os << stat.value.u64 << std::endl;
      break;
    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
      os << stat.value.f64 << std::endl;
      break;
    default:
      break;
  }
  os << "    " << stat.description << std::endl;
  return os;
}

std::ostream& operator<<(
    std::ostream& os,
    std::vector<VkPipelineExecutableStatisticKHR>& shader_stats) {
  for (int i = 0; i < shader_stats.size(); i++) {
    VkPipelineExecutableStatisticKHR& stat = shader_stats.at(i);
    os << stat;
  }
  return os;
}

void Context::print_shader_executable_properties(
    const vkapi::ShaderInfo& shader,
    const vkapi::SpecVarList& spec_constants) {
  VkPipeline pipeline = get_shader_pipeline(shader, spec_constants);

  std::vector<VkPipelineExecutablePropertiesKHR> pipeline_props_list =
      get_pipeline_executable_props(pipeline);

  VK_CHECK_COND(pipeline_props_list.size() == 1u);

  std::cout << pipeline_props_list.at(0) << std::endl;

  std::tuple<
      std::vector<VkPipelineExecutableInternalRepresentationKHR>,
      std::vector<std::vector<char>>>
      irs_and_irs_data = get_shader_executable_irs(pipeline, 0u);

  std::vector<VkPipelineExecutableInternalRepresentationKHR>& irs =
      std::get<0>(irs_and_irs_data);

  std::cout << "Found " << irs.size() << " IRs" << std::endl << std::endl;
  for (int i = 0; i < irs.size(); i++) {
    std::cout << "====== IR " << i << ": " << irs.at(i).name
              << " ======" << std::endl;
    std::cout << irs.at(i) << std::endl;
  }

  std::vector<VkPipelineExecutableStatisticKHR> shader_stats =
      get_shader_executable_stats(pipeline, 0u);
  std::cout << "Found " << shader_stats.size() << " Statistics" << std::endl;
  if (shader_stats.size() > 0) {
    std::cout << "====== Statistics: ======" << std::endl;
    std::cout << shader_stats << std::endl;
  }
}

#endif // VK_KHR_pipeline_executable_properties && ETVK_INSPECT_PIPELINES

} // namespace api
} // namespace vkcompute
