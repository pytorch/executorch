/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/utils/MacroUtils.h>

#include <executorch/backends/vulkan/runtime/vk_api/Adapter.h>
#include <executorch/backends/vulkan/runtime/vk_api/Command.h>
#include <executorch/backends/vulkan/runtime/vk_api/Descriptor.h>
#include <executorch/backends/vulkan/runtime/vk_api/Fence.h>
#include <executorch/backends/vulkan/runtime/vk_api/QueryPool.h>
#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>

namespace vkcompute {
namespace api {

struct ContextConfig final {
  uint32_t cmd_submit_frequency;
  vkapi::CommandPoolConfig cmd_pool_config;
  vkapi::DescriptorPoolConfig descriptor_pool_config;
  vkapi::QueryPoolConfig query_pool_config;
};

//
// Vulkan Context holds onto all relevant Vulkan state as it pertains to our
// use of Vulkan in PyTorch. A Context is associated with one, and only one,
// Adapter as a precursor to multi-GPU support. All Vulkan tensors in PyTorch
// are associated with a Context to make tensor <-> device affinity explicit.
// The context is currently a global object, but technically it does not need
// to be if we were to make it explicit to the user.
//

class Context final {
 public:
  explicit Context(size_t adapter_i, const ContextConfig&);

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  ~Context();

 private:
  // Config
  ContextConfig config_;
  // Important handles
  vkapi::Adapter* adapter_p_;
  VkDevice device_;
  vkapi::Adapter::Queue queue_;
  // Resource Pools
  vkapi::CommandPool command_pool_;
  vkapi::DescriptorPool descriptor_pool_;
  vkapi::FencePool fences_;
  // Diagnostics
  vkapi::QueryPool querypool_;
  // Command buffers submission
  std::mutex cmd_mutex_;
  vkapi::CommandBuffer cmd_;
  uint32_t submit_count_;
  // Memory Management
  std::mutex buffer_clearlist_mutex_;
  std::vector<vkapi::VulkanBuffer> buffers_to_clear_;
  std::mutex image_clearlist_mutex_;
  std::vector<vkapi::VulkanImage> images_to_clear_;

 public:
  // Adapter access

  inline vkapi::Adapter* adapter_ptr() {
    return adapter_p_;
  }

  inline VkDevice device() {
    return device_;
  }

  inline VkQueue queue() {
    return queue_.handle;
  }

  // Device Caches

  inline vkapi::ShaderLayoutCache& shader_layout_cache() {
    return adapter_ptr()->shader_layout_cache();
  }

  inline vkapi::ShaderCache& shader_cache() {
    return adapter_ptr()->shader_cache();
  }

  inline vkapi::PipelineLayoutCache& pipeline_layout_cache() {
    return adapter_ptr()->pipeline_layout_cache();
  }

  inline vkapi::ComputePipelineCache& pipeline_cache() {
    return adapter_ptr()->compute_pipeline_cache();
  }

  // Resource Pools

  inline vkapi::DescriptorPool& descriptor_pool() {
    return descriptor_pool_;
  }

  inline vkapi::FencePool& fences() {
    return fences_;
  }

  // Diagnostics

  inline vkapi::QueryPool& querypool() {
    return querypool_;
  }

  /*
   * By default, the querypool attached to a Context instance is uninitialized.
   * This function triggers the querypool to be created via vkCreateQueryPool.
   */
  void initialize_querypool();

  /*
   * Encodes a vkResetQueryPool command to the current command buffer, and reset
   * the internal state of the querypool. If the querypool is not initialized
   * this function is a no-op.
   */
  void cmd_reset_querypool();

  /*
   * Encodes a vkCmdWriteTimestamp command to the current command buffer and
   * record some metadata about the shader that will be dispatched. If the
   * querypool is not initialized this function is a no-op.
   */
  void report_shader_dispatch_start(
      const std::string& shader_name,
      const utils::uvec3& global_wg_size,
      const utils::uvec3& local_wg_size,
      const uint32_t dispatch_id = UINT32_MAX);

  /*
   * Encodes a vkCmdWriteTimstamp command to the current command buffer to
   * record when the last shader that was dispatched has completed execution.
   * If the querypool is not initialized this function is a no-op.
   */
  void report_shader_dispatch_end();

  // Memory Management

  void register_buffer_cleanup(vkapi::VulkanBuffer& buffer) {
    std::lock_guard<std::mutex> bufferlist_lock(buffer_clearlist_mutex_);
    buffers_to_clear_.emplace_back(std::move(buffer));
  }

  void register_image_cleanup(vkapi::VulkanImage& image) {
    std::lock_guard<std::mutex> imagelist_lock(image_clearlist_mutex_);
    images_to_clear_.emplace_back(std::move(image));
  }

  // GPU RPC

  inline std::unique_lock<std::mutex> dispatch_lock() {
    return std::unique_lock<std::mutex>(cmd_mutex_);
  }

  inline void set_cmd(bool reusable = false) {
    if (!cmd_) {
      cmd_ = command_pool_.get_new_cmd(reusable);
      cmd_.begin();
    }
  }

  vkapi::DescriptorSet get_descriptor_set(
      const vkapi::ShaderInfo&,
      const utils::uvec3&,
      const vkapi::SpecVarList&);

  inline vkapi::DescriptorSet get_descriptor_set(
      const vkapi::ShaderInfo& shader_descriptor,
      const utils::uvec3& local_work_group_size) {
    return get_descriptor_set(shader_descriptor, local_work_group_size, {});
  }

  void register_shader_dispatch(
      const vkapi::DescriptorSet&,
      vkapi::PipelineBarrier&,
      const vkapi::ShaderInfo&,
      const utils::uvec3&);

  template <typename... Arguments>
  bool submit_compute_job(
      const vkapi::ShaderInfo&,
      vkapi::PipelineBarrier&,
      const utils::uvec3&,
      const utils::uvec3&,
      const vkapi::SpecVarList&,
      VkFence fence_handle,
      const uint32_t dispatch_id,
      Arguments&&...);

  void submit_cmd_to_gpu(
      VkFence fence_handle = VK_NULL_HANDLE,
      const bool final_use = false);

  void flush();
};

bool available();

// The global runtime is retrieved using this function, where it is declared as
// a static local variable.
Context* context();

namespace detail {

inline void arg_is_empty(
    bool& any_is_empty,
    const vkapi::VulkanBuffer& buffer) {
  // bool(buffer) will evaluate to false if no memory has been allocated
  any_is_empty = any_is_empty || !buffer;
}

inline void arg_is_empty(bool& any_is_empty, const vkapi::VulkanImage& image) {
  // bool(image) will evaluate to false if no memory has been allocated
  any_is_empty = any_is_empty || !image;
}

inline void arg_is_empty(
    bool& any_is_empty,
    const vkapi::BufferBindInfo& bind_info) {
  any_is_empty = any_is_empty || (bind_info.handle == VK_NULL_HANDLE);
}

/*
  Reports if any VulkanBuffer or VulkanImage argument in a variadic argument
  list does not have any memory associated with it.
 */
template <typename... Arguments>
inline bool any_arg_is_empty(Arguments&&... arguments) {
  bool any_is_empty = false;
  VK_UNUSED const int _[]{
      0,
      (arg_is_empty(any_is_empty, std::forward<Arguments>(arguments)), 0)...,
  };

  return any_is_empty;
}

template <size_t... Indices, typename... Arguments>
inline void bind(
    vkapi::DescriptorSet& descriptor_set,
    const std::index_sequence<Indices...>&,
    Arguments&&... arguments) {
  VK_UNUSED const int _[]{
      0,
      (descriptor_set.bind(Indices, std::forward<Arguments>(arguments)), 0)...,
  };
}

} // namespace detail

/*
  Records a compute shader dispatch into the current command buffer. If the
  number of submit_*_job calls exceeds the configured frequency, or if a fence
  is provided, then the command buffer is submitted to the GPU for execution.
  Returns a bool indicating whether or not the function call resulted in a GPU
  queue submission.
 */
template <typename... Arguments>
inline bool Context::submit_compute_job(
    const vkapi::ShaderInfo& shader,
    vkapi::PipelineBarrier& pipeline_barrier,
    const utils::uvec3& global_work_group,
    const utils::uvec3& local_work_group_size,
    const vkapi::SpecVarList& specialization_constants,
    VkFence fence_handle,
    const uint32_t dispatch_id,
    Arguments&&... arguments) {
  // If any of the provided arguments does not have memory associated with it,
  // then exit early as there is no work to be done. However, if a fence has
  // been passed the command buffer is not empty, then the current command
  // buffer must still be submitted so that the fence can be signaled.
  if (detail::any_arg_is_empty(arguments...)) {
    if (fence_handle != VK_NULL_HANDLE && submit_count_ > 0) {
      submit_cmd_to_gpu(fence_handle);
      return true;
    }
    return false;
  }

  // Serialize recording to the shared command buffer. Do not initialize with a
  // mutex just yet, since in some cases it will be externally managed.
  std::unique_lock<std::mutex> cmd_lock;
  // If a fence was passed, then assume that the host intends to sync with
  // the GPU, implying there will be imminent calls to fence.wait() and flush().
  // We therefore assume the mutex is externally managed in this case, and the
  // calling thread has already locked the mutex prior to calling the function,
  // and will release the mutex manually after calling flush(). This will
  // prevent more dispatches from being recorded until we have flushed the
  // Context.
  if (fence_handle == VK_NULL_HANDLE) {
    cmd_lock = std::unique_lock<std::mutex>(cmd_mutex_);
  }

  set_cmd();

  report_shader_dispatch_start(
      shader.kernel_name,
      global_work_group,
      local_work_group_size,
      dispatch_id);

  // Factor out template parameter independent code to minimize code bloat.
  vkapi::DescriptorSet descriptor_set = get_descriptor_set(
      shader, local_work_group_size, specialization_constants);

  detail::bind(
      descriptor_set,
      std::index_sequence_for<Arguments...>{},
      std::forward<Arguments>(arguments)...);

  // Factor out template parameter independent code to minimize code bloat.
  register_shader_dispatch(
      descriptor_set, pipeline_barrier, shader, global_work_group);

  report_shader_dispatch_end();

  submit_count_++;
  if (fence_handle != VK_NULL_HANDLE ||
      submit_count_ >= config_.cmd_submit_frequency) {
    submit_cmd_to_gpu(fence_handle);
    return true;
  }

  return false;
}

} // namespace api
} // namespace vkcompute
