/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/arm/runtime/VGFVulkanFeatures.h>
#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>
#include <executorch/backends/vulkan_shared/runtime/SharedVulkanDevice.h>
#include <executorch/backends/vulkan_shared/runtime/SharedVulkanRuntimeConfig.h>
#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/core/result.h>

#include <memory>

namespace executorch {
namespace backends {
namespace vgf {

// Each representation has its own pool, but retains the registry's device.
struct VgfSharedDeviceState final {
  vulkan_shared::SharedVulkanDevicePtr device;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  bool neural_statistics_device_enabled = false;
  VgfHostMemoryImportCapabilities host_memory_import_capabilities{};

  VgfSharedDeviceState() = default;
  VgfSharedDeviceState(const VgfSharedDeviceState&) = delete;
  VgfSharedDeviceState& operator=(const VgfSharedDeviceState&) = delete;
  ~VgfSharedDeviceState();
};

// This predicate deliberately consumes ENABLED logical-device features, not
// vkGetPhysicalDeviceFeatures2 support bits. It can be tested without a GPU.
inline bool vgf_shared_features_compatible(
    const vulkan_shared::SharedVulkanDeviceFeatures& f) {
  return f.arm_tensors && f.arm_shader_tensor_access && f.arm_data_graph &&
      f.arm_data_graph_shader_module && f.shader_int16 && f.shader_int64 &&
      f.shader_int8 && f.storage_buffer_8bit_access && f.host_query_reset &&
      f.synchronization2 && f.maintenance4 &&
      f.pipeline_creation_cache_control && f.vulkan_memory_model;
  // Replicated composites are
  // optional for models that do not use them.
}

runtime::Result<vulkan_shared::SharedVulkanRuntimeConfig>
parse_vgf_shared_device_config(const runtime::BackendInitContext& context);

// Public pre-registration seam: create a VGF-capable device BEFORE loading a
// mixed method, then use lookup_only for BOTH delegates. This removes
// dependence on the order in which ExecuTorch initializes its serialized
// delegate table.
runtime::Result<vulkan_shared::SharedVulkanDevicePtr> resolve_vgf_shared_device(
    const vulkan_shared::SharedVulkanRuntimeConfig& config);

runtime::Result<std::shared_ptr<VgfSharedDeviceState>>
resolve_vgf_shared_device_state(const runtime::BackendInitContext& context);

// Existing VGF device creation, with authoritative metadata and an optional
// instance lifetime owner. A null command_pool skips per-delegate pool
// creation.
VkResult vkml_allocate_basics(
    VkInstance* instance,
    VkPhysicalDevice* physical_device,
    VkDevice* device,
    VkQueue* queue,
    VkCommandPool* command_pool,
    uint32_t* queue_family_index,
    bool request_neural_statistics,
    bool* neural_statistics_device_enabled,
    bool request_host_memory_import,
    VgfHostMemoryImportCapabilities* host_memory_import_capabilities,
    std::shared_ptr<void>* instance_lifetime,
    vulkan_shared::SharedVulkanDeviceCreateInfo* shared_info);

} // namespace vgf
} // namespace backends
} // namespace executorch
