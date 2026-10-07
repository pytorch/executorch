/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

namespace executorch {
namespace backends {
namespace vgf {

// Snapshot of VK_EXT_external_memory_host state for the exact Vulkan device
// used by VGF. Physical-device advertisement and logical-device enablement are
// deliberately separate because Vulkan does not provide a post-creation query
// for the list of extensions enabled at vkCreateDevice().
struct VgfHostMemoryImportCapabilities {
  bool physical_device_advertised = false;
  bool logical_device_enabled = false;
  VkDeviceSize min_imported_host_pointer_alignment = 0;
};

inline bool vgf_host_memory_import_should_be_enabled(
    bool requested,
    bool physical_device_advertised) {
  return requested && physical_device_advertised;
}

inline VkCommandPoolCreateFlags vgf_command_pool_flags(
    bool host_memory_import_enabled) {
  return host_memory_import_enabled
      ? VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
      : 0;
}

inline VkPhysicalDeviceDataGraphFeaturesARM make_vgf_data_graph_features(
    void* p_next) {
  VkPhysicalDeviceDataGraphFeaturesARM features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM;
  features.pNext = p_next;
  features.dataGraph = VK_TRUE;
  features.dataGraphShaderModule = VK_TRUE;
  return features;
}

inline bool vgf_data_graph_features_supported(
    const VkPhysicalDeviceDataGraphFeaturesARM& features) {
  return features.dataGraph == VK_TRUE &&
      features.dataGraphShaderModule == VK_TRUE;
}

} // namespace vgf
} // namespace backends
} // namespace executorch
