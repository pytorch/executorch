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
