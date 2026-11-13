/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Logical-device state derived from what a VkDevice was created with (its
// enabled extensions and SharedVulkanDeviceFeatures). An Adapter that creates
// the device and one that borrows it through a shared device derive it the
// same way; an Adapter on an external VkDevice knows neither and stays
// conservative.

#include <executorch/backends/vulkan_shared/runtime/SharedVulkanDevice.h>

#include <algorithm>
#include <string>
#include <vector>

namespace vkcompute {
namespace vkapi {

// Pipelines with a required subgroup size can be created with
// REQUIRE_FULL_SUBGROUPS.
inline bool full_subgroups_enabled(
    const executorch::backends::vulkan_shared::SharedVulkanDeviceFeatures& f) {
  return f.subgroup_size_control && f.compute_full_subgroups;
}

// Shaders that declare the VulkanMemoryModel capability can run: the
// extension and its feature are enabled.
inline bool vulkan_memory_model_enabled(
    const std::vector<std::string>& extensions,
    const executorch::backends::vulkan_shared::SharedVulkanDeviceFeatures& f) {
#ifdef VK_KHR_vulkan_memory_model
  return f.vulkan_memory_model &&
      std::find(
          extensions.begin(),
          extensions.end(),
          VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME) != extensions.end();
#else
  (void)extensions;
  (void)f;
  return false;
#endif /* VK_KHR_vulkan_memory_model */
}

} // namespace vkapi
} // namespace vkcompute
