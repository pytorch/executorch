/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY clang-diagnostic-missing-field-initializers

#include <executorch/backends/vulkan/runtime/vk_api/Device.h>

#include <executorch/backends/vulkan/runtime/vk_api/Exception.h>

#include <bitset>
#include <cstring>

namespace vkcompute {
namespace vkapi {

PhysicalDevice::PhysicalDevice(VkPhysicalDevice physical_device_handle)
    : handle(physical_device_handle),
      properties{},
      memory_properties{},
#ifdef VK_KHR_16bit_storage
      shader_16bit_storage{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES},
#endif /* VK_KHR_16bit_storage */
#ifdef VK_KHR_8bit_storage
      shader_8bit_storage{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES},
#endif /* VK_KHR_8bit_storage */
#ifdef VK_KHR_shader_float16_int8
      shader_float16_int8_types{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR},
#endif /* VK_KHR_shader_float16_int8 */
      extension_features{nullptr},
      queue_families{},
      num_compute_queues(0),
      supports_int16_shader_types(false),
      has_unified_memory(false),
      has_timestamps(properties.limits.timestampComputeAndGraphics),
      timestamp_period(properties.limits.timestampPeriod) {
  // Extract physical device properties
  vkGetPhysicalDeviceProperties(handle, &properties);
  vkGetPhysicalDeviceMemoryProperties(handle, &memory_properties);

  VkPhysicalDeviceFeatures2 features2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

  // Create linked list to query availability of extensions

#ifdef VK_KHR_16bit_storage
  extension_features = &shader_16bit_storage;
  features2.pNext = &shader_16bit_storage;
#elif defined(VK_KHR_8bit_storage)
  extension_features = &shader_8bit_storage;
  features2.pNext = &shader_8bit_storage;
#elif defined(VK_KHR_shader_float16_int8)
  extension_features = &shader_float16_int8_types;
  features2.pNext = &shader_float16_int8_types;
#endif /* VK_KHR_16bit_storage */

#if defined(VK_KHR_16bit_storage) && defined(VK_KHR_8bit_storage)
  shader_16bit_storage.pNext = &shader_8bit_storage;
#elif defined(VK_KHR_16bit_storage) && defined(VK_KHR_shader_float16_int8)
  shader_16bit_storage.pNext = &shader_float16_int8_types;
#elif defined(VK_KHR_16bit_storage)
  shader_16bit_storage.pNext = nullptr;
#endif

#if defined(VK_KHR_8bit_storage) && defined(VK_KHR_shader_float16_int8)
  shader_8bit_storage.pNext = &shader_float16_int8_types;
#elif defined(VK_KHR_8bit_storage)
  shader_8bit_storage.pNext = nullptr;
#endif

#ifdef VK_KHR_shader_float16_int8
  shader_float16_int8_types.pNext = nullptr;
#endif

  vkGetPhysicalDeviceFeatures2(handle, &features2);

  if (features2.features.shaderInt16 == VK_TRUE) {
    supports_int16_shader_types = true;
  }

  // Check if there are any memory types have both the HOST_VISIBLE and the
  // DEVICE_LOCAL property flags
  const VkMemoryPropertyFlags unified_memory_flags =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  for (size_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    if (memory_properties.memoryTypes[i].propertyFlags | unified_memory_flags) {
      has_unified_memory = true;
      break;
    }
  }

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      handle, &queue_family_count, nullptr);

  queue_families.resize(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      handle, &queue_family_count, queue_families.data());

  // Find the total number of compute queues
  for (const VkQueueFamilyProperties& p : queue_families) {
    // Check if this family has compute capability
    if (p.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      num_compute_queues += p.queueCount;
    }
  }
}

//
// DeviceHandle
//

DeviceHandle::DeviceHandle(VkDevice device) : handle(device) {}

DeviceHandle::~DeviceHandle() {
  if (handle == VK_NULL_HANDLE) {
    return;
  }
  vkDestroyDevice(handle, nullptr);
}

//
// Utils
//

void find_requested_device_extensions(
    VkPhysicalDevice physical_device,
    std::vector<const char*>& enabled_extensions,
    const std::vector<const char*>& requested_extensions) {
  uint32_t device_extension_properties_count = 0;
  VK_CHECK(vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &device_extension_properties_count, nullptr));
  std::vector<VkExtensionProperties> device_extension_properties(
      device_extension_properties_count);
  VK_CHECK(vkEnumerateDeviceExtensionProperties(
      physical_device,
      nullptr,
      &device_extension_properties_count,
      device_extension_properties.data()));

  std::vector<const char*> enabled_device_extensions;

  for (const auto& requested_extension : requested_extensions) {
    for (const auto& extension : device_extension_properties) {
      if (strcmp(requested_extension, extension.extensionName) == 0) {
        enabled_extensions.push_back(requested_extension);
        break;
      }
    }
  }
}

std::string get_device_type_str(const VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return "INTEGRATED_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return "DISCRETE_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return "VIRTUAL_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return "CPU";
    default:
      return "UNKNOWN";
  }
}

std::string get_memory_properties_str(const VkMemoryPropertyFlags flags) {
  std::bitset<10> values(flags);
  std::stringstream ss("|");
  if (values[0]) {
    ss << " DEVICE_LOCAL |";
  }
  if (values[1]) {
    ss << " HOST_VISIBLE |";
  }
  if (values[2]) {
    ss << " HOST_COHERENT |";
  }
  if (values[3]) {
    ss << " HOST_CACHED |";
  }
  if (values[4]) {
    ss << " LAZILY_ALLOCATED |";
  }

  return ss.str();
}

std::string get_queue_family_properties_str(const VkQueueFlags flags) {
  std::bitset<10> values(flags);
  std::stringstream ss("|");
  if (values[0]) {
    ss << " GRAPHICS |";
  }
  if (values[1]) {
    ss << " COMPUTE |";
  }
  if (values[2]) {
    ss << " TRANSFER |";
  }

  return ss.str();
}

} // namespace vkapi
} // namespace vkcompute
