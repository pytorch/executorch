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

#include <algorithm>
#include <bitset>
#include <cctype>
#include <cstring>
#include <sstream>

namespace vkcompute {
namespace vkapi {

PhysicalDevice::PhysicalDevice(
    VkInstance instance_handle,
    VkPhysicalDevice physical_device_handle)
    : instance(instance_handle),
      handle(physical_device_handle),
      properties{},
      memory_properties{},
#ifdef VK_KHR_16bit_storage
      shader_16bit_storage{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
          nullptr},
#endif /* VK_KHR_16bit_storage */
#ifdef VK_KHR_8bit_storage
      shader_8bit_storage{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
          nullptr},
#endif /* VK_KHR_8bit_storage */
#ifdef VK_KHR_shader_float16_int8
      shader_float16_int8_types{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
          nullptr},
#endif /* VK_KHR_shader_float16_int8 */
#ifdef VK_KHR_shader_integer_dot_product
      shader_int_dot_product_features{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
          nullptr},
      shader_int_dot_product_properties{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR,
          nullptr},
#endif
      queue_families{},
      num_compute_queues(0),
      api_version_major(0),
      api_version_minor(0),
      supports_int16_shader_types(false),
      supports_int64_shader_types(false),
      supports_float64_shader_types(false),
      has_unified_memory(false),
      has_timestamps(false),
      timestamp_period(0),
      min_ubo_alignment(0),
      device_name{},
      device_type{DeviceType::UNKNOWN} {
  // Extract physical device properties
  vkGetPhysicalDeviceProperties(handle, &properties);

  api_version_major = VK_VERSION_MAJOR(properties.apiVersion);
  api_version_minor = VK_VERSION_MINOR(properties.apiVersion);

  // Extract fields of interest
  has_timestamps = properties.limits.timestampComputeAndGraphics;
  timestamp_period = properties.limits.timestampPeriod;
  min_ubo_alignment = properties.limits.minUniformBufferOffsetAlignment;

  vkGetPhysicalDeviceMemoryProperties(handle, &memory_properties);

  if (properties.apiVersion >= VK_API_VERSION_1_1) {
    query_extensions_vk_1_1();
  } else {
    query_extensions_vk_1_0();
  }

  // Check if there are any memory types have both the HOST_VISIBLE and the
  // DEVICE_LOCAL property flags
  const VkMemoryPropertyFlags unified_memory_flags =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
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

  // Obtain device identity metadata
  device_name = std::string(properties.deviceName);
  std::transform(
      device_name.begin(),
      device_name.end(),
      device_name.begin(),
      [](unsigned char c) { return std::tolower(c); });

  if (device_name.find("adreno") != std::string::npos) {
    device_type = DeviceType::ADRENO;
  } else if (device_name.find("swiftshader") != std::string::npos) {
    device_type = DeviceType::SWIFTSHADER;
  } else if (device_name.find("nvidia") != std::string::npos) {
    device_type = DeviceType::NVIDIA;
  } else if (device_name.find("mali") != std::string::npos) {
    device_type = DeviceType::MALI;
  }
}

void PhysicalDevice::query_extensions_vk_1_0() {
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(handle, &features);

  if (features.shaderInt16 == VK_TRUE) {
    supports_int16_shader_types = true;
  }
  if (features.shaderInt64 == VK_TRUE) {
    supports_int64_shader_types = true;
  }
  if (features.shaderFloat64 == VK_TRUE) {
    supports_float64_shader_types = true;
  }

  // Try to query extension features using
  // VK_KHR_get_physical_device_properties2
  auto vkGetPhysicalDeviceFeatures2KHR =
      (PFN_vkGetPhysicalDeviceFeatures2KHR)vkGetInstanceProcAddr(
          instance, "vkGetPhysicalDeviceFeatures2KHR");

  if (vkGetPhysicalDeviceFeatures2KHR == nullptr) {
    // Manually set extension features to false if the function is not available
#ifdef VK_KHR_16bit_storage
    shader_16bit_storage.storageBuffer16BitAccess = VK_FALSE;
    shader_16bit_storage.uniformAndStorageBuffer16BitAccess = VK_FALSE;
    shader_16bit_storage.storagePushConstant16 = VK_FALSE;
    shader_16bit_storage.storageInputOutput16 = VK_FALSE;
#endif /* VK_KHR_16bit_storage */
#ifdef VK_KHR_8bit_storage
    shader_8bit_storage.storageBuffer8BitAccess = VK_FALSE;
    shader_8bit_storage.uniformAndStorageBuffer8BitAccess = VK_FALSE;
    shader_8bit_storage.storagePushConstant8 = VK_FALSE;
#endif /* VK_KHR_8bit_storage */
#ifdef VK_KHR_shader_float16_int8
    shader_float16_int8_types.shaderFloat16 = VK_FALSE;
    shader_float16_int8_types.shaderInt8 = VK_FALSE;
#endif /* VK_KHR_shader_float16_int8 */
#ifdef VK_KHR_shader_integer_dot_product
    shader_int_dot_product_features.shaderIntegerDotProduct = VK_FALSE;
#endif /* VK_KHR_shader_integer_dot_product */
    return;
  }

  VkPhysicalDeviceFeatures2KHR features2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR};

  void* extension_list_top = nullptr;

#ifdef VK_KHR_16bit_storage
  shader_16bit_storage.pNext = extension_list_top;
  extension_list_top = &shader_16bit_storage;
#endif /* VK_KHR_16bit_storage */

#ifdef VK_KHR_8bit_storage
  shader_8bit_storage.pNext = extension_list_top;
  extension_list_top = &shader_8bit_storage;
#endif /* VK_KHR_8bit_storage */

#ifdef VK_KHR_shader_float16_int8
  shader_float16_int8_types.pNext = extension_list_top;
  extension_list_top = &shader_float16_int8_types;
#endif /* VK_KHR_shader_float16_int8 */

#ifdef VK_KHR_shader_integer_dot_product
  shader_int_dot_product_features.pNext = extension_list_top;
  extension_list_top = &shader_int_dot_product_features;
#endif /* VK_KHR_shader_integer_dot_product */

  features2.pNext = extension_list_top;

  vkGetPhysicalDeviceFeatures2KHR(handle, &features2);

  // Query properties separately from features
  auto vkGetPhysicalDeviceProperties2KHR =
      (PFN_vkGetPhysicalDeviceProperties2KHR)vkGetInstanceProcAddr(
          instance, "vkGetPhysicalDeviceProperties2KHR");

  if (vkGetPhysicalDeviceProperties2KHR != nullptr) {
    VkPhysicalDeviceProperties2KHR properties2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR};

#ifdef VK_KHR_shader_integer_dot_product
    shader_int_dot_product_properties.pNext = nullptr;
    properties2.pNext = &shader_int_dot_product_properties;
#endif /* VK_KHR_shader_integer_dot_product */

    vkGetPhysicalDeviceProperties2KHR(handle, &properties2);
  }
}

void PhysicalDevice::query_extensions_vk_1_1() {
  VkPhysicalDeviceFeatures2 features2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

  // Create linked list to query availability of extensions
  void* extension_list_top = nullptr;

#ifdef VK_KHR_16bit_storage
  shader_16bit_storage.pNext = extension_list_top;
  extension_list_top = &shader_16bit_storage;
#endif /* VK_KHR_16bit_storage */

#ifdef VK_KHR_8bit_storage
  shader_8bit_storage.pNext = extension_list_top;
  extension_list_top = &shader_8bit_storage;
#endif /* VK_KHR_8bit_storage */

#ifdef VK_KHR_shader_float16_int8
  shader_float16_int8_types.pNext = extension_list_top;
  extension_list_top = &shader_float16_int8_types;
#endif /* VK_KHR_shader_float16_int8 */

#ifdef VK_KHR_shader_integer_dot_product
  shader_int_dot_product_features.pNext = extension_list_top;
  extension_list_top = &shader_int_dot_product_features;
#endif /* VK_KHR_shader_integer_dot_product */

  features2.pNext = extension_list_top;

  vkGetPhysicalDeviceFeatures2(handle, &features2);

  if (features2.features.shaderInt16 == VK_TRUE) {
    supports_int16_shader_types = true;
  }
  if (features2.features.shaderInt64 == VK_TRUE) {
    supports_int64_shader_types = true;
  }
  if (features2.features.shaderFloat64 == VK_TRUE) {
    supports_float64_shader_types = true;
  }

  // Query properties separately from features
  VkPhysicalDeviceProperties2 properties2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};

#ifdef VK_KHR_shader_integer_dot_product
  shader_int_dot_product_properties.pNext = nullptr;
  properties2.pNext = &shader_int_dot_product_properties;
#endif /* VK_KHR_shader_integer_dot_product */

  vkGetPhysicalDeviceProperties2(handle, &properties2);
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
