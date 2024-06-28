/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY clang-diagnostic-missing-field-initializers

#include <executorch/backends/vulkan/runtime/api/Adapter.h>

#include <bitset>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <utility>

namespace vkcompute {
namespace api {

PhysicalDevice::PhysicalDevice(VkPhysicalDevice physical_device_handle)
    : handle(physical_device_handle),
      properties{},
      memory_properties{},
      shader_16bit_storage{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES},
      shader_8bit_storage{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES},
      shader_float16_int8_types{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR},
      queue_families{},
      num_compute_queues(0),
      has_unified_memory(false),
      has_timestamps(properties.limits.timestampComputeAndGraphics),
      timestamp_period(properties.limits.timestampPeriod),
      extension_features(&shader_16bit_storage) {
  // Extract physical device properties
  vkGetPhysicalDeviceProperties(handle, &properties);
  vkGetPhysicalDeviceMemoryProperties(handle, &memory_properties);

  VkPhysicalDeviceFeatures2 features2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

  // Create linked list to query availability of extensions
  features2.pNext = &shader_16bit_storage;
  shader_16bit_storage.pNext = &shader_8bit_storage;
  shader_8bit_storage.pNext = &shader_float16_int8_types;
  shader_float16_int8_types.pNext = nullptr;

  vkGetPhysicalDeviceFeatures2(handle, &features2);

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

namespace {

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

VkDevice create_logical_device(
    const PhysicalDevice& physical_device,
    const uint32_t num_queues_to_create,
    std::vector<Adapter::Queue>& queues,
    std::vector<uint32_t>& queue_usage) {
  // Find compute queues up to the requested number of queues

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  queue_create_infos.reserve(num_queues_to_create);

  std::vector<std::pair<uint32_t, uint32_t>> queues_to_get;
  queues_to_get.reserve(num_queues_to_create);

  uint32_t remaining_queues = num_queues_to_create;
  for (uint32_t family_i = 0; family_i < physical_device.queue_families.size();
       ++family_i) {
    const VkQueueFamilyProperties& queue_properties =
        physical_device.queue_families.at(family_i);
    // Check if this family has compute capability
    if (queue_properties.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      const uint32_t queues_to_init =
          std::min(remaining_queues, queue_properties.queueCount);

      const std::vector<float> queue_priorities(queues_to_init, 1.0f);
      queue_create_infos.push_back({
          VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, // sType
          nullptr, // pNext
          0u, // flags
          family_i, // queueFamilyIndex
          queues_to_init, // queueCount
          queue_priorities.data(), // pQueuePriorities
      });

      for (size_t queue_i = 0; queue_i < queues_to_init; ++queue_i) {
        // Use this to get the queue handle once device is created
        queues_to_get.emplace_back(family_i, queue_i);
      }
      remaining_queues -= queues_to_init;
    }
    if (remaining_queues == 0) {
      break;
    }
  }

  queues.reserve(queues_to_get.size());
  queue_usage.reserve(queues_to_get.size());

  // Create the VkDevice

  std::vector<const char*> requested_device_extensions{
#ifdef VK_KHR_portability_subset
      VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
#endif /* VK_KHR_portability_subset */
      VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
      VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
      VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
  };

  std::vector<const char*> enabled_device_extensions;
  find_requested_device_extensions(
      physical_device.handle,
      enabled_device_extensions,
      requested_device_extensions);

  VkDeviceCreateInfo device_create_info{
      VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      static_cast<uint32_t>(queue_create_infos.size()), // queueCreateInfoCount
      queue_create_infos.data(), // pQueueCreateInfos
      0u, // enabledLayerCount
      nullptr, // ppEnabledLayerNames
      static_cast<uint32_t>(
          enabled_device_extensions.size()), // enabledExtensionCount
      enabled_device_extensions.data(), // ppEnabledExtensionNames
      nullptr, // pEnabledFeatures
  };

  device_create_info.pNext = physical_device.extension_features;

  VkDevice handle = nullptr;
  VK_CHECK(vkCreateDevice(
      physical_device.handle, &device_create_info, nullptr, &handle));

#ifdef USE_VULKAN_VOLK
  volkLoadDevice(handle);
#endif /* USE_VULKAN_VOLK */

  // Obtain handles for the created queues and initialize queue usage heuristic

  for (const std::pair<uint32_t, uint32_t>& queue_idx : queues_to_get) {
    VkQueue queue_handle = VK_NULL_HANDLE;
    VkQueueFlags flags =
        physical_device.queue_families.at(queue_idx.first).queueFlags;
    vkGetDeviceQueue(handle, queue_idx.first, queue_idx.second, &queue_handle);
    queues.push_back({queue_idx.first, queue_idx.second, flags, queue_handle});
    // Initial usage value
    queue_usage.push_back(0);
  }

  return handle;
}

// Print utils

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

} // namespace

//
// DeviceHandle
//

DeviceHandle::DeviceHandle(VkDevice device) : handle_(device) {}

DeviceHandle::DeviceHandle(DeviceHandle&& other) noexcept
    : handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

DeviceHandle::~DeviceHandle() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyDevice(handle_, nullptr);
}

//
// Adapter
//

Adapter::Adapter(
    VkInstance instance,
    PhysicalDevice physical_device,
    const uint32_t num_queues,
    const std::string& cache_data_path)
    : queue_usage_mutex_{},
      physical_device_(std::move(physical_device)),
      queues_{},
      queue_usage_{},
      queue_mutexes_{},
      instance_(instance),
      device_(create_logical_device(
          physical_device_,
          num_queues,
          queues_,
          queue_usage_)),
      shader_layout_cache_(device_.handle_),
      shader_cache_(device_.handle_),
      pipeline_layout_cache_(device_.handle_),
      compute_pipeline_cache_(device_.handle_, cache_data_path),
      sampler_cache_(device_.handle_),
      vma_(instance_, physical_device_.handle, device_.handle_) {}

Adapter::Queue Adapter::request_queue() {
  // Lock the mutex as multiple threads can request a queue at the same time
  std::lock_guard<std::mutex> lock(queue_usage_mutex_);

  uint32_t min_usage = UINT32_MAX;
  uint32_t min_used_i = 0;
  for (size_t i = 0; i < queues_.size(); ++i) {
    if (queue_usage_[i] < min_usage) {
      min_used_i = i;
      min_usage = queue_usage_[i];
    }
  }
  queue_usage_[min_used_i] += 1;

  return queues_[min_used_i];
}

void Adapter::return_queue(Adapter::Queue& compute_queue) {
  for (size_t i = 0; i < queues_.size(); ++i) {
    if ((queues_[i].family_index == compute_queue.family_index) &&
        (queues_[i].queue_index == compute_queue.queue_index)) {
      std::lock_guard<std::mutex> lock(queue_usage_mutex_);
      queue_usage_[i] -= 1;
      break;
    }
  }
}

void Adapter::submit_cmd(
    const Adapter::Queue& device_queue,
    VkCommandBuffer cmd,
    VkFence fence) {
  const VkSubmitInfo submit_info{
      VK_STRUCTURE_TYPE_SUBMIT_INFO, // sType
      nullptr, // pNext
      0u, // waitSemaphoreCount
      nullptr, // pWaitSemaphores
      nullptr, // pWaitDstStageMask
      1u, // commandBufferCount
      &cmd, // pCommandBuffers
      0u, // signalSemaphoreCount
      nullptr, // pSignalSemaphores
  };

  std::lock_guard<std::mutex> queue_lock(
      queue_mutexes_[device_queue.queue_index % NUM_QUEUE_MUTEXES]);

  VK_CHECK(vkQueueSubmit(device_queue.handle, 1u, &submit_info, fence));
}

void Adapter::submit_cmds(
    const Adapter::Queue& device_queue,
    const std::vector<VkCommandBuffer>& cmds,
    VkFence fence) {
  const VkSubmitInfo submit_info{
      VK_STRUCTURE_TYPE_SUBMIT_INFO, // sType
      nullptr, // pNext
      0u, // waitSemaphoreCount
      nullptr, // pWaitSemaphores
      nullptr, // pWaitDstStageMask
      utils::safe_downcast<uint32_t>(cmds.size()), // commandBufferCount
      cmds.data(), // pCommandBuffers
      0u, // signalSemaphoreCount
      nullptr, // pSignalSemaphores
  };

  VK_CHECK(vkQueueSubmit(device_queue.handle, 1u, &submit_info, fence));
}

std::string Adapter::stringize() const {
  std::stringstream ss;

  VkPhysicalDeviceProperties properties = physical_device_.properties;
  uint32_t v_major = VK_VERSION_MAJOR(properties.apiVersion);
  uint32_t v_minor = VK_VERSION_MINOR(properties.apiVersion);
  std::string device_type = get_device_type_str(properties.deviceType);
  VkPhysicalDeviceLimits limits = properties.limits;

  ss << "{" << std::endl;
  ss << "  Physical Device Info {" << std::endl;
  ss << "    apiVersion:    " << v_major << "." << v_minor << std::endl;
  ss << "    driverversion: " << properties.driverVersion << std::endl;
  ss << "    deviceType:    " << device_type << std::endl;
  ss << "    deviceName:    " << properties.deviceName << std::endl;

#define PRINT_PROP(struct, name)                                       \
  ss << "      " << std::left << std::setw(36) << #name << struct.name \
     << std::endl;

#define PRINT_PROP_VEC3(struct, name)                                     \
  ss << "      " << std::left << std::setw(36) << #name << struct.name[0] \
     << "," << struct.name[1] << "," << struct.name[2] << std::endl;

  ss << "    Physical Device Limits {" << std::endl;
  PRINT_PROP(limits, maxImageDimension1D);
  PRINT_PROP(limits, maxImageDimension2D);
  PRINT_PROP(limits, maxImageDimension3D);
  PRINT_PROP(limits, maxTexelBufferElements);
  PRINT_PROP(limits, maxPushConstantsSize);
  PRINT_PROP(limits, maxMemoryAllocationCount);
  PRINT_PROP(limits, maxSamplerAllocationCount);
  PRINT_PROP(limits, maxComputeSharedMemorySize);
  PRINT_PROP_VEC3(limits, maxComputeWorkGroupCount);
  PRINT_PROP(limits, maxComputeWorkGroupInvocations);
  PRINT_PROP_VEC3(limits, maxComputeWorkGroupSize);
  ss << "    }" << std::endl;

  ss << "    16bit Storage Features {" << std::endl;
  PRINT_PROP(physical_device_.shader_16bit_storage, storageBuffer16BitAccess);
  PRINT_PROP(
      physical_device_.shader_16bit_storage,
      uniformAndStorageBuffer16BitAccess);
  PRINT_PROP(physical_device_.shader_16bit_storage, storagePushConstant16);
  PRINT_PROP(physical_device_.shader_16bit_storage, storageInputOutput16);
  ss << "    }" << std::endl;

  ss << "    8bit Storage Features {" << std::endl;
  PRINT_PROP(physical_device_.shader_8bit_storage, storageBuffer8BitAccess);
  PRINT_PROP(
      physical_device_.shader_8bit_storage, uniformAndStorageBuffer8BitAccess);
  PRINT_PROP(physical_device_.shader_8bit_storage, storagePushConstant8);
  ss << "    }" << std::endl;

  ss << "    Shader 16bit and 8bit Features {" << std::endl;
  PRINT_PROP(physical_device_.shader_float16_int8_types, shaderFloat16);
  PRINT_PROP(physical_device_.shader_float16_int8_types, shaderInt8);
  ss << "    }" << std::endl;

  const VkPhysicalDeviceMemoryProperties& mem_props =
      physical_device_.memory_properties;

  ss << "  }" << std::endl;
  ss << "  Memory Info {" << std::endl;
  ss << "    Memory Types [" << std::endl;
  for (size_t i = 0; i < mem_props.memoryTypeCount; ++i) {
    ss << "      " << " [Heap " << mem_props.memoryTypes[i].heapIndex << "] "
       << get_memory_properties_str(mem_props.memoryTypes[i].propertyFlags)
       << std::endl;
  }
  ss << "    ]" << std::endl;
  ss << "    Memory Heaps [" << std::endl;
  for (size_t i = 0; i < mem_props.memoryHeapCount; ++i) {
    ss << "      " << mem_props.memoryHeaps[i].size << std::endl;
  }
  ss << "    ]" << std::endl;
  ss << "  }" << std::endl;

  ss << "  Queue Families {" << std::endl;
  for (const VkQueueFamilyProperties& queue_family_props :
       physical_device_.queue_families) {
    ss << "    (" << queue_family_props.queueCount << " Queues) "
       << get_queue_family_properties_str(queue_family_props.queueFlags)
       << std::endl;
  }
  ss << "  }" << std::endl;
  ss << "  VkDevice: " << device_.handle_ << std::endl;
  ss << "  Compute Queues [" << std::endl;
  for (const Adapter::Queue& compute_queue : queues_) {
    ss << "    Family " << compute_queue.family_index << ", Queue "
       << compute_queue.queue_index << ": " << compute_queue.handle
       << std::endl;
    ;
  }
  ss << "  ]" << std::endl;
  ss << "}";

#undef PRINT_PROP
#undef PRINT_PROP_VEC3

  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Adapter& adapter) {
  os << adapter.stringize() << std::endl;
  return os;
}

} // namespace api
} // namespace vkcompute
