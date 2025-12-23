/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY clang-diagnostic-missing-field-initializers

#include <executorch/backends/vulkan/runtime/vk_api/Adapter.h>

#include <iomanip>
#include <sstream>

namespace vkcompute {
namespace vkapi {

namespace {

void find_compute_queues(
    const PhysicalDevice& physical_device,
    const uint32_t num_queues_to_create,
    std::vector<VkDeviceQueueCreateInfo>& queue_create_infos,
    std::vector<std::pair<uint32_t, uint32_t>>& queues_to_get,
    std::vector<std::vector<float>>& queue_priorities) {
  queue_create_infos.reserve(num_queues_to_create);
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

      queue_priorities.emplace_back(queues_to_init, 1.0f);
      queue_create_infos.push_back({
          VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, // sType
          nullptr, // pNext
          0u, // flags
          family_i, // queueFamilyIndex
          queues_to_init, // queueCount
          queue_priorities.back().data(), // pQueuePriorities
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
}

void populate_queue_info(
    const PhysicalDevice& physical_device,
    VkDevice logical_device,
    const std::vector<std::pair<uint32_t, uint32_t>>& queues_to_get,
    std::vector<Adapter::Queue>& queues,
    std::vector<uint32_t>& queue_usage) {
  queues.reserve(queues_to_get.size());
  queue_usage.reserve(queues_to_get.size());

  // Obtain handles for the created queues and initialize queue usage heuristic

  for (const std::pair<uint32_t, uint32_t>& queue_idx : queues_to_get) {
    VkQueue queue_handle = VK_NULL_HANDLE;
    VkQueueFlags flags =
        physical_device.queue_families.at(queue_idx.first).queueFlags;
    vkGetDeviceQueue(
        logical_device, queue_idx.first, queue_idx.second, &queue_handle);
    queues.push_back({queue_idx.first, queue_idx.second, flags, queue_handle});
    // Initial usage value
    queue_usage.push_back(0);
  }
}

VkDevice create_logical_device(
    const PhysicalDevice& physical_device,
    const uint32_t num_queues_to_create,
    std::vector<Adapter::Queue>& queues,
    std::vector<uint32_t>& queue_usage) {
  // Find compute queues up to the requested number of queues

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::vector<std::pair<uint32_t, uint32_t>> queues_to_get;
  std::vector<std::vector<float>> queue_priorities;
  find_compute_queues(
      physical_device,
      num_queues_to_create,
      queue_create_infos,
      queues_to_get,
      queue_priorities);

  // Create the VkDevice
  std::vector<const char*> requested_device_extensions{
#ifdef VK_KHR_portability_subset
      VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
#endif /* VK_KHR_portability_subset */
#ifdef VK_ANDROID_external_memory_android_hardware_buffer
      VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
#endif /* VK_ANDROID_external_memory_android_hardware_buffer */
#ifdef VK_KHR_16bit_storage
      VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
#endif /* VK_KHR_16bit_storage */
#ifdef VK_KHR_8bit_storage
      VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
#endif /* VK_KHR_8bit_storage */
#ifdef VK_KHR_shader_float16_int8
      VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
#endif /* VK_KHR_shader_float16_int8 */
#ifdef VK_KHR_shader_integer_dot_product
      VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME,
#endif /* VK_KHR_shader_integer_dot_product */
#if defined(VK_KHR_pipeline_executable_properties) && \
    defined(ETVK_INSPECT_PIPELINES)
      VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME,
#endif /* VK_KHR_pipeline_executable_properties && ETVK_INSPECT_PIPELINES */
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

  void* extension_list_top = nullptr;

#ifdef VK_KHR_16bit_storage
  VkPhysicalDevice16BitStorageFeatures shader_16bit_storage{
      physical_device.shader_16bit_storage};

  shader_16bit_storage.pNext = extension_list_top;
  extension_list_top = &shader_16bit_storage;
#endif /* VK_KHR_16bit_storage */

#ifdef VK_KHR_8bit_storage
  VkPhysicalDevice8BitStorageFeatures shader_8bit_storage{
      physical_device.shader_8bit_storage};

  shader_8bit_storage.pNext = extension_list_top;
  extension_list_top = &shader_8bit_storage;
#endif /* VK_KHR_8bit_storage */

#ifdef VK_KHR_shader_float16_int8
  VkPhysicalDeviceShaderFloat16Int8Features shader_float16_int8_types{
      physical_device.shader_float16_int8_types};

  shader_float16_int8_types.pNext = extension_list_top;
  extension_list_top = &shader_float16_int8_types;
#endif /* VK_KHR_shader_float16_int8 */

#ifdef VK_KHR_shader_integer_dot_product
  VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR
      shader_int_dot_product_features{
          physical_device.shader_int_dot_product_features};
  shader_int_dot_product_features.pNext = extension_list_top;
  extension_list_top = &shader_int_dot_product_features;
#endif /* VK_KHR_shader_integer_dot_product */

  device_create_info.pNext = extension_list_top;

  VkDevice handle = nullptr;
  VK_CHECK(vkCreateDevice(
      physical_device.handle, &device_create_info, nullptr, &handle));

#ifdef USE_VULKAN_VOLK
  volkLoadDevice(handle);
#endif /* USE_VULKAN_VOLK */

  populate_queue_info(
      physical_device, handle, queues_to_get, queues, queue_usage);

  return handle;
}

bool test_linear_tiling_3d_image_support(VkDevice device) {
  // Test creating a 3D image with linear tiling to see if it is supported.
  // According to the Vulkan spec, linear tiling may not be supported for 3D
  // images.
  VkExtent3D image_extents{1u, 1u, 1u};
  const VkImageCreateInfo image_create_info{
      VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      VK_IMAGE_TYPE_3D, // imageType
      VK_FORMAT_R32G32B32A32_SFLOAT, // format
      image_extents, // extents
      1u, // mipLevels
      1u, // arrayLayers
      VK_SAMPLE_COUNT_1_BIT, // samples
      VK_IMAGE_TILING_LINEAR, // tiling
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
      VK_IMAGE_LAYOUT_UNDEFINED, // initialLayout
  };
  VkImage image = VK_NULL_HANDLE;
  VkResult res = vkCreateImage(device, &image_create_info, nullptr, &image);

  if (res == VK_SUCCESS) {
    vkDestroyImage(device, image, nullptr);
  }

  return res == VK_SUCCESS;
}

} // namespace

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
      shader_layout_cache_(device_.handle),
      shader_cache_(device_.handle),
      pipeline_layout_cache_(device_.handle),
      compute_pipeline_cache_(device_.handle, cache_data_path),
      sampler_cache_(device_.handle),
      vma_(instance_, physical_device_.handle, device_.handle),
      linear_tiling_3d_enabled_{
          test_linear_tiling_3d_image_support(device_.handle)},
      owns_device_{true} {}

Adapter::Adapter(
    VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice logical_device,
    const uint32_t num_queues,
    const std::string& cache_data_path)
    : queue_usage_mutex_{},
      physical_device_(physical_device),
      queues_{},
      queue_usage_{},
      queue_mutexes_{},
      instance_(instance),
      device_(logical_device),
      shader_layout_cache_(device_.handle),
      shader_cache_(device_.handle),
      pipeline_layout_cache_(device_.handle),
      compute_pipeline_cache_(device_.handle, cache_data_path),
      sampler_cache_(device_.handle),
      vma_(instance_, physical_device_.handle, device_.handle),
      linear_tiling_3d_enabled_{
          test_linear_tiling_3d_image_support(device_.handle)},
      owns_device_{false} {
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::vector<std::pair<uint32_t, uint32_t>> queues_to_get;
  std::vector<std::vector<float>> queue_priorities;
  find_compute_queues(
      physical_device_,
      num_queues,
      queue_create_infos,
      queues_to_get,
      queue_priorities);
  populate_queue_info(
      physical_device_, device_.handle, queues_to_get, queues_, queue_usage_);
}

Adapter::~Adapter() {
  if (!owns_device_) {
    device_.handle = VK_NULL_HANDLE;
  }
}

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
    VkFence fence,
    VkSemaphore wait_semaphore,
    VkSemaphore signal_semaphore) {
  const VkPipelineStageFlags flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  const bool set_wait_semaphore = wait_semaphore != VK_NULL_HANDLE;
  const bool set_signal_semaphore = signal_semaphore != VK_NULL_HANDLE;
  const VkSubmitInfo submit_info{
      VK_STRUCTURE_TYPE_SUBMIT_INFO, // sType
      nullptr, // pNext
      set_wait_semaphore ? 1u : 0u, // waitSemaphoreCount
      set_wait_semaphore ? &wait_semaphore : nullptr, // pWaitSemaphores
      &flags, // pWaitDstStageMask
      1u, // commandBufferCount
      &cmd, // pCommandBuffers
      set_signal_semaphore ? 1u : 0u, // signalSemaphoreCount
      set_signal_semaphore ? &signal_semaphore : nullptr, // pSignalSemaphores
  };

  std::lock_guard<std::mutex> queue_lock(
      queue_mutexes_[device_queue.queue_index % NUM_QUEUE_MUTEXES]);

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

#define PRINT_BOOL(value, name) \
  ss << "      " << std::left << std::setw(36) << #name << value << std::endl;

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
  PRINT_PROP(limits, maxStorageBufferRange);
  PRINT_PROP(limits, maxTexelBufferElements);
  PRINT_PROP(limits, maxPushConstantsSize);
  PRINT_PROP(limits, maxMemoryAllocationCount);
  PRINT_PROP(limits, maxSamplerAllocationCount);
  PRINT_PROP(limits, maxComputeSharedMemorySize);
  PRINT_PROP_VEC3(limits, maxComputeWorkGroupCount);
  PRINT_PROP(limits, maxComputeWorkGroupInvocations);
  PRINT_PROP_VEC3(limits, maxComputeWorkGroupSize);
  ss << "    }" << std::endl;

#ifdef VK_KHR_16bit_storage
  ss << "    16bit Storage Features {" << std::endl;
  PRINT_PROP(physical_device_.shader_16bit_storage, storageBuffer16BitAccess);
  PRINT_PROP(
      physical_device_.shader_16bit_storage,
      uniformAndStorageBuffer16BitAccess);
  PRINT_PROP(physical_device_.shader_16bit_storage, storagePushConstant16);
  PRINT_PROP(physical_device_.shader_16bit_storage, storageInputOutput16);
  ss << "    }" << std::endl;
#endif /* VK_KHR_16bit_storage */

#ifdef VK_KHR_8bit_storage
  ss << "    8bit Storage Features {" << std::endl;
  PRINT_PROP(physical_device_.shader_8bit_storage, storageBuffer8BitAccess);
  PRINT_PROP(
      physical_device_.shader_8bit_storage, uniformAndStorageBuffer8BitAccess);
  PRINT_PROP(physical_device_.shader_8bit_storage, storagePushConstant8);
  ss << "    }" << std::endl;
#endif /* VK_KHR_8bit_storage */

  ss << "    Shader 16bit and 8bit Features {" << std::endl;
  PRINT_BOOL(physical_device_.supports_int16_shader_types, shaderInt16)
#ifdef VK_KHR_shader_float16_int8
  PRINT_PROP(physical_device_.shader_float16_int8_types, shaderFloat16);
  PRINT_PROP(physical_device_.shader_float16_int8_types, shaderInt8);
#endif /* VK_KHR_shader_float16_int8 */
  ss << "    }" << std::endl;

  ss << "    Shader 64bit Features {" << std::endl;
  PRINT_BOOL(physical_device_.supports_int64_shader_types, shaderInt64)
  PRINT_BOOL(physical_device_.supports_float64_shader_types, shaderFloat64)
  ss << "    }" << std::endl;

#ifdef VK_KHR_shader_integer_dot_product
  ss << "    Shader Integer Dot Product Features {" << std::endl;
  PRINT_PROP(
      physical_device_.shader_int_dot_product_features,
      shaderIntegerDotProduct);
  ss << "    }" << std::endl;

  ss << "    Shader Integer Dot Product Properties {" << std::endl;
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct8BitUnsignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct8BitSignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct8BitMixedSignednessAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct4x8BitPackedUnsignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct4x8BitPackedSignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct4x8BitPackedMixedSignednessAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct16BitUnsignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct16BitSignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct16BitMixedSignednessAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct32BitUnsignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct32BitSignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct32BitMixedSignednessAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct64BitUnsignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct64BitSignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProduct64BitMixedSignednessAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating8BitUnsignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating8BitSignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating16BitUnsignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating16BitSignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating32BitUnsignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating32BitSignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating64BitUnsignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating64BitSignedAccelerated);
  PRINT_PROP(
      physical_device_.shader_int_dot_product_properties,
      integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated);
  ss << "    }" << std::endl;
#endif /* VK_KHR_shader_integer_dot_product */

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
  ss << "  VkDevice: " << device_.handle << std::endl;
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

} // namespace vkapi
} // namespace vkcompute
