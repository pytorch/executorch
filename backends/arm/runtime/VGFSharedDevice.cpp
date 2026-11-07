/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/arm/runtime/VGFNeuralStatistics.h>
#include <executorch/backends/arm/runtime/VGFSharedDevice.h>
#include <executorch/backends/arm/runtime/VGFZeroCopy.h>
#include <executorch/runtime/platform/log.h>

#if defined(EXECUTORCH_VGF_SHARED_DEVICE) && EXECUTORCH_VGF_SHARED_DEVICE
#include <executorch/backends/vulkan/runtime/ResolveSharedDevice.h>
#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>
#endif

#include <algorithm>
#include <cstring>
#include <exception>
#include <new>
#include <utility>
#include <vector>

namespace executorch {
namespace backends {
namespace vgf {
using runtime::Error;
using runtime::Result;
using namespace vulkan_shared;

VgfSharedDeviceState::~VgfSharedDeviceState() {
  // VgfRepr drains its queue and frees its children before releasing this
  // state.
  if (command_pool != VK_NULL_HANDLE && device) {
    vkDestroyCommandPool(device->device(), command_pool, nullptr);
  }
}

Result<SharedVulkanRuntimeConfig> parse_vgf_shared_device_config(
    const runtime::BackendInitContext& context) {
  auto parsed = parse_shared_vulkan_runtime_config(context);
  if (!parsed.ok()) {
    return parsed.error();
  }
  const auto name =
      context.get_runtime_spec<const char*>(kSharedDeviceNameOption);
  const auto legacy_name =
      context.get_runtime_spec<const char*>(kLegacySharedContextNameOption);
  const auto mode =
      context.get_runtime_spec<const char*>(kSharedDeviceModeOption);
  const auto legacy_mode =
      context.get_runtime_spec<const char*>(kLegacySharedContextModeOption);
  const auto group = context.get_runtime_spec<int>(kSharedGroupIdOption);
  const auto missing = [](const auto& result) {
    return !result.ok() && result.error() == Error::NotFound;
  };
  if (missing(name) && missing(legacy_name) && missing(mode) &&
      missing(legacy_mode) && missing(group)) {
    parsed->device_mode = SharedDeviceMode::kDisabled;
  }
  return parsed;
}

#if defined(EXECUTORCH_VGF_SHARED_DEVICE) && EXECUTORCH_VGF_SHARED_DEVICE
namespace {
Error validate_device(const SharedVulkanDevicePtr& device) {
  if (!device || !device->is_valid()) {
    return Error::DelegateInvalidCompatibility;
  }
  // We use global instance-loaded Volk dispatch. Never overwrite it with
  // volkLoadDevice(), or accept a device from an unrelated instance.
  if (device->instance() != vkcompute::vkapi::runtime()->instance()) {
    ET_LOG(Error, "VGF shared device must use the ExecuTorch Vulkan instance");
    return Error::DelegateInvalidCompatibility;
  }
  static constexpr const char* required_extensions[] = {
      "VK_ARM_tensors",
      "VK_ARM_data_graph",
      "VK_KHR_maintenance5",
      "VK_KHR_deferred_host_operations"};
  for (const char* extension : required_extensions) {
    if (!device->has_device_extension(extension)) {
      ET_LOG(Error, "VGF shared device did not enable %s", extension);
      return Error::DelegateInvalidCompatibility;
    }
  }
  // An external device must not claim a feature that was not enabled at
  // vkCreateDevice. The extension is optional, but its metadata must agree.
  if (device->enabled_device_features().shader_replicated_composites &&
      !device->has_device_extension(
          VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME)) {
    ET_LOG(
        Error, "Replicated-composites feature requires the enabled extension");
    return Error::DelegateInvalidCompatibility;
  }
  if (!vgf_shared_features_compatible(device->enabled_device_features())) {
    ET_LOG(
        Error,
        "VGF shared device lacks required ENABLED feature metadata; "
        "pre-create a VGF-capable device before loading both delegates");
    return Error::DelegateInvalidCompatibility;
  }
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(device->physical_device(), &properties);
  if (properties.apiVersion < VK_API_VERSION_1_3) {
    ET_LOG(Error, "VGF shared execution requires Vulkan 1.3");
    return Error::DelegateInvalidCompatibility;
  }
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      device->physical_device(), &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      device->physical_device(), &count, families.data());
  const uint32_t family = device->queue_family_index();
  constexpr VkQueueFlags required =
      VK_QUEUE_COMPUTE_BIT | VK_QUEUE_DATA_GRAPH_BIT_ARM;
  if (family >= count || families[family].queueCount == 0 ||
      (families[family].queueFlags & required) != required) {
    ET_LOG(Error, "VGF shared queue must support BOTH compute and data graph");
    return Error::DelegateInvalidCompatibility;
  }
  static constexpr const char* required_commands[] = {
      "vkCreateTensorARM",
      "vkDestroyTensorARM",
      "vkCreateTensorViewARM",
      "vkDestroyTensorViewARM",
      "vkGetTensorMemoryRequirementsARM",
      "vkBindTensorMemoryARM",
      "vkCreateDataGraphPipelinesARM",
      "vkCmdDispatchDataGraphARM",
      "vkCreateDataGraphPipelineSessionARM",
      "vkCmdPipelineBarrier2"};
  for (const char* command : required_commands) {
    if (vkGetDeviceProcAddr(device->device(), command) == nullptr) {
      ET_LOG(Error, "VGF shared device is missing %s", command);
      return Error::DelegateInvalidCompatibility;
    }
  }
  const auto& features = device->enabled_device_features();
  if (features.arm_data_graph_neural_statistics &&
      !device->has_device_extension(
          "VK_ARM_data_graph_neural_accelerator_statistics")) {
    return Error::DelegateInvalidCompatibility;
  }
  if (features.shader_bfloat16 &&
      !device->has_device_extension("VK_KHR_shader_bfloat16")) {
    return Error::DelegateInvalidCompatibility;
  }
  return Error::Ok;
}

Result<SharedVulkanDevicePtr> create_device(const SharedVulkanDeviceKey& key) {
  struct DeviceOwner final {
    // Destroy the VkDevice before releasing the owner of its VkInstance.
    std::shared_ptr<void> instance_lifetime;
    VkDevice device = VK_NULL_HANDLE;
    ~DeviceOwner() {
      if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
      }
    }
  };
  auto owner = std::make_shared<DeviceOwner>();
  SharedVulkanDeviceCreateInfo info;
  info.key = key;
  const auto statistics = get_vgf_neural_statistics_runtime_config();
  bool statistics_enabled = false;
  VgfHostMemoryImportCapabilities imports{};
  const VkResult result = vkml_allocate_basics(
      &info.instance,
      &info.physical_device,
      &owner->device,
      &info.queue,
      nullptr,
      &info.queue_family_index,
      statistics.requested,
      &statistics_enabled,
      vgf_zero_copy_io_build_enabled(),
      &imports,
      &owner->instance_lifetime,
      &info);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "VGF shared device creation failed: %d", result);
    return Error::DelegateInvalidCompatibility;
  }
  info.device = owner->device;
  info.lifetime_anchor = owner;
  auto device = std::make_shared<SharedVulkanDevice>(std::move(info));
  const Error compatible = validate_device(device);
  if (compatible != Error::Ok) {
    return compatible;
  }
  ET_LOG(
      Info,
      "VGF shared device created: name=%s group=%d device=%p",
      key.device_name.c_str(),
      key.group_id,
      static_cast<void*>(device->device()));
  return device;
}

VgfHostMemoryImportCapabilities import_capabilities(
    const SharedVulkanDevice& device) {
  VgfHostMemoryImportCapabilities result{};
#if defined(VK_EXT_external_memory_host)
  uint32_t count = 0;
  if (vkEnumerateDeviceExtensionProperties(
          device.physical_device(), nullptr, &count, nullptr) != VK_SUCCESS) {
    return result;
  }
  std::vector<VkExtensionProperties> extensions(count);
  if (vkEnumerateDeviceExtensionProperties(
          device.physical_device(), nullptr, &count, extensions.data()) !=
      VK_SUCCESS) {
    return result;
  }
  result.physical_device_advertised = std::any_of(
      extensions.begin(), extensions.end(), [](const auto& extension) {
        return std::strcmp(
                   extension.extensionName,
                   VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0;
      });
  result.logical_device_enabled =
      device.has_device_extension(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
  if (result.physical_device_advertised) {
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host{};
    host.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &host;
    vkGetPhysicalDeviceProperties2(device.physical_device(), &properties);
    result.min_imported_host_pointer_alignment =
        host.minImportedHostPointerAlignment;
  }
#endif
  return result;
}
} // namespace
#endif

Result<SharedVulkanDevicePtr> resolve_vgf_shared_device(
    const SharedVulkanRuntimeConfig& config) {
  if (!config.enabled()) {
    return SharedVulkanDevicePtr{};
  }
#if defined(EXECUTORCH_VGF_SHARED_DEVICE) && EXECUTORCH_VGF_SHARED_DEVICE
  try {
    const SharedVulkanDeviceKey key{config.device_name, config.group_id};
    // Reuse single-creator/provenance policy, including create_only.
    auto result = vulkan::resolve_vulkan_shared_device(
        config, [&]() { return create_device(key); });
    if (!result.ok()) {
      return result.error();
    }
    const Error compatible = validate_device(result.get());
    if (compatible != Error::Ok) {
      return compatible;
    }
    return result;
  } catch (const std::bad_alloc&) {
    return Error::MemoryAllocationFailed;
  } catch (const std::exception& error) {
    ET_LOG(Error, "VGF shared device resolution failed: %s", error.what());
    return Error::DelegateInvalidCompatibility;
  }
#else
  ET_LOG(
      Error,
      "VGF shared execution requires EXECUTORCH_BUILD_VULKAN=ON "
      "and EXECUTORCH_BUILD_VGF=ON");
  return Error::NotSupported;
#endif
}

Result<std::shared_ptr<VgfSharedDeviceState>> resolve_vgf_shared_device_state(
    const runtime::BackendInitContext& context) {
  auto config = parse_vgf_shared_device_config(context);
  if (!config.ok()) {
    return config.error();
  }
  auto shared = resolve_vgf_shared_device(config.get());
  if (!shared.ok()) {
    return shared.error();
  }
  if (!shared.get()) {
    return std::shared_ptr<VgfSharedDeviceState>{};
  }
#if defined(EXECUTORCH_VGF_SHARED_DEVICE) && EXECUTORCH_VGF_SHARED_DEVICE
  try {
    auto state = std::make_shared<VgfSharedDeviceState>();
    state->device = shared.get();
    state->neural_statistics_device_enabled =
        state->device->enabled_device_features()
            .arm_data_graph_neural_statistics;
    state->host_memory_import_capabilities =
        import_capabilities(*state->device);
    VkCommandPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = state->device->queue_family_index();
    VkCommandPool created_pool = VK_NULL_HANDLE;
    const auto status = vkCreateCommandPool(
        state->device->device(), &pool, nullptr, &created_pool);
    if (status != VK_SUCCESS) {
      ET_LOG(Error, "VGF shared command pool creation failed: %d", status);
      return Error::Internal;
    }
    state->command_pool = created_pool;
    ET_LOG(
        Info,
        "VGF shared init: name=%s group=%d device=%p pool=%p",
        state->device->key().device_name.c_str(),
        state->device->key().group_id,
        static_cast<void*>(state->device->device()),
        reinterpret_cast<void*>(state->command_pool));
    return state;
  } catch (const std::bad_alloc&) {
    return Error::MemoryAllocationFailed;
  }
#else
  return Error::NotSupported;
#endif
}

} // namespace vgf
} // namespace backends
} // namespace executorch
