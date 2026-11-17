/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/ResolveSharedDevice.h>
#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>
#include <executorch/runtime/platform/log.h>

#include <exception>
#include <map>
#include <mutex>
#include <new>
#include <utility>

namespace executorch {
namespace backends {
namespace vulkan {

using runtime::Error;
using runtime::Result;
using vulkan_shared::SharedDeviceMode;
using vulkan_shared::SharedVulkanDevice;
using vulkan_shared::SharedVulkanDeviceCreateInfo;
using vulkan_shared::SharedVulkanDeviceKey;
using vulkan_shared::SharedVulkanDevicePtr;
using vulkan_shared::SharedVulkanDeviceRegistry;
using vulkan_shared::SharedVulkanRuntimeConfig;
namespace vkapi = vkcompute::vkapi;

namespace {
using DeviceKey = std::pair<std::string, int>;

// Weak provenance prevents stale keys from being mistaken for backend-owned
// devices after unregister/re-register, and does not extend VkDevice lifetime.
struct DeviceProvenance {
  std::mutex mutex;
  std::map<DeviceKey, std::weak_ptr<SharedVulkanDevice>> created;
};

DeviceProvenance& provenance() {
  static auto* record = new DeviceProvenance;
  return *record;
}

void remember_created_device(const SharedVulkanDevicePtr& device) {
  auto& record = provenance();
  std::lock_guard<std::mutex> guard(record.mutex);
  record.created[{device->key().device_name, device->key().group_id}] = device;
}

bool backend_created_device(const SharedVulkanDevicePtr& device) {
  auto& record = provenance();
  std::lock_guard<std::mutex> guard(record.mutex);
  const auto it =
      record.created.find({device->key().device_name, device->key().group_id});
  return it != record.created.end() && it->second.lock() == device;
}

struct AdapterCache {
  std::mutex mutex;
  std::map<DeviceKey, std::weak_ptr<vkapi::Adapter>> adapters;
};

AdapterCache& adapter_cache() {
  static auto* cache = new AdapterCache;
  return *cache;
}

std::shared_ptr<vkapi::Adapter> get_or_create_shared_adapter(
    const SharedVulkanDevicePtr& device) {
  auto& cache = adapter_cache();
  std::lock_guard<std::mutex> guard(cache.mutex);
  auto& slot =
      cache.adapters[{device->key().device_name, device->key().group_id}];
  if (auto existing = slot.lock()) {
    if (existing->shared_device() == device) {
      return existing;
    }
  }
  auto adapter = std::make_shared<vkapi::Adapter>(
      device, vkapi::set_and_get_pipeline_cache_data_path(""));
  slot = adapter;
  return adapter;
}
} // namespace

Result<SharedVulkanRuntimeConfig> parse_vulkan_shared_device_config(
    const runtime::BackendInitContext& context) {
  auto parsed = vulkan_shared::parse_shared_vulkan_runtime_config(context);
  if (!parsed.ok()) {
    return parsed.error();
  }
  const auto device_name = context.get_runtime_spec<const char*>(
      vulkan_shared::kSharedDeviceNameOption);
  const auto legacy_name = context.get_runtime_spec<const char*>(
      vulkan_shared::kLegacySharedContextNameOption);
  const auto mode = context.get_runtime_spec<const char*>(
      vulkan_shared::kSharedDeviceModeOption);
  const auto legacy_mode = context.get_runtime_spec<const char*>(
      vulkan_shared::kLegacySharedContextModeOption);
  const auto group =
      context.get_runtime_spec<int>(vulkan_shared::kSharedGroupIdOption);
  const auto missing = [](const auto& result) {
    return !result.ok() && result.error() == Error::NotFound;
  };
  if (missing(device_name) && missing(legacy_name) && missing(mode) &&
      missing(legacy_mode) && missing(group)) {
    // Preserve legacy Vulkan behavior when no sharing options are provided.
    parsed->device_mode = SharedDeviceMode::kDisabled;
  }
  return parsed;
}

Result<SharedVulkanDevicePtr> resolve_vulkan_shared_device(
    const SharedVulkanRuntimeConfig& config,
    const SharedVulkanDeviceRegistry::CreateFn& create_fn) {
  if (!config.enabled()) {
    return SharedVulkanDevicePtr{};
  }
  const SharedVulkanDeviceKey key{config.device_name, config.group_id};
  if (!key.valid()) {
    return Error::InvalidArgument;
  }
  auto& registry = SharedVulkanDeviceRegistry::Get();
  if (config.lookup_only()) {
    auto existing = registry.lookup(key);
    if (!existing) {
      return Error::NotFound;
    }
    return existing;
  }
  if ((!config.lookup_or_create() && !config.create_only()) || !create_fn) {
    return Error::InvalidArgument;
  }

  // Use the registry's single-creator protocol for BOTH creation modes:
  // lookup_or_create or create_only. A lookup-then-register sequence would
  // unnecessarily construct multiple devices and is particularly hazardous
  // with global device-specific dispatch volk.
  SharedVulkanDevicePtr candidate;
  auto selected =
      registry.lookup_or_create(key, [&]() -> Result<SharedVulkanDevicePtr> {
        try {
          auto created = create_fn();
          if (created.ok()) {
            candidate = created.get();
            if (candidate && candidate->is_valid() && candidate->key() == key) {
              // Must precede registry publication for concurrent create_only.
              remember_created_device(candidate);
            }
          }
          return created;
        } catch (const std::bad_alloc&) {
          return Error::MemoryAllocationFailed;
        } catch (...) {
          return Error::Internal;
        }
      });
  if (!selected.ok()) {
    return selected.error();
  }
  // An external registration may have won while the creator ran. create_only
  // must not report success for someone else's context in that case, either.
  if (config.create_only() && candidate != selected.get() &&
      !backend_created_device(selected.get())) {
    return Error::AlreadyLoaded;
  }
  return selected;
}

Result<SharedVulkanDevicePtr> create_vulkan_shared_device(
    const SharedVulkanDeviceKey& key) {
  if (!key.valid()) {
    return Error::InvalidArgument;
  }

  try {
    // We use Volk global instance-loaded dispatch, so
    // we need to use one instance and multiple devices
    // architecture.
    auto runtime_owner = vkapi::runtime_owner();
    auto* runtime = runtime_owner.get();
    auto* default_adapter = runtime->get_adapter_p();

    // We create Adapter under the same instance VkInstance.
    auto owner = std::make_shared<vkapi::Adapter>(
        runtime->instance(),
        vkapi::PhysicalDevice(
            runtime->instance(), default_adapter->physical_handle()),
        1,
        "");

    struct QueueLease final {
      vkapi::Adapter* adapter;
      vkapi::Adapter::Queue queue;
      ~QueueLease() {
        adapter->return_queue(queue);
      }
    } lease{owner.get(), owner->request_queue()};

    SharedVulkanDeviceCreateInfo info;
    info.key = key;
    info.instance = runtime->instance();
    info.physical_device = owner->physical_handle();
    info.device = owner->device_handle();
    info.queue = lease.queue.handle;
    info.queue_family_index = lease.queue.family_index;
    info.enabled_device_extensions = owner->enabled_device_extensions();
    info.enabled_device_features = owner->enabled_device_features();
    // A registered context may outlive the global Runtime's static owner.
    // Members are destroyed in reverse order: release the device and its
    // resources before releasing the Runtime that owns their VkInstance.
    struct ContextOwner final {
      std::shared_ptr<vkapi::Runtime> runtime;
      std::shared_ptr<vkapi::Adapter> adapter;
    };
    info.lifetime_anchor = std::make_shared<ContextOwner>(
        ContextOwner{std::move(runtime_owner), owner});

    auto result = std::make_shared<SharedVulkanDevice>(std::move(info));
    if (!result->is_valid()) {
      return Error::DelegateInvalidCompatibility;
    }
    return result;
  } catch (const std::bad_alloc&) {
    return Error::MemoryAllocationFailed;
  } catch (const std::exception& error) {
    ET_LOG(Error, "Shared Vulkan device creation failed: %s", error.what());
    return Error::DelegateInvalidCompatibility;
  } catch (...) {
    return Error::Internal;
  }
}

Result<std::shared_ptr<vkapi::Adapter>> resolve_vulkan_shared_adapter(
    const runtime::BackendInitContext& context) {
  auto config = parse_vulkan_shared_device_config(context);
  if (!config.ok()) {
    return config.error();
  }
  if (!config->enabled()) {
    return std::shared_ptr<vkapi::Adapter>{};
  }

  if (vkapi::set_and_get_external_adapter() != nullptr) {
    ET_LOG(
        Error,
        "Do not combine shared-device options with the local global external adapter");
    return Error::InvalidArgument;
  }

  const SharedVulkanDeviceKey key{config->device_name, config->group_id};
  auto shared = resolve_vulkan_shared_device(
      config.get(), [&]() { return create_vulkan_shared_device(key); });
  if (!shared.ok()) {
    ET_LOG(
        Error,
        "Unable to resolve shared Vulkan device '%s' (group %d): 0x%x",
        key.device_name.c_str(),
        key.group_id,
        static_cast<unsigned>(shared.error()));
    return shared.error();
  }

  // Global Volk instance entry points are valid only for the VkInstance passed
  // to volkLoadInstance() and its children. We permit multiple VkDevice objects
  // but requires one VkInstance per Vulkan backend linkage unit.
  // An application-owned context from another VkInstance needs
  // direct-loader dispatch or per-instance/per-device dispatch tables instead.
  auto* runtime = vkapi::runtime();
  if (shared.get()->instance() != runtime->instance()) {
    ET_LOG(
        Error,
        "Shared Vulkan device '%s' (group %d) uses a different VkInstance; "
        "Volk loader-dispatch mode requires the ExecuTorch Vulkan runtime instance",
        key.device_name.c_str(),
        key.group_id);
    return Error::DelegateInvalidCompatibility;
  }

  try {
    return get_or_create_shared_adapter(shared.get());
  } catch (const std::bad_alloc&) {
    return Error::MemoryAllocationFailed;
  } catch (const std::exception& error) {
    ET_LOG(Error, "Shared Vulkan device is incompatible: %s", error.what());
    return Error::DelegateInvalidCompatibility;
  } catch (...) {
    return Error::Internal;
  }
}

} // namespace vulkan
} // namespace backends
} // namespace executorch
