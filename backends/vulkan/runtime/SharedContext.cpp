/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/SharedContext.h>
#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>
#include <executorch/runtime/platform/log.h>

#include <exception>
#include <new>
#include <utility>

namespace executorch {
namespace backends {
namespace vulkan {

using runtime::Error;
using runtime::Result;
using vulkan_shared::SharedContextMode;
using vulkan_shared::SharedVulkanContext;
using vulkan_shared::SharedVulkanContextCreateInfo;
using vulkan_shared::SharedVulkanContextKey;
using vulkan_shared::SharedVulkanContextPtr;
using vulkan_shared::SharedVulkanContextRegistry;
using vulkan_shared::SharedVulkanRuntimeConfig;
namespace vkapi = vkcompute::vkapi;

Result<SharedVulkanRuntimeConfig> parse_vulkan_shared_context_config(
    const runtime::BackendInitContext& context) {
  auto parsed = vulkan_shared::parse_shared_vulkan_runtime_config(context);
  if (!parsed.ok()) {
    return parsed.error();
  }
  const auto context_name = context.get_runtime_spec<const char*>(
      vulkan_shared::kSharedContextNameOption);
  const auto mode = context.get_runtime_spec<const char*>(
      vulkan_shared::kSharedContextModeOption);
  const auto group =
      context.get_runtime_spec<int>(vulkan_shared::kSharedGroupIdOption);
  if (!context_name.ok() && context_name.error() == Error::NotFound &&
      !mode.ok() && mode.error() == Error::NotFound && !group.ok() &&
      group.error() == Error::NotFound) {
    // for backwards compatibility we preserve old behavour, if
    // there are no options.
    parsed->context_mode = SharedContextMode::kDisabled;
  }
  return parsed;
}

Result<SharedVulkanContextPtr> resolve_vulkan_shared_context(
    const SharedVulkanRuntimeConfig& config,
    const SharedVulkanContextRegistry::CreateFn& create_fn) {
  if (!config.enabled()) {
    return SharedVulkanContextPtr{};
  }
  const SharedVulkanContextKey key{config.context_name, config.group_id};
  if (!key.valid()) {
    return Error::InvalidArgument;
  }
  auto& registry = SharedVulkanContextRegistry::Get();
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
  SharedVulkanContextPtr candidate;
  auto selected =
      registry.lookup_or_create(key, [&]() -> Result<SharedVulkanContextPtr> {
        try {
          auto created = create_fn();
          if (created.ok()) {
            candidate = created.get();
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
  if (config.create_only() && candidate != selected.get()) {
    return Error::AlreadyLoaded;
  }
  return selected;
}

Result<SharedVulkanContextPtr> create_vulkan_shared_context(
    const SharedVulkanContextKey& key) {
  if (!key.valid()) {
    return Error::InvalidArgument;
  }

  try {
    // We use Volk global instance-loaded dispatch, so
    // we need to use one instance and multiple devices
    // architecture.
    auto* runtime = vkapi::runtime();
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

    SharedVulkanContextCreateInfo info;
    info.key = key;
    info.instance = runtime->instance();
    info.physical_device = owner->physical_handle();
    info.device = owner->device_handle();
    info.queue = lease.queue.handle;
    info.queue_family_index = lease.queue.family_index;
    info.enabled_device_extensions = owner->enabled_device_extensions();
    // The global Runtime owns the VkInstance for process lifetime. The anchor
    // owns this key's Adapter and therefore its VkDevice and device resources.
    info.lifetime_anchor = owner;

    auto result = std::make_shared<SharedVulkanContext>(std::move(info));
    if (!result->is_valid()) {
      return Error::DelegateInvalidCompatibility;
    }
    return result;
  } catch (const std::bad_alloc&) {
    return Error::MemoryAllocationFailed;
  } catch (const std::exception& error) {
    ET_LOG(Error, "Shared Vulkan context creation failed: %s", error.what());
    return Error::DelegateInvalidCompatibility;
  } catch (...) {
    return Error::Internal;
  }
}

Result<std::unique_ptr<vkapi::Adapter>> resolve_vulkan_shared_adapter(
    const runtime::BackendInitContext& context) {
  auto config = parse_vulkan_shared_context_config(context);
  if (!config.ok()) {
    return config.error();
  }
  if (!config->enabled()) {
    return std::unique_ptr<vkapi::Adapter>{};
  }

  if (vkapi::set_and_get_external_adapter() != nullptr) {
    ET_LOG(
        Error,
        "Do not combine shared-context options with the local global external adapter");
    return Error::InvalidArgument;
  }

  const SharedVulkanContextKey key{config->context_name, config->group_id};
  auto shared = resolve_vulkan_shared_context(
      config.get(), [&]() { return create_vulkan_shared_context(key); });
  if (!shared.ok()) {
    ET_LOG(
        Error,
        "Unable to resolve shared Vulkan context '%s' (group %d): 0x%x",
        key.context_name.c_str(),
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
        "Shared Vulkan context '%s' (group %d) uses a different VkInstance; "
        "Volk loader-dispatch mode requires the ExecuTorch Vulkan runtime instance",
        key.context_name.c_str(),
        key.group_id);
    return Error::DelegateInvalidCompatibility;
  }

  try {
    return std::make_unique<vkapi::Adapter>(shared.get(), "");
  } catch (const std::bad_alloc&) {
    return Error::MemoryAllocationFailed;
  } catch (const std::exception& error) {
    ET_LOG(Error, "Shared Vulkan context is incompatible: %s", error.what());
    return Error::DelegateInvalidCompatibility;
  } catch (...) {
    return Error::Internal;
  }
}

} // namespace vulkan
} // namespace backends
} // namespace executorch
