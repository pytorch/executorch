/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Set Volk's header mode before any shared runtime Vulkan headers.
#include <executorch/backends/vulkan/runtime/vk_api/Adapter.h>

#include <executorch/backends/vulkan_shared/runtime/SharedVulkanContextRegistry.h>
#include <executorch/backends/vulkan_shared/runtime/SharedVulkanRuntimeConfig.h>

#include <memory>

namespace executorch {
namespace backends {
namespace vulkan {

// This is a bridge between the Vulkan backend and the shared Vulkan runtime.
// We set a small set of function here to
// 1. read shared-context runtime options
// 2. decide whether to use local Vulkan or shared Vulkan
// 3. find/create a SharedVulkanContext
// 4. turn that shared context into a Vulkan Adapter

// Depending on BackendInitContext we decide whether this Vulkan backend
// initialization should use the shared-context path.
runtime::Result<vulkan_shared::SharedVulkanRuntimeConfig>
parse_vulkan_shared_context_config(const runtime::BackendInitContext& context);

// This function implements registry policy.
// Given a requested sharing mode, we decide whether we should look up a
// context, create one, or fail? We pass CreateFn here, so we can provide a fake
// context in tests. Note that creation is done outside of the registry lock.
runtime::Result<vulkan_shared::SharedVulkanContextPtr>
resolve_vulkan_shared_context(
    const vulkan_shared::SharedVulkanRuntimeConfig& config,
    const vulkan_shared::SharedVulkanContextRegistry::CreateFn& create_fn);

// Each registry key gets its own owned Adapter/VkDevice under the common
// instance. Note that for safe teardown we use lifetime_anchor that retains the
// owner Adapter.
runtime::Result<vulkan_shared::SharedVulkanContextPtr>
create_vulkan_shared_context(const vulkan_shared::SharedVulkanContextKey& key);

// This is the highest level function -
// entry point for configuring Vulkan shared-context support.
// Parses the runtime configuration, resolves or creates the corresponding
// SharedVulkanContext when enabled, and returns an Adapter backed by it.
// Returns nullptr when shared-context support is disabled.
runtime::Result<std::unique_ptr<vkcompute::vkapi::Adapter>>
resolve_vulkan_shared_adapter(const runtime::BackendInitContext& context);

} // namespace vulkan
} // namespace backends
} // namespace executorch
