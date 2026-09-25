/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan_shared/runtime/SharedVulkanContext.h>
#include <executorch/backends/vulkan_shared/runtime/export.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace vulkan_shared {

// Process-local registry used by independently initialized GPU delegates. The
// canonical target is a shared library so all delegate DSOs observe the same
// registry instance.
class EXECUTORCH_VULKAN_SHARED_API SharedVulkanContextRegistry final {
 public:
  using CreateFn = std::function<runtime::Result<SharedVulkanContextPtr>()>;

  static SharedVulkanContextRegistry& Get();

  SharedVulkanContextRegistry(const SharedVulkanContextRegistry&) = delete;
  SharedVulkanContextRegistry& operator=(const SharedVulkanContextRegistry&) =
      delete;

  SharedVulkanContextPtr lookup(const SharedVulkanContextKey& key);

  runtime::Result<SharedVulkanContextPtr> lookup_or_create(
      const SharedVulkanContextKey& key,
      CreateFn create_fn);

  runtime::Error register_context(SharedVulkanContextPtr context);

  runtime::Result<SharedVulkanContextPtr> register_external_context(
      SharedVulkanContextCreateInfo create_info);

  runtime::Error unregister_context(const SharedVulkanContextKey& key);

  // Test-only. The caller must ensure that there are no concurrent
  // registry operations or in-flight lookup_or_create() calls.
  void clear_for_testing();

 private:
  struct Entry final {
    SharedVulkanContextPtr context;
    bool creating = false;
    std::condition_variable creation_complete;
  };

  struct KeyHash final {
    size_t operator()(const SharedVulkanContextKey& key) const;
  };

  SharedVulkanContextRegistry() = default;

  // Never release a SharedVulkanContext/lifetime_anchor while this mutex is
  // held: backend teardown may re-enter the registry.
  std::mutex mutex_;
  std::unordered_map<SharedVulkanContextKey, std::shared_ptr<Entry>, KeyHash>
      registry_;
};

} // namespace vulkan_shared
} // namespace backends
} // namespace executorch
