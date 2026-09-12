/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/gpu_shared/runtime/SharedGpuContext.h>
#include <executorch/backends/gpu_shared/runtime/export.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace gpu_shared {

// Process-local registry used by independently initialized GPU delegates. The
// canonical target is a shared library so all delegate DSOs observe the same
// registry instance.
class EXECUTORCH_GPU_SHARED_API SharedGpuContextRegistry final {
 public:
  using CreateFn = std::function<runtime::Result<SharedGpuContextPtr>()>;

  static SharedGpuContextRegistry& Get();

  SharedGpuContextRegistry(const SharedGpuContextRegistry&) = delete;
  SharedGpuContextRegistry& operator=(const SharedGpuContextRegistry&) = delete;

  SharedGpuContextPtr lookup(const SharedGpuContextKey& key);

  runtime::Result<SharedGpuContextPtr> lookup_or_create(
      const SharedGpuContextKey& key,
      CreateFn create_fn);

  runtime::Error register_context(SharedGpuContextPtr context);

  runtime::Result<SharedGpuContextPtr> register_external_context(
      SharedGpuContextCreateInfo create_info);

  runtime::Error unregister_context(const SharedGpuContextKey& key);

  void clear_for_testing();

 private:
  struct Entry final {
    SharedGpuContextPtr context;
    bool creating = false;
    std::condition_variable creation_complete;
  };

  struct KeyHash final {
    size_t operator()(const SharedGpuContextKey& key) const;
  };

  SharedGpuContextRegistry() = default;

  // Never release a SharedGpuContext/lifetime_anchor while this mutex is held:
  // backend teardown may re-enter the registry.
  std::mutex mutex_;
  std::unordered_map<SharedGpuContextKey, std::shared_ptr<Entry>, KeyHash>
      registry_;
};

} // namespace gpu_shared
} // namespace backends
} // namespace executorch
