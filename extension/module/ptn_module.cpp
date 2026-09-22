/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/ptn_module.h>

#include <atomic>
#include <mutex>

namespace executorch::extension::native_module::internal {
namespace {

struct PtnHookRegistry {
  std::atomic<const PtnHooks*> hooks{nullptr};
  PtnHooks registered_hooks{};
  std::mutex mutex;
};

PtnHookRegistry& registry() {
  static PtnHookRegistry value;
  return value;
}

} // namespace

PtnModule::~PtnModule() = default;

runtime::Error register_ptn_hooks(const PtnHooks& candidate) {
  if (candidate.load == nullptr) {
    return runtime::Error::InvalidArgument;
  }
  auto& hook_registry = registry();
  const std::lock_guard<std::mutex> lock(hook_registry.mutex);
  if (hook_registry.hooks.load(std::memory_order_relaxed) != nullptr) {
    return runtime::Error::AlreadyLoaded;
  }
  hook_registry.registered_hooks = candidate;
  hook_registry.hooks.store(
      &hook_registry.registered_hooks, std::memory_order_release);
  return runtime::Error::Ok;
}

const PtnHooks* get_ptn_hooks() {
  return registry().hooks.load(std::memory_order_acquire);
}

} // namespace executorch::extension::native_module::internal

namespace executorch::extension::native_module {

runtime::Result<std::unique_ptr<internal::PtnModule>> load_ptn(
    const internal::PtnSource& source,
    internal::Program::Verification verification) {
  const internal::PtnHooks* hooks = internal::get_ptn_hooks();
  if (hooks == nullptr) {
    return runtime::Error::NotSupported;
  }
  return hooks->load(source, verification);
}

} // namespace executorch::extension::native_module
