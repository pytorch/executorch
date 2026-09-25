/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan_shared/runtime/SharedVulkanContextRegistry.h>

#include <utility>

namespace executorch {
namespace backends {
namespace vulkan_shared {

SharedVulkanContextRegistry& SharedVulkanContextRegistry::Get() {
  // The default context is process-persistent. Intentionally do not register a
  // static destructor: delegate DSOs may be unloaded before their lifetime
  // anchors, so teardown must be explicit through unregister_context().
  static auto* registry = new SharedVulkanContextRegistry();
  return *registry;
}

size_t SharedVulkanContextRegistry::KeyHash::operator()(
    const SharedVulkanContextKey& key) const {
  const size_t token_hash = std::hash<std::string>{}(key.context_name);
  const size_t group_hash = std::hash<int>{}(key.group_id);
  return token_hash ^
      (group_hash + static_cast<size_t>(0x9e3779b9) + (token_hash << 6) +
       (token_hash >> 2));
}

SharedVulkanContextPtr SharedVulkanContextRegistry::lookup(
    const SharedVulkanContextKey& key) {
  if (!key.valid()) {
    return nullptr;
  }

  SharedVulkanContextPtr stale_context;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(key);
    if (it == registry_.end()) {
      return nullptr;
    }

    const auto& entry = it->second;
    if (entry->context && entry->context->is_valid()) {
      return entry->context;
    }

    // Move any stale context out while holding the registry lock, but release
    // it only after unlocking: its lifetime_anchor may run backend teardown and
    // re-enter this registry.
    stale_context = std::move(entry->context);
    if (!entry->creating) {
      registry_.erase(it);
    }
  }

  stale_context.reset();
  return nullptr;
}

runtime::Result<SharedVulkanContextPtr>
SharedVulkanContextRegistry::lookup_or_create(
    const SharedVulkanContextKey& key,
    CreateFn create_fn) {
  if (!key.valid() || !create_fn) {
    return runtime::Error::InvalidArgument;
  }

  std::shared_ptr<Entry> entry;
  SharedVulkanContextPtr stale_context;

  {
    std::unique_lock<std::mutex> lock(mutex_);
    auto [it, inserted] = registry_.try_emplace(key, std::make_shared<Entry>());
    (void)inserted;
    entry = it->second;

    while (entry->creating) {
      entry->creation_complete.wait(lock);
    }

    if (entry->context && entry->context->is_valid()) {
      return entry->context;
    }

    stale_context = std::move(entry->context);
    entry->creating = true;
  }

  // lifetime_anchor destruction may re-enter the registry.
  stale_context.reset();

  auto maybe_created = create_fn();
  runtime::Error create_error = runtime::Error::Ok;
  SharedVulkanContextPtr created;
  if (!maybe_created.ok()) {
    create_error = maybe_created.error();
  } else {
    created = maybe_created.get();
    if (!created || !created->is_valid() || created->key() != key) {
      created.reset();
      create_error = runtime::Error::InvalidArgument;
    }
  }

  SharedVulkanContextPtr selected;
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // register_context() is allowed to win a race with a creator. In that
    // case, discard the newly created context and return the registered one.
    if (entry->context && entry->context->is_valid()) {
      selected = entry->context;
    } else if (created) {
      entry->context = std::move(created);
      selected = entry->context;
    }

    entry->creating = false;
    entry->creation_complete.notify_all();
  }

  if (selected) {
    return selected;
  }
  return create_error == runtime::Error::Ok ? runtime::Error::Internal
                                            : create_error;
}

runtime::Error SharedVulkanContextRegistry::register_context(
    SharedVulkanContextPtr context) {
  if (!context || !context->is_valid()) {
    return runtime::Error::InvalidArgument;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] =
      registry_.try_emplace(context->key(), std::make_shared<Entry>());
  (void)inserted;
  auto& entry = it->second;

  if (entry->context && entry->context->is_valid()) {
    return entry->context.get() == context.get()
        ? runtime::Error::Ok
        : runtime::Error::AlreadyLoaded;
  }

  entry->context = std::move(context);
  entry->creation_complete.notify_all();
  return runtime::Error::Ok;
}

runtime::Result<SharedVulkanContextPtr>
SharedVulkanContextRegistry::register_external_context(
    SharedVulkanContextCreateInfo create_info) {
  auto context = std::make_shared<SharedVulkanContext>(std::move(create_info));
  const runtime::Error error = register_context(context);
  if (error != runtime::Error::Ok) {
    return error;
  }
  return context;
}

runtime::Error SharedVulkanContextRegistry::unregister_context(
    const SharedVulkanContextKey& key) {
  if (!key.valid()) {
    return runtime::Error::InvalidArgument;
  }

  std::shared_ptr<Entry> removed_entry;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(key);
    if (it == registry_.end()) {
      return runtime::Error::NotFound;
    }
    if (it->second->creating) {
      return runtime::Error::InvalidState;
    }

    // Unlink the entry under the registry lock, but retain ownership locally so
    // SharedVulkanContext/lifetime_anchor destruction cannot run while mutex_
    // is held. Backend teardown is allowed to re-enter this registry.
    removed_entry = std::move(it->second);
    registry_.erase(it);
  }

  removed_entry.reset();
  return runtime::Error::Ok;
}

void SharedVulkanContextRegistry::clear_for_testing() {
  decltype(registry_) removed_entries;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : registry_) {
      item.second->creating = false;
      item.second->creation_complete.notify_all();
    }

    // Remove everything atomically from the live registry, then allow entries
    // (and their lifetime anchors) to be destroyed after mutex_ is released.
    removed_entries.swap(registry_);
  }

  removed_entries.clear();
}

} // namespace vulkan_shared
} // namespace backends
} // namespace executorch
