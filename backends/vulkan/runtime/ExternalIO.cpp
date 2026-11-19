/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/ExternalIO.h>

#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>

#ifdef VK_ANDROID_external_memory_android_hardware_buffer
#include <executorch/backends/vulkan/runtime/vk_api/memory/Buffer.h>
#endif /* VK_ANDROID_external_memory_android_hardware_buffer */

namespace executorch {
namespace backends {
namespace vulkan {

using executorch::runtime::Error;
using executorch::runtime::Result;

namespace {

struct ExternalIOSet {
  std::vector<ExternalIOBinding> bindings;
  const void* owner = nullptr;
};

struct Registry {
  std::mutex mutex;
  std::map<std::string, ExternalIOSet> sets;
};

// Never destroyed, so that delegates destroyed during static destruction can
// still release their sets.
Registry& registry() {
  static Registry* const r = new Registry();
  return *r;
}

#ifdef VK_ANDROID_external_memory_android_hardware_buffer
using vkcompute::vkapi::ahardwarebuffer_fns;

void update_ahb_refs(
    const std::vector<ExternalIOBinding>& bindings,
    void (*const fn)(AHardwareBuffer*)) {
  for (const ExternalIOBinding& binding : bindings) {
    if (const auto* r =
            std::get_if<ExternalAHardwareBuffer>(&binding.resource)) {
      fn(r->ahb);
    }
  }
}
#endif /* VK_ANDROID_external_memory_android_hardware_buffer */

bool same_memory(const ExternalIOResource& a, const ExternalIOResource& b) {
  if (a.index() != b.index()) {
    return false;
  }
  if (const auto* r = std::get_if<ExternalAHardwareBuffer>(&a)) {
    return r->ahb == std::get<ExternalAHardwareBuffer>(b).ahb;
  }
  if (const auto* r = std::get_if<ExternalVkBuffer>(&a)) {
    return r->buffer == std::get<ExternalVkBuffer>(b).buffer;
  }
  return std::get<ExternalVkImage>(a).image ==
      std::get<ExternalVkImage>(b).image;
}

Error check_binding(const ExternalIOBinding& binding) {
  const char* const io = slot_name(binding.slot);
  if (const auto* ahb =
          std::get_if<ExternalAHardwareBuffer>(&binding.resource)) {
#ifdef VK_ANDROID_external_memory_android_hardware_buffer
    ET_CHECK_OR_RETURN_ERROR(
        ahb->ahb != nullptr,
        InvalidArgument,
        "%s %zu: null AHardwareBuffer",
        io,
        binding.index);
    ET_CHECK_OR_RETURN_ERROR(
        ahardwarebuffer_fns().acquire != nullptr &&
            ahardwarebuffer_fns().release != nullptr,
        NotSupported,
        "AHardwareBuffer_acquire and AHardwareBuffer_release are unavailable");
#else
    (void)ahb;
    ET_LOG(
        Error,
        "%s %zu: AHardwareBuffer needs an Android build",
        io,
        binding.index);
    return Error::NotSupported;
#endif /* VK_ANDROID_external_memory_android_hardware_buffer */
  } else if (
      const auto* buffer = std::get_if<ExternalVkBuffer>(&binding.resource)) {
    ET_CHECK_OR_RETURN_ERROR(
        buffer->buffer != VK_NULL_HANDLE && buffer->nbytes > 0,
        InvalidArgument,
        "%s %zu: null or empty VkBuffer",
        io,
        binding.index);
  } else {
    const auto& image = std::get<ExternalVkImage>(binding.resource);
    ET_CHECK_OR_RETURN_ERROR(
        image.image != VK_NULL_HANDLE && image.view != VK_NULL_HANDLE,
        InvalidArgument,
        "%s %zu: null VkImage or VkImageView",
        io,
        binding.index);
    ET_CHECK_OR_RETURN_ERROR(
        binding.target == ExternalIOTarget::Tensor,
        InvalidArgument,
        "%s %zu: a VkImage can only back a tensor, not a staging buffer",
        io,
        binding.index);
  }
  return Error::Ok;
}

} // namespace

Error register_external_io(
    const char* name,
    std::vector<ExternalIOBinding> bindings) {
  ET_CHECK_OR_RETURN_ERROR(
      name != nullptr && name[0] != '\0' &&
          std::strlen(name) < executorch::runtime::kMaxOptionValueLength,
      InvalidArgument,
      "external I/O set name must be non-empty and shorter than %zu chars",
      executorch::runtime::kMaxOptionValueLength);
  ET_CHECK_OR_RETURN_ERROR(
      !bindings.empty(),
      InvalidArgument,
      "external I/O set '%s' is empty",
      name);
  std::set<std::pair<SlotType, size_t>> seen;
  std::vector<const ExternalIOResource*> tensor_resources;
  for (const ExternalIOBinding& binding : bindings) {
    ET_CHECK_OK_OR_RETURN_ERROR(check_binding(binding));
    const char* const io = slot_name(binding.slot);
    ET_CHECK_OR_RETURN_ERROR(
        seen.insert({binding.slot, binding.index}).second,
        InvalidArgument,
        "external I/O set '%s' binds %s %zu twice",
        name,
        io,
        binding.index);
    if (binding.target == ExternalIOTarget::Tensor) {
      for (const ExternalIOResource* other : tensor_resources) {
        ET_CHECK_OR_RETURN_ERROR(
            !same_memory(binding.resource, *other),
            InvalidArgument,
            "external I/O set '%s': %s %zu backs its tensor with a resource "
            "that already backs another tensor",
            name,
            io,
            binding.index);
      }
      tensor_resources.push_back(&binding.resource);
    }
  }

  std::lock_guard<std::mutex> lock(registry().mutex);
  ET_CHECK_OR_RETURN_ERROR(
      registry().sets.count(name) == 0,
      InvalidState,
      "external I/O set '%s' is already registered",
      name);
#ifdef VK_ANDROID_external_memory_android_hardware_buffer
  update_ahb_refs(bindings, ahardwarebuffer_fns().acquire);
#endif /* VK_ANDROID_external_memory_android_hardware_buffer */
  registry().sets[name].bindings = std::move(bindings);
  return Error::Ok;
}

Error unregister_external_io(const char* name) {
  std::lock_guard<std::mutex> lock(registry().mutex);
  const auto it = registry().sets.find(name == nullptr ? "" : name);
  ET_CHECK_OR_RETURN_ERROR(
      it != registry().sets.end(),
      NotFound,
      "external I/O set '%s' is not registered",
      name == nullptr ? "" : name);
  ET_CHECK_OR_RETURN_ERROR(
      it->second.owner == nullptr,
      InvalidState,
      "external I/O set '%s' is bound to a loaded method",
      name);
#ifdef VK_ANDROID_external_memory_android_hardware_buffer
  update_ahb_refs(it->second.bindings, ahardwarebuffer_fns().release);
#endif /* VK_ANDROID_external_memory_android_hardware_buffer */
  registry().sets.erase(it);
  return Error::Ok;
}

Result<std::vector<ExternalIOBinding>> claim_external_io(
    const char* name,
    const void* owner) {
  std::lock_guard<std::mutex> lock(registry().mutex);
  const auto it = registry().sets.find(name);
  ET_CHECK_OR_RETURN_ERROR(
      it != registry().sets.end(),
      NotFound,
      "external I/O set '%s' is not registered",
      name);
  ET_CHECK_OR_RETURN_ERROR(
      it->second.owner == nullptr,
      InvalidState,
      "external I/O set '%s' is already bound to another Vulkan delegate; a "
      "set binds one delegate at a time",
      name);
  it->second.owner = owner;
  return it->second.bindings;
}

void release_external_io(const void* owner) {
  std::lock_guard<std::mutex> lock(registry().mutex);
  for (auto& entry : registry().sets) {
    if (entry.second.owner == owner) {
      entry.second.owner = nullptr;
    }
  }
}

} // namespace vulkan
} // namespace backends
} // namespace executorch
