/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#if defined(__GNUC__) || defined(__clang__)
#define VGF_FUSED_STREAM_API __attribute__((visibility("default")))
#else
#define VGF_FUSED_STREAM_API
#endif

namespace executorch::backends::vgf {

// Callback runs synchronously inside VGFBackend::execute(), after input
// preparation and before queue submission, while the backend's queue/pool
// mutex is held. Never re-enter ExecuTorch/VGF from this callback.
//
// Record *executable primary* producer and consumer command buffers with
// `device`, for `queue_family_index`. Use a separate application-owned command
// pool for this device/family, rather than mutating VGF's command pool.
// Both commands must remain valid through the return of Method::execute().
// The caller is responsible for producer->VGF and VGF->consumer barriers.
struct VgfFusedDeviceContext {
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t queue_family_index = UINT32_MAX;
};

using VgfPrepareFrameCommands = bool (*)(
    void* user_data,
    const VgfFusedDeviceContext& context,
    VkCommandBuffer* producer,
    VkCommandBuffer* consumer) noexcept;

// Exactly one VGF delegate execution is allowed per request. This is a
// synchronous, non-owning API. Register it on the same thread that calls
// Method::execute(), and retain its resources until execute returns.
//
// Without prepare(), pre-recorded commands require both expected_device and
// expected_queue_family_index to avoid submitting on the wrong device.
// With prepare(), the callback receives VGF's actual device and family.
struct VgfFusedCommandStreamRequest {
  VgfPrepareFrameCommands prepare = nullptr;
  void* user_data = nullptr;
  VkCommandBuffer producer = VK_NULL_HANDLE;
  VkCommandBuffer consumer = VK_NULL_HANDLE;
  VkDevice expected_device = VK_NULL_HANDLE;
  uint32_t expected_queue_family_index = UINT32_MAX;
  bool attempted = false;
  bool submitted = false;
  bool completed = false;
  VkResult submit_result = VK_NOT_READY;
  VkResult wait_result = VK_NOT_READY;
};

// Fails closed for overlapping/nested registrations on the same thread.
// The request is not owned by the runtime.
VGF_FUSED_STREAM_API bool vgf_try_register_fused_command_stream(
    VgfFusedCommandStreamRequest* request) noexcept;
VGF_FUSED_STREAM_API void vgf_unregister_fused_command_stream(
    const VgfFusedCommandStreamRequest* request) noexcept;

class VgfFusedCommandStreamScope final {
 public:
  explicit VgfFusedCommandStreamScope(
      VgfFusedCommandStreamRequest& request) noexcept
      : request_(&request),
        active_(vgf_try_register_fused_command_stream(request_)) {}

  ~VgfFusedCommandStreamScope() {
    if (active_) {
      vgf_unregister_fused_command_stream(request_);
    }
  }

  bool active() const noexcept {
    return active_;
  }

  VgfFusedCommandStreamScope(const VgfFusedCommandStreamScope&) = delete;
  VgfFusedCommandStreamScope& operator=(const VgfFusedCommandStreamScope&) =
      delete;

 private:
  VgfFusedCommandStreamRequest* request_ = nullptr;
  bool active_ = false;
};

} // namespace executorch::backends::vgf
