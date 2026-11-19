/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

struct AHardwareBuffer;

namespace executorch {
namespace backends {
namespace vulkan {

/*
 * Zero-copy graph I/O for the Vulkan delegate.
 *
 * An app registers a named set of resources for a delegate's inputs and
 * outputs, then loads the method with the `kExternalIOOption` runtime spec of
 * "VulkanBackend" set to that name:
 *
 *   register_external_io(
 *       "frame",
 *       {{SlotType::Input, 0, ExternalIOTarget::Staging,
 *         ExternalAHardwareBuffer{ahb}}});
 *   BackendOptions<1> options;
 *   options.set_option(kExternalIOOption, "frame");
 *   LoadBackendOptionsMap map;
 *   map.set_options("VulkanBackend", options.view());
 *   module.load(map);
 *
 * The delegate then reads that input from, or writes that output to, the
 * resource on the GPU, and skips the host copy: the ExecuTorch tensor of that
 * argument is not read or written, although its sizes still drive resizing.
 *
 * Indices are the delegate's argument order. When the whole method is one
 * Vulkan delegate, its inputs are the method's in order, but its outputs need
 * not be: the `getitem` nodes after `executorch_call_delegate` in the lowered
 * graph map them to the method's. A set binds to one delegate at a time,
 * so a method with two Vulkan delegates fails to load rather than binding a
 * resource to the wrong tensor. The runtime spec reaches every method loaded
 * with it, so give it only to the method the set is for (e.g. through
 * Module::load_method()). Every binding must match an input or output.
 *
 * execute() returns after the delegate's GPU work completes; the app finishes
 * its own GPU writes to inputs before calling it. No queue family ownership
 * transfer is recorded. An input and an output may share a staging resource
 * (e.g. a recurrent state): inputs are staged before outputs are written. A
 * resource can back only one tensor, since the graph cannot order accesses to
 * two tensors that alias the same memory.
 */

// Runtime spec of "VulkanBackend" naming the set a method load binds. A char
// array, since BackendOptions::set_option() takes keys as arrays.
// @lint-ignore CLANGTIDY facebook-hte-CArray
inline constexpr char kExternalIOOption[] = "external_io";

// Whether a binding is for a graph input or output.
enum class SlotType : uint8_t {
  Input,
  Output,
};

inline const char* slot_name(const SlotType slot) {
  return slot == SlotType::Input ? "input" : "output";
}

// What a resource replaces for a graph input or output.
enum class ExternalIOTarget : uint8_t {
  // The staging buffer: contiguous NCHW data in the staging dtype, which is
  // the ExecuTorch tensor's dtype except that 64-bit types are staged as their
  // 32-bit counterparts, and half as float on devices without full 16-bit
  // buffer support. 8-bit data is accessed in whole 32-bit words, so its
  // resource must hold a multiple of 4 bytes. The delegate still converts it
  // to or from the tensor's own storage on the GPU.
  Staging,
  // The tensor's own storage: no staging buffer and no conversion. The
  // resource must match the tensor's storage type, sizes, GPU dtype and
  // packing; a mismatch fails the load.
  // A buffer pads the packed dim to a multiple of 4, as a texture does. Keep
  // an input's padding zero: operators assume it is.
  Tensor,
};

// A BLOB AHardwareBuffer with GPU_DATA_BUFFER usage, imported into a buffer.
// Android only. Registration holds a reference until unregistration, and the
// delegate holds its own once it imports the buffer.
struct ExternalAHardwareBuffer {
  AHardwareBuffer* ahb;
};

// A VkBuffer with VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, created on the
// delegate's VkDevice, which the app shares through
// vkcompute::vkapi::set_and_get_external_adapter(). The app destroys it after
// every method bound to it.
struct ExternalVkBuffer {
  VkBuffer buffer;
  VkDeviceSize nbytes;
};

// A 3D VkImage with VK_IMAGE_USAGE_STORAGE_BIT and VK_IMAGE_USAGE_SAMPLED_BIT,
// and a 3D view of it, created on the delegate's VkDevice. Can only replace a
// tensor's own storage, and only texture3d storage. It must be in
// VK_IMAGE_LAYOUT_GENERAL whenever execute() runs; the delegate keeps it
// there. The app destroys it after every method bound to it.
struct ExternalVkImage {
  VkImage image;
  VkImageView view;
  VkFormat format;
  VkExtent3D extents;
};

using ExternalIOResource =
    std::variant<ExternalAHardwareBuffer, ExternalVkBuffer, ExternalVkImage>;

struct ExternalIOBinding {
  SlotType slot;
  // Position among the delegate's inputs or outputs.
  size_t index;
  ExternalIOTarget target;
  ExternalIOResource resource;
};

executorch::runtime::Error register_external_io(
    const char* name,
    std::vector<ExternalIOBinding> bindings);

// Fails while a loaded method is bound to the set.
executorch::runtime::Error unregister_external_io(const char* name);

// Used by the delegate: binds the set to `owner` and returns its bindings.
executorch::runtime::Result<std::vector<ExternalIOBinding>> claim_external_io(
    const char* name,
    const void* owner);

// Used by the delegate: unbinds every set bound to `owner`.
void release_external_io(const void* owner);

} // namespace vulkan
} // namespace backends
} // namespace executorch
