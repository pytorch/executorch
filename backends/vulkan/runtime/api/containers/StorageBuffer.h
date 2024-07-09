/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/api/Context.h>

#include <executorch/backends/vulkan/runtime/vk_api/memory/Buffer.h>

namespace vkcompute {
namespace api {

class StorageBuffer final {
 private:
  Context* context_p_;
  vkapi::ScalarType dtype_;
  size_t numel_;
  size_t nbytes_;
  vkapi::VulkanBuffer vulkan_buffer_;

 public:
  StorageBuffer(
      Context* context_p,
      const vkapi::ScalarType dtype,
      const size_t numel,
      const bool gpuonly = false)
      : context_p_(context_p),
        dtype_(dtype),
        numel_(numel),
        nbytes_(element_size(dtype_) * numel_),
        vulkan_buffer_(context_p_->adapter_ptr()->vma().create_storage_buffer(
            nbytes_,
            gpuonly)) {}

  StorageBuffer(const StorageBuffer&) = delete;
  StorageBuffer& operator=(const StorageBuffer&) = delete;

  StorageBuffer(StorageBuffer&&) = default;
  StorageBuffer& operator=(StorageBuffer&&) = default;

  ~StorageBuffer() {
    context_p_->register_buffer_cleanup(vulkan_buffer_);
  }

  inline vkapi::ScalarType dtype() {
    return dtype_;
  }

  inline vkapi::VulkanBuffer& buffer() {
    return vulkan_buffer_;
  }

  inline size_t numel() {
    return numel_;
  }

  inline size_t nbytes() {
    return nbytes_;
  }
};

} // namespace api
} // namespace vkcompute
