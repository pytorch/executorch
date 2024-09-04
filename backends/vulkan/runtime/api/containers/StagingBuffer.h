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

#include <cstring>

namespace vkcompute {
namespace api {

class StagingBuffer final {
 private:
  Context* context_p_;
  vkapi::ScalarType dtype_;
  size_t numel_;
  size_t nbytes_;
  vkapi::VulkanBuffer vulkan_buffer_;

 public:
  StagingBuffer(
      Context* context_p,
      const vkapi::ScalarType dtype,
      const size_t numel)
      : context_p_(context_p),
        dtype_(dtype),
        numel_(numel),
        nbytes_(element_size(dtype_) * numel_),
        vulkan_buffer_(
            context_p_->adapter_ptr()->vma().create_staging_buffer(nbytes_)) {}

  StagingBuffer(const StagingBuffer&) = delete;
  StagingBuffer& operator=(const StagingBuffer&) = delete;

  StagingBuffer(StagingBuffer&&) = default;
  StagingBuffer& operator=(StagingBuffer&&) = default;

  ~StagingBuffer() {
    context_p_->register_buffer_cleanup(vulkan_buffer_);
  }

  inline vkapi::ScalarType dtype() {
    return dtype_;
  }

  inline vkapi::VulkanBuffer& buffer() {
    return vulkan_buffer_;
  }

  inline void* data() {
    return vulkan_buffer_.allocation_info().pMappedData;
  }

  inline size_t numel() {
    return numel_;
  }

  inline size_t nbytes() {
    return nbytes_;
  }

  inline void copy_from(const void* src, const size_t nbytes) {
    VK_CHECK_COND(nbytes <= nbytes_);
    memcpy(data(), src, nbytes);
    vmaFlushAllocation(
        vulkan_buffer_.vma_allocator(),
        vulkan_buffer_.allocation(),
        0u,
        VK_WHOLE_SIZE);
  }

  inline void copy_to(void* dst, const size_t nbytes) {
    VK_CHECK_COND(nbytes <= nbytes_);
    vmaInvalidateAllocation(
        vulkan_buffer_.vma_allocator(),
        vulkan_buffer_.allocation(),
        0u,
        VK_WHOLE_SIZE);
    memcpy(dst, data(), nbytes);
  }

  inline void set_staging_zeros() {
    memset(data(), 0, nbytes_);
  }
};

} // namespace api
} // namespace vkcompute
