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
  vkapi::VulkanBuffer vulkan_buffer_;

  void* mapped_data_;

 public:
  StagingBuffer(
      Context* context_p,
      const vkapi::ScalarType dtype,
      const size_t numel)
      : context_p_(context_p),
        dtype_(dtype),
        vulkan_buffer_(context_p_->adapter_ptr()->vma().create_staging_buffer(
            element_size(dtype_) * numel)),
        mapped_data_(nullptr) {}

  StagingBuffer(const StagingBuffer&) = delete;
  StagingBuffer& operator=(const StagingBuffer&) = delete;

  StagingBuffer(StagingBuffer&&) = default;
  StagingBuffer& operator=(StagingBuffer&&) = default;

  ~StagingBuffer() {
    context_p_->register_buffer_cleanup(vulkan_buffer_);
  }

  inline vkapi::ScalarType dtype() const {
    return dtype_;
  }

  inline vkapi::VulkanBuffer& buffer() {
    return vulkan_buffer_;
  }

  inline void* data() {
    if (!mapped_data_) {
      mapped_data_ = vulkan_buffer_.allocation_info().pMappedData;
    }
    return mapped_data_;
  }

  inline size_t numel() {
    return nbytes() / element_size(dtype_);
  }

  inline size_t nbytes() {
    return vulkan_buffer_.mem_size();
  }

  inline void copy_from(const void* src, const size_t nbytes) {
    VK_CHECK_COND(nbytes <= this->nbytes());
    memcpy(data(), src, nbytes);
    vmaFlushAllocation(
        vulkan_buffer_.vma_allocator(),
        vulkan_buffer_.allocation(),
        0u,
        VK_WHOLE_SIZE);
  }

  template <typename SRC_T, typename DST_T>
  void cast_and_copy_from(const SRC_T* src, const size_t numel) {
    VK_CHECK_COND(numel <= this->numel());
    DST_T* dst = reinterpret_cast<DST_T*>(data());
    for (size_t i = 0; i < numel; ++i) {
      dst[i] = static_cast<DST_T>(src[i]);
    }
  }

  inline void copy_to(void* dst, const size_t nbytes) {
    VK_CHECK_COND(nbytes <= this->nbytes());
    vmaInvalidateAllocation(
        vulkan_buffer_.vma_allocator(),
        vulkan_buffer_.allocation(),
        0u,
        VK_WHOLE_SIZE);
    memcpy(dst, data(), nbytes);
  }

  template <typename SRC_T, typename DST_T>
  void cast_and_copy_to(DST_T* dst, const size_t numel) {
    VK_CHECK_COND(numel <= this->numel());
    const SRC_T* src = reinterpret_cast<const SRC_T*>(data());
    for (size_t i = 0; i < numel; ++i) {
      dst[i] = static_cast<DST_T>(src[i]);
    }
  }

  inline void set_staging_zeros() {
    memset(data(), 0, nbytes());
  }
};

} // namespace api
} // namespace vkcompute
