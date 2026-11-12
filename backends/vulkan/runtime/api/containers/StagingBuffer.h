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

// IEEE half-precision conversions; float_to_half rounds to the nearest f16,
// ties to even.
float half_to_float(uint16_t h);
uint16_t float_to_half(float f);

vkapi::ScalarType get_staging_dtype(
    Context* context_p,
    vkapi::ScalarType dtype);

// Bytes a staging buffer needs for `numel` elements of staging dtype `dtype`.
size_t get_staging_nbytes(vkapi::ScalarType dtype, size_t numel);

class StagingBuffer final {
 private:
  Context* context_p_;
  vkapi::ScalarType dtype_;
  vkapi::VulkanBuffer vulkan_buffer_;
  // Memory another API reads or writes on the GPU; it has no host mapping.
  bool external_;

  void* mapped_data_;

 public:
  StagingBuffer(
      Context* context_p,
      const vkapi::ScalarType dtype,
      const size_t numel,
      const vkapi::CopyDirection direction);

  // Staging over external memory, e.g. a buffer that an app reads or writes,
  // which must hold `numel` elements.
  StagingBuffer(
      Context* context_p,
      const vkapi::ScalarType dtype,
      const size_t numel,
      vkapi::VulkanBuffer&& external_buffer);

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

  inline bool is_external() const {
    return external_;
  }

  inline void* data() {
    if (!mapped_data_) {
      VK_CHECK_COND(
          !external_, "an external staging buffer has no host mapping");
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
    vmaFlushAllocation(
        vulkan_buffer_.vma_allocator(),
        vulkan_buffer_.allocation(),
        0u,
        VK_WHOLE_SIZE);
  }

  void cast_half_to_float_and_copy_from(
      const uint16_t* src,
      const size_t numel);

  void cast_float_to_half_and_copy_to(uint16_t* dst, const size_t numel);

  inline void copy_to(void* dst, const size_t nbytes) {
    VK_CHECK_COND(nbytes <= this->nbytes());
    vmaInvalidateAllocation(
        vulkan_buffer_.vma_allocator(),
        vulkan_buffer_.allocation(),
        0u,
        nbytes);
    memcpy(dst, data(), nbytes);
  }

  template <typename SRC_T, typename DST_T>
  void cast_and_copy_to(DST_T* dst, const size_t numel) {
    VK_CHECK_COND(numel <= this->numel());
    vmaInvalidateAllocation(
        vulkan_buffer_.vma_allocator(),
        vulkan_buffer_.allocation(),
        0u,
        numel * sizeof(SRC_T));
    const SRC_T* src = reinterpret_cast<const SRC_T*>(data());
    for (size_t i = 0; i < numel; ++i) {
      dst[i] = static_cast<DST_T>(src[i]);
    }
  }

  inline void set_staging_zeros() {
    memset(data(), 0, nbytes());
  }

  template <typename T>
  T select_element_at_dim(
      const std::vector<int64_t>& sizes,
      const int64_t dim,
      const int64_t index) {
    int64_t stride = 1;
    for (size_t i = dim + 1; i < sizes.size(); ++i) {
      stride *= sizes[i];
    }
    const int64_t offset = index * stride;
    const T* typed_data = reinterpret_cast<const T*>(data());
    return typed_data[offset];
  }
};

} // namespace api
} // namespace vkcompute
