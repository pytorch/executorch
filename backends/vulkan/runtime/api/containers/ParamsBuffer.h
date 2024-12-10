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

class ParamsBuffer final {
 private:
  Context* context_p_;
  vkapi::VulkanBuffer vulkan_buffer_;

 public:
  ParamsBuffer() : context_p_{nullptr}, vulkan_buffer_{} {}

  template <typename Block>
  ParamsBuffer(Context* context_p, const Block& block)
      : context_p_(context_p),
        vulkan_buffer_(
            context_p_->adapter_ptr()->vma().create_params_buffer(block)) {}

  template <typename Block>
  ParamsBuffer(Context* context_p, const VkDeviceSize nbytes)
      : context_p_(context_p),
        vulkan_buffer_(
            context_p_->adapter_ptr()->vma().create_uniform_buffer(nbytes)) {}

  ParamsBuffer(const ParamsBuffer&);
  ParamsBuffer& operator=(const ParamsBuffer&);

  ParamsBuffer(ParamsBuffer&&) = default;
  ParamsBuffer& operator=(ParamsBuffer&&) = default;

  ~ParamsBuffer() {
    if (vulkan_buffer_) {
      context_p_->register_buffer_cleanup(vulkan_buffer_);
    }
  }

  const vkapi::VulkanBuffer& buffer() const {
    return vulkan_buffer_;
  }

  template <typename Block>
  void update(const Block& block, const uint32_t offset = 0) {
    // Fill the uniform buffer with data in block
    {
      vkapi::MemoryMap mapping(vulkan_buffer_, vkapi::kWrite);
      Block* data_ptr = mapping.template data<Block>(offset);

      *data_ptr = block;
    }
  }

  template <typename T>
  T read() const {
    T val;
    if (sizeof(val) != vulkan_buffer_.mem_size()) {
      VK_THROW(
          "Attempted to store value from ParamsBuffer to type of different size");
    }
    // Read value from uniform buffer and store in val
    {
      vkapi::MemoryMap mapping(vulkan_buffer_, vkapi::kRead);
      T* data_ptr = mapping.template data<T>();

      val = *data_ptr;
    }
    return val;
  }
};

} // namespace api
} // namespace vkcompute
