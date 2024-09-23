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
  size_t nbytes_;
  vkapi::VulkanBuffer vulkan_buffer_;

 public:
  ParamsBuffer() : context_p_{nullptr}, vulkan_buffer_{} {}

  template <typename Block>
  ParamsBuffer(Context* context_p, const Block& block)
      : context_p_(context_p),
        nbytes_(sizeof(block)),
        vulkan_buffer_(
            context_p_->adapter_ptr()->vma().create_params_buffer(block)) {}

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
  void update(const Block& block) {
    if (sizeof(block) != nbytes_) {
      VK_THROW("Attempted to update ParamsBuffer with data of different size");
    }
    // Fill the uniform buffer with data in block
    {
      vkapi::MemoryMap mapping(vulkan_buffer_, vkapi::MemoryAccessType::WRITE);
      Block* data_ptr = mapping.template data<Block>();

      *data_ptr = block;
    }
  }
};

} // namespace api
} // namespace vkcompute
