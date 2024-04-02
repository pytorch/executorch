/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/api/vk_api.h>

#include <executorch/backends/vulkan/runtime/api/Resource.h>
#include <executorch/backends/vulkan/runtime/api/Shader.h>

#include <unordered_map>

namespace vkcompute {
namespace api {

class DescriptorSet final {
 public:
  explicit DescriptorSet(VkDevice, VkDescriptorSet, ShaderLayout::Signature);

  DescriptorSet(const DescriptorSet&) = delete;
  DescriptorSet& operator=(const DescriptorSet&) = delete;

  DescriptorSet(DescriptorSet&&) noexcept;
  DescriptorSet& operator=(DescriptorSet&&) noexcept;

  ~DescriptorSet() = default;

  struct ResourceBinding final {
    uint32_t binding_idx;
    VkDescriptorType descriptor_type;
    bool is_image;

    union {
      VkDescriptorBufferInfo buffer_info;
      VkDescriptorImageInfo image_info;
    } resource_info;
  };

 private:
  VkDevice device_;
  VkDescriptorSet handle_;
  ShaderLayout::Signature shader_layout_signature_;
  std::vector<ResourceBinding> bindings_;

 public:
  DescriptorSet& bind(const uint32_t, const VulkanBuffer&);
  DescriptorSet& bind(const uint32_t, const VulkanImage&);

  VkDescriptorSet get_bind_handle() const;

 private:
  void add_binding(const ResourceBinding& resource);
};

class DescriptorSetPile final {
 public:
  DescriptorSetPile(
      const uint32_t,
      VkDescriptorSetLayout,
      VkDevice,
      VkDescriptorPool);

  DescriptorSetPile(const DescriptorSetPile&) = delete;
  DescriptorSetPile& operator=(const DescriptorSetPile&) = delete;

  DescriptorSetPile(DescriptorSetPile&&) = default;
  DescriptorSetPile& operator=(DescriptorSetPile&&) = default;

  ~DescriptorSetPile() = default;

 private:
  uint32_t pile_size_;
  VkDescriptorSetLayout set_layout_;
  VkDevice device_;
  VkDescriptorPool pool_;
  std::vector<VkDescriptorSet> descriptors_;
  size_t in_use_;

 public:
  VkDescriptorSet get_descriptor_set();

 private:
  void allocate_new_batch();
};

struct DescriptorPoolConfig final {
  // Overall Pool capacity
  uint32_t descriptorPoolMaxSets;
  // DescriptorCounts by type
  uint32_t descriptorUniformBufferCount;
  uint32_t descriptorStorageBufferCount;
  uint32_t descriptorCombinedSamplerCount;
  uint32_t descriptorStorageImageCount;
  // Pile size for pre-allocating descriptor sets
  uint32_t descriptorPileSizes;
};

class DescriptorPool final {
 public:
  explicit DescriptorPool(VkDevice, const DescriptorPoolConfig&);

  DescriptorPool(const DescriptorPool&) = delete;
  DescriptorPool& operator=(const DescriptorPool&) = delete;

  DescriptorPool(DescriptorPool&&) = delete;
  DescriptorPool& operator=(DescriptorPool&&) = delete;

  ~DescriptorPool();

 private:
  VkDevice device_;
  VkDescriptorPool pool_;
  DescriptorPoolConfig config_;
  // New Descriptors
  std::mutex mutex_;
  std::unordered_map<VkDescriptorSetLayout, DescriptorSetPile> piles_;

 public:
  operator bool() const {
    return (pool_ != VK_NULL_HANDLE);
  }

  void init(const DescriptorPoolConfig& config);

  DescriptorSet get_descriptor_set(
      VkDescriptorSetLayout handle,
      const ShaderLayout::Signature& signature);

  void flush();
};

} // namespace api
} // namespace vkcompute
