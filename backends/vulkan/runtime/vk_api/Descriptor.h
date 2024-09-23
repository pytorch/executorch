/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#include <executorch/backends/vulkan/runtime/vk_api/Shader.h>

#include <executorch/backends/vulkan/runtime/vk_api/memory/Buffer.h>
#include <executorch/backends/vulkan/runtime/vk_api/memory/Image.h>

#include <unordered_map>

namespace vkcompute {
namespace vkapi {

/*
 * Stores the binding information of a Vulkan Buffer so that the buffer can be
 * bound at a later time. This struct should only be used if the buffer to be
 * bound is guaranteed to be active at the time of binding.
 */
struct BufferBindInfo final {
  VkBuffer handle;
  VkDeviceSize offset;
  VkDeviceSize range;

  BufferBindInfo();
  BufferBindInfo(const VulkanBuffer& buffer_p);
};

struct ParamsBindList final {
  std::vector<BufferBindInfo> bind_infos;

  ParamsBindList() = default;
  ParamsBindList(std::initializer_list<const BufferBindInfo> init_list);

  void append(const BufferBindInfo& bind_info);
  void append(const ParamsBindList& other);
};

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
  DescriptorSet& bind(const uint32_t, const BufferBindInfo&);
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
  uint32_t descriptor_pool_max_sets;
  // DescriptorCounts by type
  uint32_t descriptor_uniform_buffer_count;
  uint32_t descriptor_storage_buffer_count;
  uint32_t descriptor_combined_sampler_count;
  uint32_t descriptor_storage_image_count;
  // Pile size for pre-allocating descriptor sets
  uint32_t descriptor_pile_sizes;
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

} // namespace vkapi
} // namespace vkcompute
