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

#include <executorch/backends/vulkan/runtime/utils/VecUtils.h>

#include <executorch/backends/vulkan/runtime/vk_api/Types.h>

#include <mutex>
#include <unordered_map>

namespace vkcompute {
namespace vkapi {

class ShaderLayout final {
 public:
  using Signature = std::vector<VkDescriptorType>;

  explicit ShaderLayout(VkDevice, const Signature&);

  ShaderLayout(const ShaderLayout&) = delete;
  ShaderLayout& operator=(const ShaderLayout&) = delete;

  ShaderLayout(ShaderLayout&&) noexcept;
  ShaderLayout& operator=(ShaderLayout&&) = delete;

  ~ShaderLayout();

 private:
  VkDevice device_;
  VkDescriptorSetLayout handle_;

 public:
  VkDescriptorSetLayout handle() const {
    return handle_;
  }

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ShaderLayout& lhs, ShaderLayout& rhs) noexcept;
};

struct ShaderInfo final {
  struct {
    const uint32_t* bin = nullptr;
    uint32_t size = 0u;
  } src_code;

  std::string kernel_name{""};
  ShaderLayout::Signature kernel_layout{};

  // Shader Metadata
  utils::uvec3 out_tile_size{1u, 1u, 1u};

  explicit ShaderInfo();

  explicit ShaderInfo(
      std::string,
      const uint32_t*,
      const uint32_t,
      std::vector<VkDescriptorType>,
      const utils::uvec3 tile_size);

  operator bool() const {
    return src_code.bin != nullptr;
  };
};

bool operator==(const ShaderInfo& _1, const ShaderInfo& _2);

class ShaderModule final {
 public:
  explicit ShaderModule(VkDevice device, const ShaderInfo& source);

  ShaderModule(const ShaderModule&) = delete;
  ShaderModule& operator=(const ShaderModule&) = delete;

  ShaderModule(ShaderModule&&) noexcept;
  ShaderModule& operator=(ShaderModule&&) = delete;

  ~ShaderModule();

 private:
  VkDevice device_;
  VkShaderModule handle_;

 public:
  inline VkShaderModule handle() const {
    return handle_;
  }

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ShaderModule& lhs, ShaderModule& rhs) noexcept;
};

class ShaderLayoutCache final {
 public:
  explicit ShaderLayoutCache(VkDevice device);

  ShaderLayoutCache(const ShaderLayoutCache&) = delete;
  ShaderLayoutCache& operator=(const ShaderLayoutCache&) = delete;

  ShaderLayoutCache(ShaderLayoutCache&&) noexcept;
  ShaderLayoutCache& operator=(ShaderLayoutCache&&) = delete;

  ~ShaderLayoutCache();

  using Key = ShaderLayout::Signature;
  using Value = ShaderLayout;

  struct Hasher {
    inline size_t operator()(const ShaderLayout::Signature& signature) const {
      size_t hashed = 0u;

      for (const VkDescriptorType type : signature) {
        hashed =
            utils::hash_combine(hashed, std::hash<VkDescriptorType>()(type));
      }

      return hashed;
    }
  };

 private:
  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  VkDescriptorSetLayout retrieve(const Key&);
  void purge();
};

class ShaderCache final {
 public:
  explicit ShaderCache(VkDevice device);

  ShaderCache(const ShaderCache&) = delete;
  ShaderCache& operator=(const ShaderCache&) = delete;

  ShaderCache(ShaderCache&&) noexcept;
  ShaderCache& operator=(ShaderCache&&) = delete;

  ~ShaderCache();

  using Key = ShaderInfo;
  using Value = ShaderModule;

  struct Hasher {
    inline size_t operator()(const ShaderInfo& source) const {
      size_t seed = 0;
      seed = utils::hash_combine(
          seed, std::hash<const uint32_t*>()(source.src_code.bin));
      seed = utils::hash_combine(
          seed, std::hash<uint32_t>()(source.src_code.size));

      return seed;
    }
  };

 private:
  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  VkShaderModule retrieve(const Key&);
  void purge();
};

} // namespace vkapi
} // namespace vkcompute

inline bool operator==(
    const VkDescriptorSetLayoutBinding& _1,
    const VkDescriptorSetLayoutBinding& _2) {
  return (
      _1.binding == _2.binding && _1.descriptorType == _2.descriptorType &&
      _1.descriptorCount == _2.descriptorCount &&
      _1.stageFlags == _2.stageFlags &&
      _1.pImmutableSamplers == _2.pImmutableSamplers);
}
