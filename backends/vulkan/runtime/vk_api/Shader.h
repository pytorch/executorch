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

#include <executorch/backends/vulkan/runtime/vk_api/DispatchGrid.h>

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

enum class SpirvCapability : uint32_t {
  MATRIX = 0,
  SHADER = 1,
  FLOAT16 = 9,
  FLOAT64 = 10,
  INT64 = 11,
  INT16 = 22,
  IMAGE_GATHER_EXTENDED = 25,
  STORAGE_IMAGE_MULTISAMPLE = 27,
  UNIFORM_BUFFER_ARRAY_DYNAMIC_INDEXING = 28,
  SAMPLED_IMAGE_ARRAY_DYNAMIC_INDEXING = 29,
  STORAGE_BUFFER_ARRAY_DYNAMIC_INDEXING = 30,
  STORAGE_IMAGE_ARRAY_DYNAMIC_INDEXING = 31,
  IMAGE_CUBE_ARRAY = 34,
  INT8 = 39,
  MIN_LOD = 42,
  SAMPLED_1D = 43,
  IMAGE_1D = 44,
  SAMPLED_CUBE_ARRAY = 45,
  SAMPLED_BUFFER = 46,
  IMAGE_BUFFER = 47,
  STORAGE_IMAGE_EXTENDED_FORMATS = 49,
  IMAGE_QUERY = 50,
  STORAGE_IMAGE_READ_WITHOUT_FORMAT = 55,
  STORAGE_IMAGE_WRITE_WITHOUT_FORMAT = 56,
  GROUP_NON_UNIFORM = 61,
  GROUP_NON_UNIFORM_VOTE = 62,
  GROUP_NON_UNIFORM_ARITHMETIC = 63,
  GROUP_NON_UNIFORM_BALLOT = 64,
  GROUP_NON_UNIFORM_SHUFFLE = 65,
  GROUP_NON_UNIFORM_SHUFFLE_RELATIVE = 66,
  GROUP_NON_UNIFORM_CLUSTERED = 67,
  GROUP_NON_UNIFORM_QUAD = 68,
  STORAGE_BUFFER_16_BIT_ACCESS = 4433,
  UNIFORM_AND_STORAGE_BUFFER_16_BIT_ACCESS = 4434,
  STORAGE_PUSH_CONSTANT_16 = 4435,
  STORAGE_INPUT_OUTPUT_16 = 4436,
  STORAGE_BUFFER_8_BIT_ACCESS = 4448,
  UNIFORM_AND_STORAGE_BUFFER_8_BIT_ACCESS = 4449,
  STORAGE_PUSH_CONSTANT_8 = 4450,
  COOPERATIVE_MATRIX_REDUCTIONS_NV = 5430,
  COOPERATIVE_MATRIX_CONVERSIONS_NV = 5431,
  COOPERATIVE_MATRIX_PER_ELEMENT_OPERATIONS_NV = 5432,
  COOPERATIVE_MATRIX_TENSOR_ADDRESSING_NV = 5433,
  COOPERATIVE_MATRIX_BLOCK_LOADS_NV = 5434,
  DOT_PRODUCT_INPUT_ALL = 6016,
  DOT_PRODUCT_INPUT_4X8_BIT = 6017,
  DOT_PRODUCT_INPUT_4X8_BIT_PACKED = 6018,
  DOT_PRODUCT = 6019,
  COOPERATIVE_MATRIX_KHR = 6022,
};

struct ShaderInfo final {
  struct {
    const uint32_t* bin = nullptr;
    uint32_t size = 0u;
  } src_code;

  std::string kernel_name{""};
  ShaderLayout::Signature kernel_layout{};

  // Shader Metadata
  LocalWorkGroup out_tile_size{1u, 1u, 1u};
  bool requires_shader_int16 = false;
  bool requires_16bit_storage = false;
  bool requires_8bit_storage = false;
  bool requires_integer_dot_product = false;
  bool requires_shader_int64 = false;
  bool requires_shader_float64 = false;

  // Subgroup size requirement declared in the shader's yaml.
  //   0  = no requirement (default)
  //  >0  = literal fixed size; pipeline is pinned to this subgroup size.
  // Sourced from the yaml's `SUBGROUP_SIZE` template parameter — single
  // source of truth shared with GLSL ${SUBGROUP_SIZE} substitution.
  uint32_t required_subgroup_size = 0u;
  std::vector<uint32_t> required_capabilities;

  explicit ShaderInfo();

  explicit ShaderInfo(
      std::string,
      const uint32_t*,
      const uint32_t,
      std::vector<VkDescriptorType>,
      const utils::uvec3 tile_size,
      const bool requires_shader_int16_ext,
      const bool requires_16bit_storage_ext,
      const bool requires_8bit_storage_ext,
      const bool requires_integer_dot_product_ext,
      const bool requires_shader_int64_ext,
      const bool requires_shader_float64_ext,
      const uint32_t required_subgroup_size_arg = 0u,
      std::vector<uint32_t> required_capabilities_arg = {});

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
