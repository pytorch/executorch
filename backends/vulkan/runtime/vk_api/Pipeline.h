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

#include <mutex>
#include <unordered_map>

#define SV(x) ::vkcompute::vkapi::SpecVar(x)

namespace vkcompute {
namespace vkapi {

struct SpecVar final {
  enum class Type : uint8_t {
    FLOAT,
    INT,
    UINT,
    BOOL,
  };

  union Value {
    int32_t as_int32;
    uint32_t as_uint32;
    float as_float;
    bool as_bool;
  };

  Value value;
  Type type;

  SpecVar();
  SpecVar(const float val);
  SpecVar(const int32_t val);
  SpecVar(const uint32_t val);
  SpecVar(const bool val);

  uint32_t val_size() const;
  uint32_t val_offset() const;
};

bool operator==(const SpecVar& lhs, const SpecVar& rhs);

bool operator!=(const SpecVar& lhs, const SpecVar& rhs);

class SpecVarList final {
  std::vector<SpecVar> vars;

 public:
  SpecVarList();
  SpecVarList(std::initializer_list<SpecVar> init_list);

  inline const SpecVar& at(const size_t index) const {
    return vars.at(index);
  }

  inline const SpecVar* data() const {
    return vars.data();
  }

  inline uint32_t size() const {
    return utils::safe_downcast<uint32_t>(vars.size());
  }

  inline uint32_t data_nbytes() const {
    return vars.size() * sizeof(SpecVar);
  }

  void append(const SpecVarList& other);

  std::vector<VkSpecializationMapEntry> generate_map_entries() const;

  friend bool operator==(const SpecVarList& lhs, const SpecVarList& rhs);
};

bool operator==(const SpecVarList& lhs, const SpecVarList& rhs);

struct PipelineBarrier final {
  struct Stages final {
    VkPipelineStageFlags src;
    VkPipelineStageFlags dst;
  } stage;

  std::vector<BufferMemoryBarrier> buffers;
  std::vector<ImageMemoryBarrier> images;
  std::vector<VkBufferMemoryBarrier> buffer_barrier_handles;
  std::vector<VkImageMemoryBarrier> image_barrier_handles;

  inline operator bool() const {
    return (0u != stage.src) || (0u != stage.dst) || !buffers.empty() ||
        !images.empty();
  }
};

using PipelineStageFlags = uint8_t;

enum PipelineStage : PipelineStageFlags {
  NO_STAGE = 0u << 0u,
  COMPUTE = 1u << 0u,
  HOST = 1u << 1u,
  TRANSFER = 1u << 2u,
};

VkAccessFlags vk_access(const PipelineStageFlags, const MemoryAccessFlags);
VkPipelineStageFlags vk_stage(const PipelineStageFlags);
VkImageLayout vk_layout(const PipelineStageFlags, const MemoryAccessFlags);

class PipelineLayout final {
 public:
  explicit PipelineLayout(VkDevice, VkDescriptorSetLayout);

  PipelineLayout(const PipelineLayout&) = delete;
  PipelineLayout& operator=(const PipelineLayout&) = delete;

  PipelineLayout(PipelineLayout&&) noexcept;
  PipelineLayout& operator=(PipelineLayout&&) = delete;

  ~PipelineLayout();

 private:
  VkDevice device_;
  VkPipelineLayout handle_;

 public:
  VkPipelineLayout handle() const {
    return handle_;
  }

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(PipelineLayout& lhs, PipelineLayout& rhs) noexcept;
};

class ComputePipeline final {
 public:
  struct Descriptor final {
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader_module;
    SpecVarList specialization_constants;
  };

  explicit ComputePipeline(
      VkDevice device,
      const Descriptor& descriptor,
      VkPipelineCache pipeline_cache);

  ComputePipeline(const ComputePipeline&) = delete;
  ComputePipeline& operator=(const ComputePipeline&) = delete;

  ComputePipeline(ComputePipeline&&) noexcept;
  ComputePipeline& operator=(ComputePipeline&&) = delete;

  ~ComputePipeline();

 private:
  VkDevice device_;
  VkPipeline handle_;

 public:
  inline VkPipeline handle() const {
    return handle_;
  }

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ComputePipeline& lhs, ComputePipeline& rhs) noexcept;
};

class PipelineLayoutCache final {
 public:
  explicit PipelineLayoutCache(VkDevice device);

  PipelineLayoutCache(const PipelineLayoutCache&) = delete;
  PipelineLayoutCache& operator=(const PipelineLayoutCache&) = delete;

  PipelineLayoutCache(PipelineLayoutCache&&) noexcept;
  PipelineLayoutCache& operator=(PipelineLayoutCache&&) = delete;

  ~PipelineLayoutCache();

  using Key = VkDescriptorSetLayout;
  using Value = PipelineLayout;

  struct Hasher {
    inline size_t operator()(VkDescriptorSetLayout descriptor_layout) const {
      return std::hash<VkDescriptorSetLayout>()(descriptor_layout);
    }
  };

 private:
  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  VkPipelineLayout retrieve(const Key&);
  void purge();
};

class ComputePipelineCache final {
 public:
  explicit ComputePipelineCache(
      VkDevice device,
      const std::string& cache_data_path);

  ComputePipelineCache(const ComputePipelineCache&) = delete;
  ComputePipelineCache& operator=(const ComputePipelineCache&) = delete;

  ComputePipelineCache(ComputePipelineCache&&) noexcept;
  ComputePipelineCache& operator=(ComputePipelineCache&&) = delete;

  ~ComputePipelineCache();

  using Key = ComputePipeline::Descriptor;
  using Value = ComputePipeline;

  struct Hasher {
    inline size_t operator()(
        const ComputePipeline::Descriptor& descriptor) const {
      size_t seed = 0;
      seed = utils::hash_combine(
          seed, std::hash<VkPipelineLayout>()(descriptor.pipeline_layout));
      seed = utils::hash_combine(
          seed, std::hash<VkShaderModule>()(descriptor.shader_module));

      const SpecVarList& spec_vars = descriptor.specialization_constants;
      seed = utils::hash_combine(seed, std::hash<uint32_t>()(spec_vars.size()));

      for (int i = 0; i < spec_vars.size(); ++i) {
        const SpecVar& spec_var = spec_vars.at(i);
        size_t new_seed = 0;
        switch (spec_var.type) {
          case SpecVar::Type::FLOAT:
            new_seed = std::hash<float>()(spec_var.value.as_float);
            break;
          case SpecVar::Type::INT:
            new_seed = std::hash<int32_t>()(spec_var.value.as_int32);
            break;
          case SpecVar::Type::UINT:
            new_seed = std::hash<uint32_t>()(spec_var.value.as_uint32);
            break;
          case SpecVar::Type::BOOL:
            new_seed = std::hash<bool>()(spec_var.value.as_bool);
            break;
        }
        seed = utils::hash_combine(seed, new_seed);
      }

      return seed;
    }
  };

 private:
  std::vector<char> load_cache();
  void save_cache();

  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  VkPipelineCache pipeline_cache_;
  std::unordered_map<Key, Value, Hasher> cache_;
  const std::string cache_data_path_;

 public:
  VkPipeline retrieve(const Key&);
  void purge();
};

//
// Impl
//

} // namespace vkapi
} // namespace vkcompute
