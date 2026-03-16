/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/vk_api/Pipeline.h>

#include <fstream>

namespace vkcompute {
namespace vkapi {

//
// Utility Functions
//

VkAccessFlags vk_access(
    const PipelineStageFlags stage,
    const MemoryAccessFlags access) {
  VkAccessFlags vk_access = 0u;

  if (access & MemoryAccessType::READ) {
    if (stage & PipelineStage::COMPUTE) {
      vk_access |= VK_ACCESS_SHADER_READ_BIT;
    }

    if (stage & PipelineStage::HOST) {
      vk_access |= VK_ACCESS_HOST_READ_BIT;
    }

    if (stage & PipelineStage::TRANSFER) {
      vk_access |= VK_ACCESS_TRANSFER_READ_BIT;
    }
  }

  if (access & MemoryAccessType::WRITE) {
    if (stage & PipelineStage::COMPUTE) {
      vk_access |= VK_ACCESS_SHADER_WRITE_BIT;
    }

    if (stage & PipelineStage::HOST) {
      vk_access |= VK_ACCESS_HOST_WRITE_BIT;
    }

    if (stage & PipelineStage::TRANSFER) {
      vk_access |= VK_ACCESS_TRANSFER_WRITE_BIT;
    }
  }

  return vk_access;
}

VkPipelineStageFlags vk_stage(const PipelineStageFlags stage) {
  VkPipelineStageFlags vk_stage = 0u;

  if (stage & PipelineStage::COMPUTE) {
    vk_stage |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }

  if (stage & PipelineStage::HOST) {
    vk_stage |= VK_PIPELINE_STAGE_HOST_BIT;
  }

  if (stage & PipelineStage::TRANSFER) {
    vk_stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;
  }

  return vk_stage;
}

VkImageLayout vk_layout(
    const PipelineStageFlags stage,
    const MemoryAccessFlags access) {
  switch (stage) {
    case PipelineStage::COMPUTE:
      switch (access) {
        case MemoryAccessType::READ:
          return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        default:
          return VK_IMAGE_LAYOUT_GENERAL;
      }
      break;
    case PipelineStage::TRANSFER:
      switch (access) {
        case MemoryAccessType::READ:
          return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        case MemoryAccessType::WRITE:
          return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        default:
          VK_THROW("Invalid memory access type for transfer stage!");
      }
      break;
    default:
      VK_THROW("Cannot determine appropriate image layout");
  }

  return VK_IMAGE_LAYOUT_UNDEFINED;
}

//
// SpecVar
//

SpecVar::SpecVar() : type(SpecVar::Type::INT) {
  value.as_int32 = 0;
}

SpecVar::SpecVar(const float val) : type(SpecVar::Type::FLOAT) {
  value.as_float = val;
}

SpecVar::SpecVar(const int32_t val) : type(SpecVar::Type::INT) {
  value.as_int32 = val;
}

SpecVar::SpecVar(const uint32_t val) : type(SpecVar::Type::UINT) {
  value.as_uint32 = val;
}

SpecVar::SpecVar(const bool val) : type(SpecVar::Type::BOOL) {
  value.as_bool = val;
}

uint32_t SpecVar::val_size() const {
  switch (type) {
    case SpecVar::Type::FLOAT:
      return sizeof(float);
    case SpecVar::Type::INT:
      return sizeof(int32_t);
    case SpecVar::Type::UINT:
      return sizeof(uint32_t);
    case SpecVar::Type::BOOL:
      return sizeof(bool);
  }
  return 4;
}

uint32_t SpecVar::val_offset() const {
  return utils::safe_downcast<uint32_t>(offsetof(SpecVar, value));
}

bool operator==(const SpecVar& lhs, const SpecVar& rhs) {
  if (lhs.type != rhs.type) {
    return false;
  }
  switch (lhs.type) {
    case SpecVar::Type::FLOAT:
      return lhs.value.as_float == rhs.value.as_float;
    case SpecVar::Type::INT:
      return lhs.value.as_int32 == rhs.value.as_int32;
    case SpecVar::Type::UINT:
      return lhs.value.as_uint32 == rhs.value.as_uint32;
    case SpecVar::Type::BOOL:
      return lhs.value.as_bool == rhs.value.as_bool;
  }
  return false;
}

bool operator!=(const SpecVar& lhs, const SpecVar& rhs) {
  return !(lhs == rhs);
}

SpecVarList::SpecVarList() {}

SpecVarList::SpecVarList(std::initializer_list<SpecVar> init_list) {
  vars.resize(init_list.size());
  std::copy(init_list.begin(), init_list.end(), vars.begin());
}

void SpecVarList::append(const SpecVarList& other) {
  vars.insert(vars.end(), other.vars.begin(), other.vars.end());
}

void SpecVarList::reserve(const size_t size) {
  vars.reserve(size);
}

void SpecVarList::append(const SpecVar& other) {
  vars.push_back(other);
}

std::vector<VkSpecializationMapEntry> SpecVarList::generate_map_entries()
    const {
  std::vector<VkSpecializationMapEntry> map_entries;
  map_entries.resize(vars.size());
  uint32_t cur_offset = 0u;
  for (uint32_t i = 0; i < vars.size(); ++i) {
    map_entries.at(i) = {
        i, cur_offset + vars.at(i).val_offset(), vars.at(i).val_size()};
    cur_offset += sizeof(SpecVar);
  }
  return map_entries;
}

bool operator==(const SpecVarList& lhs, const SpecVarList& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (uint32_t i = 0; i < lhs.size(); ++i) {
    if (lhs.vars.at(i) != rhs.vars.at(i)) {
      return false;
    }
  }
  return true;
}

//
// PipelineLayout
//

PipelineLayout::PipelineLayout(
    VkDevice device,
    VkDescriptorSetLayout descriptor_layout,
    const uint32_t push_constants_size)
    : device_(device), handle_{VK_NULL_HANDLE} {
  VkPushConstantRange pc_range{
      VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
      0u, // offset
      push_constants_size, // size
  };
  uint32_t num_push_constants = 0u;
  VkPushConstantRange* pc_ranges_ptr = nullptr;
  if (push_constants_size > 0u) {
    num_push_constants = 1u;
    pc_ranges_ptr = &pc_range;
  }

  const VkPipelineLayoutCreateInfo pipeline_layout_create_info{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      1u, // setLayoutCount
      &descriptor_layout, // pSetLayouts
      num_push_constants, // pushConstantRangeCount
      pc_ranges_ptr, // pPushConstantRanges
  };

  VK_CHECK(vkCreatePipelineLayout(
      device_, &pipeline_layout_create_info, nullptr, &handle_));
}

PipelineLayout::PipelineLayout(PipelineLayout&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

PipelineLayout::~PipelineLayout() {
  if (handle_ == VK_NULL_HANDLE) {
    return;
  }
  vkDestroyPipelineLayout(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(PipelineLayout& lhs, PipelineLayout& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkPipelineLayout tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// ComputePipeline
//

ComputePipeline::ComputePipeline(VkDevice device, VkPipeline handle)
    : device_{device}, handle_{handle} {}

ComputePipeline::ComputePipeline(
    VkDevice device,
    const ComputePipeline::Descriptor& descriptor,
    VkPipelineCache pipeline_cache)
    : device_(device), handle_{VK_NULL_HANDLE} {
  map_entries_ = descriptor.specialization_constants.generate_map_entries();

  const VkSpecializationInfo specialization_info{
      descriptor.specialization_constants.size(), // mapEntryCount
      map_entries_.data(), // pMapEntries
      descriptor.specialization_constants.data_nbytes(), // dataSize
      descriptor.specialization_constants.data(), // pData
  };

  const VkPipelineShaderStageCreateInfo shader_stage_create_info{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      VK_SHADER_STAGE_COMPUTE_BIT, // stage
      descriptor.shader_module, // module
      "main", // pName
      &specialization_info, // pSpecializationInfo
  };

  VkPipelineCreateFlags flags = 0u;
#if defined(VK_KHR_pipeline_executable_properties) && \
    defined(ETVK_INSPECT_PIPELINES)
  flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
      VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR | flags;
#endif // VK_KHR_pipeline_executable_properties && ETVK_INSPECT_PIPELINES

  const VkComputePipelineCreateInfo compute_pipeline_create_info{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // sType
      nullptr, // pNext
      flags, // flags
      shader_stage_create_info, // stage
      descriptor.pipeline_layout, // layout
      VK_NULL_HANDLE, // basePipelineHandle
      0u, // basePipelineIndex
  };

  VK_CHECK(vkCreateComputePipelines(
      device_,
      pipeline_cache,
      1u,
      &compute_pipeline_create_info,
      nullptr,
      &handle_));
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : device_(other.device_),
      handle_(other.handle_),
      map_entries_(std::move(other.map_entries_)) {
  other.handle_ = VK_NULL_HANDLE;
}

ComputePipeline::~ComputePipeline() {
  if (handle_ == VK_NULL_HANDLE) {
    return;
  }
  vkDestroyPipeline(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(ComputePipeline& lhs, ComputePipeline& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkPipeline tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

bool operator==(
    const ComputePipeline::Descriptor& _1,
    const ComputePipeline::Descriptor& _2) {
  return (
      _1.pipeline_layout == _2.pipeline_layout &&
      _1.shader_module == _2.shader_module &&
      _1.specialization_constants == _2.specialization_constants);
}

//
// PipelineLayoutCache
//

PipelineLayoutCache::PipelineLayoutCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

PipelineLayoutCache::PipelineLayoutCache(PipelineLayoutCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

PipelineLayoutCache::~PipelineLayoutCache() {
  purge();
}

VkPipelineLayout PipelineLayoutCache::retrieve(
    const VkDescriptorSetLayout layout,
    const uint32_t push_constants_size) {
  PipelineLayoutCache::Key key{layout, push_constants_size};
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_
             .insert(
                 {key,
                  PipelineLayoutCache::Value(
                      device_, layout, push_constants_size)})
             .first;
  }

  return it->second.handle();
}

void PipelineLayoutCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear();
}

//
// ComputePipelineCache
//

ComputePipelineCache::ComputePipelineCache(
    VkDevice device,
    const std::string& cache_data_path)
    : cache_mutex_{},
      device_(device),
      pipeline_cache_{VK_NULL_HANDLE},
      cache_{},
      cache_data_path_(cache_data_path) {
  VkPipelineCacheCreateInfo pipeline_cache_create_info{};

  auto buffer = load_cache();

  pipeline_cache_create_info = {
      VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      buffer.size(), // initialDataSize
      buffer.data(), // pInitialData
  };

  VK_CHECK(vkCreatePipelineCache(
      device, &pipeline_cache_create_info, nullptr, &pipeline_cache_));
}

ComputePipelineCache::ComputePipelineCache(
    ComputePipelineCache&& other) noexcept
    : cache_mutex_{},
      device_(other.device_),
      pipeline_cache_(other.pipeline_cache_),
      cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);

  other.pipeline_cache_ = VK_NULL_HANDLE;
}

ComputePipelineCache::~ComputePipelineCache() {
  purge();

  if (pipeline_cache_ == VK_NULL_HANDLE) {
    return;
  }

  vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
  pipeline_cache_ = VK_NULL_HANDLE;
}

bool ComputePipelineCache::contains(const ComputePipelineCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  return it != cache_.cend();
}

void ComputePipelineCache::create_pipelines(
    const std::unordered_set<Key, Hasher>& descriptors) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  // Filter out descriptors already in cache to avoid creating duplicate
  // pipelines and to ensure correct indexing between created pipelines and
  // cache insertion.
  std::vector<Key> keys_to_create;
  keys_to_create.reserve(descriptors.size());
  for (const auto& key : descriptors) {
    if (cache_.find(key) == cache_.cend()) {
      keys_to_create.push_back(key);
    }
  }

  if (keys_to_create.empty()) {
    return;
  }

  const auto num_pipelines = keys_to_create.size();
  std::vector<VkPipeline> pipelines(num_pipelines);

  std::vector<std::vector<VkSpecializationMapEntry>> map_entries;
  map_entries.reserve(num_pipelines);

  std::vector<VkSpecializationInfo> specialization_infos;
  specialization_infos.reserve(num_pipelines);

  std::vector<VkPipelineShaderStageCreateInfo> shader_stage_create_infos;
  shader_stage_create_infos.reserve(num_pipelines);

  std::vector<VkComputePipelineCreateInfo> create_infos;
  create_infos.reserve(num_pipelines);

  for (const auto& key : keys_to_create) {
    map_entries.push_back(key.specialization_constants.generate_map_entries());

    specialization_infos.push_back(VkSpecializationInfo{
        key.specialization_constants.size(), // mapEntryCount
        map_entries.back().data(), // pMapEntries
        key.specialization_constants.data_nbytes(), // dataSize
        key.specialization_constants.data(), // pData
    });

    shader_stage_create_infos.push_back(VkPipelineShaderStageCreateInfo{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // sType
        nullptr, // pNext
        0u, // flags
        VK_SHADER_STAGE_COMPUTE_BIT, // stage
        key.shader_module, // module
        "main", // pName
        &specialization_infos.back(), // pSpecializationInfo
    });

    create_infos.push_back(VkComputePipelineCreateInfo{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // sType
        nullptr, // pNext
        0u, // flags
        shader_stage_create_infos.back(), // stage
        key.pipeline_layout, // layout
        VK_NULL_HANDLE, // basePipelineHandle
        0u, // basePipelineIndex
    });
  }

  VK_CHECK(vkCreateComputePipelines(
      device_,
      pipeline_cache_,
      create_infos.size(),
      create_infos.data(),
      nullptr,
      pipelines.data()));

  for (size_t i = 0; i < keys_to_create.size(); ++i) {
    cache_.insert(
        {keys_to_create[i],
         ComputePipelineCache::Value(device_, pipelines[i])});
  }
}

VkPipeline ComputePipelineCache::retrieve(
    const ComputePipelineCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (it == cache_.cend()) {
    it = cache_
             .insert(
                 {key,
                  ComputePipelineCache::Value(device_, key, pipeline_cache_)})
             .first;
  }
  return it->second.handle();
}

void ComputePipelineCache::purge() {
  cache_.clear();
}

std::vector<char> ComputePipelineCache::load_cache() {
  // No optimization if path is unspecified
  if (cache_data_path_.empty()) {
    return {};
  }

  // Return if file doesn't exist; this is expected on first model-load
  std::ifstream file(cache_data_path_, std::ios::binary | std::ios::ate);
  if (file.fail()) {
    return {};
  }

  auto size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  file.read(buffer.data(), size);

  return buffer;
}

void ComputePipelineCache::save_cache() {
  // No optimization if path is unspecified
  if (cache_data_path_.empty()) {
    return;
  }

  // Return if file exists; the cache is already saved
  std::ifstream ifile(cache_data_path_);
  if (ifile.good()) {
    return;
  }

  size_t size{};
  vkGetPipelineCacheData(device_, pipeline_cache_, &size, nullptr);

  std::vector<char> buffer(size);
  vkGetPipelineCacheData(device_, pipeline_cache_, &size, buffer.data());

  std::ofstream file(cache_data_path_, std::ios::binary);
  file.write(buffer.data(), buffer.size());
}

} // namespace vkapi
} // namespace vkcompute
