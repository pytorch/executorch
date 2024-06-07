/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <functional>

#include <executorch/backends/vulkan/runtime/api/vk_api.h>

#include <executorch/backends/vulkan/runtime/api/Adapter.h>
#include <executorch/backends/vulkan/runtime/api/Command.h>
#include <executorch/backends/vulkan/runtime/api/Pipeline.h>

#ifndef VULKAN_QUERY_POOL_SIZE
#define VULKAN_QUERY_POOL_SIZE 4096u
#endif

namespace vkcompute {
namespace api {

struct QueryPoolConfig final {
  uint32_t max_query_count = VULKAN_QUERY_POOL_SIZE;
  uint32_t initial_reserve_size = 256u;
};

struct ShaderDuration final {
  uint32_t idx;

  // Execution Properties
  std::string kernel_name;
  VkExtent3D global_workgroup_size;
  VkExtent3D local_workgroup_size;

  // Query indexes
  uint32_t start_query_idx;
  uint32_t end_query_idx;

  // Timings
  uint64_t start_time_ns;
  uint64_t end_time_ns;
  uint64_t execution_duration_ns;
};

class QueryPool final {
  // Configuration
  QueryPoolConfig config_;
  uint64_t ns_per_tick_;

  // Vulkan handles
  VkDevice device_;
  VkQueryPool querypool_;

  // Internal State
  uint32_t num_queries_;
  std::vector<ShaderDuration> shader_durations_;

  std::mutex mutex_;

 public:
  explicit QueryPool(const QueryPoolConfig&, const Adapter* adapter_p);

  QueryPool(const QueryPool&) = delete;
  QueryPool& operator=(const QueryPool&) = delete;

  QueryPool(QueryPool&&) = delete;
  QueryPool& operator=(QueryPool&&) = delete;

  ~QueryPool();

  void initialize(const Adapter* adapter_p);

 private:
  size_t write_timestamp(const CommandBuffer&);

 public:
  void reset_querypool(const CommandBuffer&);

  void reset_state();

  void shader_profile_begin(
      const CommandBuffer&,
      const std::string&,
      const VkExtent3D,
      const VkExtent3D);

  void shader_profile_end(const CommandBuffer&);

  void extract_results();

  std::string generate_string_report();
  void print_results();

  operator bool() const {
    return querypool_ != VK_NULL_HANDLE;
  }
};

} // namespace api
} // namespace vkcompute
