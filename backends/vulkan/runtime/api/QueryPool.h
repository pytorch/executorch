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
 public:
  explicit QueryPool(const QueryPoolConfig&, const Adapter* adapter_p);

  QueryPool(const QueryPool&) = delete;
  QueryPool& operator=(const QueryPool&) = delete;

  QueryPool(QueryPool&&) = delete;
  QueryPool& operator=(QueryPool&&) = delete;

  ~QueryPool();

 private:
  std::mutex mutex_;

  VkDevice device_;
  QueryPoolConfig config_;

  VkQueryPool querypool_;

  std::vector<std::vector<ShaderDuration>> shader_logs_;
  size_t in_use_;

  /** Total number of entries in shader logs from before most recent reset */
  size_t previous_shader_count_;

  /**
   * Indicates whether there are new log entries in the shader log since the
   * most recent call to extract_results()
   */
  bool results_pending_;

 private:
  size_t write_timestamp(const CommandBuffer&);

  std::string generate_string_report();

  /** Most recent shader log since the last time the QueryPool was reset */
  inline std::vector<ShaderDuration>& shader_log() {
    return shader_logs_[shader_logs_.size() - 1];
  }

  /** Total number of entries in all shader logs, but without locking mutex */
  size_t shader_logs_entry_count_thread_unsafe();

 public:
  inline bool is_enabled() const {
    return VK_NULL_HANDLE != querypool_;
  }

  void reset(const CommandBuffer&);

  uint32_t shader_profile_begin(
      const CommandBuffer&,
      const std::string&,
      const VkExtent3D,
      const VkExtent3D);
  void shader_profile_end(const CommandBuffer&, const uint32_t);

  void extract_results();
  void print_results();
  uint64_t get_total_op_ns(const std::string& op_name);
  uint64_t ns_per_tick_;
  void shader_log_for_each(std::function<void(const ShaderDuration&)> fn);
  /**
   * query_index is what number entry across all of the QueryPool's shader logs
   * is being queried, regardless of resets. This may be different than
   * ShaderDuration's idx field, which is what number entry it is since the last
   * reset before it was added to the shader logs.
   */
  std::tuple<std::string, uint64_t> get_shader_name_and_execution_duration_ns(
      size_t query_index);
  /** Total number of entries in all shader logs */
  size_t shader_logs_entry_count();
};

} // namespace api
} // namespace vkcompute
