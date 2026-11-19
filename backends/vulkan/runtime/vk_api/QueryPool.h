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

#include <executorch/backends/vulkan/runtime/vk_api/Adapter.h>
#include <executorch/backends/vulkan/runtime/vk_api/Command.h>
#include <executorch/backends/vulkan/runtime/vk_api/Pipeline.h>

#include <cstdint>
#include <functional>

#ifndef VULKAN_QUERY_POOL_SIZE
#define VULKAN_QUERY_POOL_SIZE 131072u
#endif

namespace vkcompute {
namespace vkapi {

struct ShaderMetadata final {
  uint32_t gwg[3];
  uint32_t lwg[3];
};

// Each profiled dispatch records three bottom-of-pipe timestamps. Each marks
// when all previously recorded GPU work had finished:
//   previous_work_finish: before the dispatch's pipeline barrier.
//   barrier_finish: after the barrier, before the dispatch. Whether this waits
//     for the barrier's cache maintenance is driver-dependent.
//   shader_exec_finish: after the dispatch.
// The barrier interval is the cost of making earlier writes visible; the shader
// execution interval is the dispatch itself.

struct ShaderResult final {
  const std::string kernel_name;
  const uint32_t dispatch_id;
  const uint64_t previous_work_finish_ns;
  const uint64_t barrier_finish_ns;
  const uint64_t shader_exec_finish_ns;
  const ShaderMetadata metadata;
};

struct ShaderTimestamp final {
  std::string kernel_name;
  uint64_t previous_work_finish_ticks;
  uint64_t barrier_finish_ticks;
  uint64_t shader_exec_finish_ticks;
  ShaderMetadata metadata;
};

// Previous work finish, barrier finish, and shader execution finish.
constexpr uint32_t kQueriesPerDispatch = 3u;

// Counts entries whose timestamps are out of order: within an entry, or between
// an entry's shader execution finish and the next entry's previous work finish.
// Dispatches serialized by full barriers never violate this order.
size_t count_order_violations(const std::vector<ShaderTimestamp>& timestamps);

struct QueryPoolConfig final {
  uint32_t max_query_count = VULKAN_QUERY_POOL_SIZE;
  uint32_t initial_reserve_size = 256u;
};

struct ShaderDuration final {
  uint32_t idx;

  // Execution Properties
  uint32_t dispatch_id;
  std::string kernel_name;
  GlobalWorkGrid gwg;
  LocalWorkGroup lwg;

  // Query indexes
  uint32_t previous_work_finish_query_idx;
  uint32_t barrier_finish_query_idx;
  uint32_t shader_exec_finish_query_idx;

  // Timings
  uint64_t previous_work_finish_ns;
  uint64_t barrier_finish_ns;
  uint64_t shader_exec_finish_ns;
  uint64_t barrier_duration_ns;
  uint64_t shader_exec_duration_ns;
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
  size_t num_order_violations_;

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

  // Converts raw query results into per-dispatch timestamps. A dispatch without
  // a barrier finish timestamp reports its previous work finish instead.
  std::vector<ShaderTimestamp> to_shader_timestamps(
      const std::vector<uint64_t>& ticks) const;

 public:
  // Starts a new capture by clearing recorded entries. Each dispatch resets the
  // queries it uses when it records them.
  void reset_querypool();

  // Records a reset of every query in the pool.
  void reset_all(const CommandBuffer&);

  void reset_state();

  // Opens a profiling entry for a dispatch and records its previous work
  // finish timestamp.
  void record_previous_work_finish(
      const CommandBuffer&,
      const uint32_t,
      const std::string&,
      const GlobalWorkGrid&,
      const LocalWorkGroup&);

  // Records the open entry's barrier finish timestamp. No-op without an open
  // entry.
  void record_barrier_finish(const CommandBuffer&);

  // Records the open entry's shader execution finish timestamp and closes it.
  void record_shader_exec_finish(const CommandBuffer&);

  void extract_results();

  std::vector<ShaderResult> get_shader_timestamp_data();
  // Read after fence completion; unfinished or unavailable captures return no
  // samples and never wait for GPU work.
  std::vector<ShaderTimestamp> get_shader_timestamp_ticks();
  void print_results(const bool tsv_format = false);
  // Order violations found by the last extract_results().
  size_t num_order_violations() const {
    return num_order_violations_;
  }
  unsigned long get_total_shader_ns(std::string kernel_name);
  unsigned long get_mean_shader_ns(std::string kernel_name);

  operator bool() const {
    return querypool_ != VK_NULL_HANDLE;
  }

 private:
  std::string generate_string_report();
  std::string generate_tsv_string_report();
};

} // namespace vkapi
} // namespace vkcompute
