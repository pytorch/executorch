/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-BadImplicitCast

#include <executorch/backends/vulkan/runtime/utils/VecUtils.h>

#include <executorch/backends/vulkan/runtime/vk_api/QueryPool.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>

namespace vkcompute {
namespace vkapi {

namespace {

// On Mali gpus timestamp_period seems to return 0.
// For some reason when 52.08 is used op runtimes seem to make more sense
// TODO: Figure out what is special about 52.08
constexpr int64_t kDefaultNsPerTick = 52; // lround(52.08f);

} // namespace

#define EARLY_RETURN_IF_UNINITIALIZED() \
  if (VK_NULL_HANDLE == querypool_) {   \
    return;                             \
  }

QueryPool::QueryPool(const QueryPoolConfig& config, const Adapter* adapter_p)
    : config_(config),
      ns_per_tick_(1u),
      device_(VK_NULL_HANDLE),
      querypool_(VK_NULL_HANDLE),
      num_queries_(0u),
      shader_durations_(0),
      mutex_{} {
  initialize(adapter_p);
}

QueryPool::~QueryPool() {
  EARLY_RETURN_IF_UNINITIALIZED();
  vkDestroyQueryPool(device_, querypool_, nullptr);
}

void QueryPool::initialize(const Adapter* adapter_p) {
  // No-op if adapter_p is nullptr or querypool is already created
  if (!adapter_p || querypool_ != VK_NULL_HANDLE) {
    return;
  }

  device_ = adapter_p->device_handle();

  ns_per_tick_ = std::lround(adapter_p->timestamp_period());
  ns_per_tick_ = (ns_per_tick_ == 0) ? kDefaultNsPerTick : ns_per_tick_;

  shader_durations_.reserve(config_.initial_reserve_size);

  const VkQueryPoolCreateInfo info{
      VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      VK_QUERY_TYPE_TIMESTAMP, // queryType
      config_.max_query_count, // queryCount
      0u, // pipelineStatistics
  };

  VK_CHECK(vkCreateQueryPool(device_, &info, nullptr, &querypool_));
}

size_t QueryPool::write_timestamp(const CommandBuffer& cmd) {
  VK_CHECK_COND(
      num_queries_ < config_.max_query_count,
      "Vulkan QueryPool: Exceeded the maximum number of queries "
      "allowed by the queryPool (",
      config_.max_query_count,
      ")!");

  cmd.write_timestamp(querypool_, num_queries_++);
  return num_queries_ - 1;
}

void QueryPool::reset_querypool(const CommandBuffer& cmd) {
  EARLY_RETURN_IF_UNINITIALIZED();
  std::lock_guard<std::mutex> lock(mutex_);

  cmd.reset_querypool(querypool_, 0u, config_.max_query_count);
  reset_state();
}

void QueryPool::reset_state() {
  num_queries_ = 0u;
  shader_durations_.clear();
}

void QueryPool::shader_profile_begin(
    const CommandBuffer& cmd,
    const uint32_t dispatch_id,
    const std::string& kernel_name,
    const VkExtent3D global_workgroup_size,
    const VkExtent3D local_workgroup_size) {
  EARLY_RETURN_IF_UNINITIALIZED();
  std::lock_guard<std::mutex> lock(mutex_);

  uint32_t query_idx = write_timestamp(cmd);

  ShaderDuration log_entry{
      utils::safe_downcast<uint32_t>(shader_durations_.size()),
      // Execution Properties
      dispatch_id,
      kernel_name,
      global_workgroup_size,
      local_workgroup_size,
      // Query indexes
      query_idx, // start query idx
      UINT32_MAX, // end query idx
      // Timings
      0u, // start time
      0u, // end time
      0u, // duration
  };

  shader_durations_.emplace_back(log_entry);
}

void QueryPool::shader_profile_end(const CommandBuffer& cmd) {
  EARLY_RETURN_IF_UNINITIALIZED();
  std::lock_guard<std::mutex> lock(mutex_);

  size_t query_idx = write_timestamp(cmd);
  shader_durations_.back().end_query_idx = query_idx;
}

void QueryPool::extract_results() {
  EARLY_RETURN_IF_UNINITIALIZED();
  std::lock_guard<std::mutex> lock(mutex_);

  const VkQueryResultFlags flags = VK_QUERY_RESULT_64_BIT;

  std::vector<uint64_t> query_data;
  query_data.resize(num_queries_);

  VK_CHECK(vkGetQueryPoolResults(
      device_,
      querypool_,
      0u, // firstQuery
      num_queries_, // queryCount
      sizeof(uint64_t) * num_queries_, // dataSize
      query_data.data(), // pData
      sizeof(uint64_t), // stride
      flags)); // flags

  for (ShaderDuration& entry : shader_durations_) {
    entry.start_time_ns = query_data.at(entry.start_query_idx) * ns_per_tick_;
    entry.end_time_ns = query_data.at(entry.end_query_idx) * ns_per_tick_;
    entry.execution_duration_ns = entry.end_time_ns - entry.start_time_ns;
  }
}

std::ostream& operator<<(std::ostream& os, const VkExtent3D& extents) {
  os << "{" << extents.width << ", " << extents.height << ", " << extents.depth
     << "}";
  return os;
}

std::string stringize(const VkExtent3D& extents) {
  std::stringstream ss;
  ss << "{" << extents.width << ", " << extents.height << ", " << extents.depth
     << "}";
  return ss.str();
}
std::vector<std::tuple<std::string, uint32_t, uint64_t, uint64_t>>
QueryPool::get_shader_timestamp_data() {
  if (VK_NULL_HANDLE == querypool_) {
    return {};
  }
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::tuple<std::string, uint32_t, uint64_t, uint64_t>>
      shader_timestamp_data;
  for (ShaderDuration& entry : shader_durations_) {
    shader_timestamp_data.emplace_back(std::make_tuple(
        entry.kernel_name,
        entry.dispatch_id,
        entry.start_time_ns,
        entry.end_time_ns));
  }
  return shader_timestamp_data;
}

std::string QueryPool::generate_string_report() {
  std::lock_guard<std::mutex> lock(mutex_);

  std::stringstream ss;

  int kernel_name_w = 40;
  int global_size_w = 25;
  int local_size_w = 25;
  int duration_w = 25;

  ss << std::left;
  ss << std::setw(kernel_name_w) << "Kernel Name";
  ss << std::setw(global_size_w) << "Global Workgroup Size";
  ss << std::setw(local_size_w) << "Local Workgroup Size";
  ss << std::right << std::setw(duration_w) << "Duration (ns)";
  ss << std::endl;

  ss << std::left;
  ss << std::setw(kernel_name_w) << "===========";
  ss << std::setw(global_size_w) << "=====================";
  ss << std::setw(local_size_w) << "====================";
  ss << std::right << std::setw(duration_w) << "=============";
  ss << std::endl;

  for (ShaderDuration& entry : shader_durations_) {
    std::chrono::duration<size_t, std::nano> exec_duration_ns(
        entry.execution_duration_ns);

    ss << std::left;
    ss << std::setw(kernel_name_w) << entry.kernel_name;
    ss << std::setw(global_size_w) << stringize(entry.global_workgroup_size);
    ss << std::setw(local_size_w) << stringize(entry.local_workgroup_size);
    ss << std::right << std::setw(duration_w) << exec_duration_ns.count();
    ss << std::endl;
  }

  return ss.str();
}

void QueryPool::print_results() {
  EARLY_RETURN_IF_UNINITIALIZED();
  std::cout << generate_string_report() << std::endl;
}

unsigned long QueryPool::get_total_shader_ns(std::string kernel_name) {
  for (ShaderDuration& entry : shader_durations_) {
    if (entry.kernel_name == kernel_name) {
      std::chrono::duration<size_t, std::nano> exec_duration_ns(
          entry.execution_duration_ns);
      return exec_duration_ns.count();
    }
  }
  return 0;
}

unsigned long QueryPool::get_mean_shader_ns(std::string kernel_name) {
  uint64_t total_ns = 0;
  uint32_t count = 0;
  for (ShaderDuration& entry : shader_durations_) {
    if (entry.kernel_name == kernel_name) {
      std::chrono::duration<size_t, std::nano> exec_duration_ns(
          entry.execution_duration_ns);
      total_ns += exec_duration_ns.count();
      count++;
    }
  }
  if (count == 0) {
    return 0;
  }
  return total_ns / count;
}
} // namespace vkapi
} // namespace vkcompute
