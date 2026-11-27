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

#include <algorithm>
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

size_t count_order_violations(const std::vector<ShaderTimestamp>& timestamps) {
  size_t violations = 0;
  for (size_t i = 0; i < timestamps.size(); ++i) {
    const ShaderTimestamp& entry = timestamps[i];
    const bool ordered =
        entry.previous_work_finish_ticks <= entry.barrier_finish_ticks &&
        entry.barrier_finish_ticks <= entry.shader_exec_finish_ticks &&
        (i + 1 == timestamps.size() ||
         entry.shader_exec_finish_ticks <=
             timestamps[i + 1].previous_work_finish_ticks);
    violations += ordered ? 0 : 1;
  }
  return violations;
}

#define EARLY_RETURN_IF_UNINITIALIZED() \
  if (querypool_ == VK_NULL_HANDLE) {   \
    return;                             \
  }

QueryPool::QueryPool(const QueryPoolConfig& config, const Adapter* adapter_p)
    : config_(config),
      ns_per_tick_(1u),
      device_(VK_NULL_HANDLE),
      querypool_(VK_NULL_HANDLE),
      num_queries_(0u),
      shader_durations_{},
      num_order_violations_(0u),
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

bool QueryPool::has_incomplete_entries() const {
  for (const ShaderDuration& entry : shader_durations_) {
    if (entry.previous_work_finish_query_idx >= num_queries_ ||
        entry.shader_exec_finish_query_idx >= num_queries_) {
      return true;
    }
  }
  return false;
}

void QueryPool::reset_querypool() {
  EARLY_RETURN_IF_UNINITIALIZED();
  std::lock_guard<std::mutex> lock(mutex_);

  reset_state();
}

void QueryPool::reset_all(const CommandBuffer& cmd) {
  EARLY_RETURN_IF_UNINITIALIZED();
  cmd.reset_querypool(querypool_, 0u, config_.max_query_count);
}

void QueryPool::reset_state() {
  num_queries_ = 0u;
  shader_durations_.clear();
  num_order_violations_ = 0u;
}

void QueryPool::record_previous_work_finish(
    const CommandBuffer& cmd,
    const uint32_t dispatch_id,
    const std::string& kernel_name,
    const GlobalWorkGrid& gwg,
    const LocalWorkGroup& lwg) {
  EARLY_RETURN_IF_UNINITIALIZED();
  std::lock_guard<std::mutex> lock(mutex_);

  // Reset only the queries this dispatch records. Command buffers are replayed,
  // so resetting the whole pool would cost time proportional to its size on
  // every execution.
  if (num_queries_ < config_.max_query_count) {
    cmd.reset_querypool(
        querypool_,
        num_queries_,
        std::min(kQueriesPerDispatch, config_.max_query_count - num_queries_));
  }
  uint32_t query_idx = write_timestamp(cmd);

  ShaderDuration log_entry{
      utils::safe_downcast<uint32_t>(shader_durations_.size()),
      // Execution Properties
      dispatch_id,
      kernel_name,
      gwg,
      lwg,
      // Query indexes
      query_idx, // previous work finish query idx
      UINT32_MAX, // barrier finish query idx
      UINT32_MAX, // shader exec finish query idx
      // Timings
      0u, // previous work finish
      0u, // barrier finish
      0u, // shader exec finish
      0u, // barrier duration
      0u, // shader exec duration
  };

  shader_durations_.emplace_back(log_entry);
}

void QueryPool::record_barrier_finish(const CommandBuffer& cmd) {
  EARLY_RETURN_IF_UNINITIALIZED();
  std::lock_guard<std::mutex> lock(mutex_);

  if (shader_durations_.empty()) {
    return;
  }
  ShaderDuration& entry = shader_durations_.back();
  if (entry.shader_exec_finish_query_idx != UINT32_MAX ||
      entry.barrier_finish_query_idx != UINT32_MAX) {
    return;
  }
  entry.barrier_finish_query_idx = write_timestamp(cmd);
}

void QueryPool::record_shader_exec_finish(const CommandBuffer& cmd) {
  EARLY_RETURN_IF_UNINITIALIZED();
  std::lock_guard<std::mutex> lock(mutex_);

  if (shader_durations_.empty() ||
      shader_durations_.back().shader_exec_finish_query_idx != UINT32_MAX) {
    return;
  }
  shader_durations_.back().shader_exec_finish_query_idx = write_timestamp(cmd);
}

std::vector<ShaderTimestamp> QueryPool::to_shader_timestamps(
    const std::vector<uint64_t>& ticks) const {
  std::vector<ShaderTimestamp> results;
  results.reserve(shader_durations_.size());
  for (const ShaderDuration& entry : shader_durations_) {
    const uint32_t barrier_finish_idx =
        entry.barrier_finish_query_idx < ticks.size()
        ? entry.barrier_finish_query_idx
        : entry.previous_work_finish_query_idx;
    results.push_back(
        {entry.kernel_name,
         ticks.at(entry.previous_work_finish_query_idx),
         ticks.at(barrier_finish_idx),
         ticks.at(entry.shader_exec_finish_query_idx),
         {{entry.gwg[0u], entry.gwg[1u], entry.gwg[2u]},
          {entry.lwg.x(), entry.lwg.y(), entry.lwg.z()}}});
  }
  return results;
}

void QueryPool::extract_results() {
  EARLY_RETURN_IF_UNINITIALIZED();
  std::lock_guard<std::mutex> lock(mutex_);
  if (has_incomplete_entries()) {
    return;
  }

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

  const std::vector<ShaderTimestamp> timestamps =
      to_shader_timestamps(query_data);
  for (size_t i = 0; i < shader_durations_.size(); ++i) {
    ShaderDuration& entry = shader_durations_[i];
    entry.previous_work_finish_ns =
        timestamps[i].previous_work_finish_ticks * ns_per_tick_;
    entry.barrier_finish_ns = timestamps[i].barrier_finish_ticks * ns_per_tick_;
    entry.shader_exec_finish_ns =
        timestamps[i].shader_exec_finish_ticks * ns_per_tick_;
    entry.barrier_duration_ns =
        entry.barrier_finish_ns - entry.previous_work_finish_ns;
    entry.shader_exec_duration_ns =
        entry.shader_exec_finish_ns - entry.barrier_finish_ns;
  }
  num_order_violations_ = count_order_violations(timestamps);
}

std::ostream& operator<<(std::ostream& os, const utils::uvec3& extents) {
  os << "{" << extents[0] << ", " << extents[1] << ", " << extents[2] << "}";
  return os;
}

std::string stringize(const utils::uvec3& extents) {
  std::stringstream ss;
  ss << extents;
  return ss.str();
}

std::vector<ShaderResult> QueryPool::get_shader_timestamp_data() {
  if (querypool_ == VK_NULL_HANDLE) {
    return {};
  }
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<ShaderResult> shader_result;
  for (ShaderDuration& entry : shader_durations_) {
    shader_result.push_back(ShaderResult{
        /* .kernel_name = */ entry.kernel_name,
        /* .dispatch_id = */ entry.dispatch_id,
        /* .previous_work_finish_ns = */ entry.previous_work_finish_ns,
        /* .barrier_finish_ns = */ entry.barrier_finish_ns,
        /* .shader_exec_finish_ns = */ entry.shader_exec_finish_ns,
        /* .metadata = */
        ShaderMetadata{
            /* .gwg = */
            {entry.gwg[0u], entry.gwg[1u], entry.gwg[2u]},
            /* .lwg = */
            {entry.lwg.x(), entry.lwg.y(), entry.lwg.z()},
        }});
  }
  return shader_result;
}

std::vector<ShaderTimestamp> QueryPool::get_shader_timestamp_ticks() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (querypool_ == VK_NULL_HANDLE || num_queries_ == 0) {
    return {};
  }
  if (has_incomplete_entries()) {
    return {};
  }
  std::vector<uint64_t> ticks(num_queries_);
  // Without PARTIAL_BIT, any unavailable query makes the whole call NOT_READY.
  const VkResult status = vkGetQueryPoolResults(
      device_,
      querypool_,
      0u,
      num_queries_,
      sizeof(uint64_t) * ticks.size(),
      ticks.data(),
      sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT);
  if (status == VK_NOT_READY) {
    return {};
  }
  VK_CHECK(status);
  return to_shader_timestamps(ticks);
}

std::string QueryPool::generate_string_report() {
  std::lock_guard<std::mutex> lock(mutex_);

  std::stringstream ss;

  int kernel_name_w = 120;
  int global_size_w = 25;
  int local_size_w = 25;
  int duration_w = 25;

  ss << std::left;
  ss << std::setw(kernel_name_w) << "Kernel Name";
  ss << std::setw(global_size_w) << "Global Workgroup Size";
  ss << std::setw(local_size_w) << "Local Workgroup Size";
  ss << std::right << std::setw(duration_w) << "Barrier (ns)";
  ss << std::right << std::setw(duration_w) << "Duration (ns)";
  ss << std::endl;

  ss << std::left;
  ss << std::setw(kernel_name_w) << "===========";
  ss << std::setw(global_size_w) << "=====================";
  ss << std::setw(local_size_w) << "====================";
  ss << std::right << std::setw(duration_w) << "============";
  ss << std::right << std::setw(duration_w) << "=============";
  ss << std::endl;

  for (ShaderDuration& entry : shader_durations_) {
    ss << std::left;
    ss << std::setw(kernel_name_w) << entry.kernel_name;
    ss << std::setw(global_size_w) << stringize(entry.gwg.extents());
    ss << std::setw(local_size_w)
       << stringize(static_cast<utils::uvec3>(entry.lwg));
    ss << std::right << std::setw(duration_w) << entry.barrier_duration_ns;
    ss << std::right << std::setw(duration_w) << entry.shader_exec_duration_ns;
    ss << std::endl;
  }

  return ss.str();
}

std::string QueryPool::generate_tsv_string_report() {
  std::lock_guard<std::mutex> lock(mutex_);

  std::stringstream ss;

  ss << "Kernel Name\t";
  ss << "Global Workgroup Size\t";
  ss << "Local Workgroup Size\t";
  ss << "Barrier (ns)\t";
  ss << "Duration (ns)\t";
  ss << std::endl;

  ss << "===========\t";
  ss << "=====================\t";
  ss << "====================\t";
  ss << "============\t";
  ss << "=============\t";
  ss << std::endl;

  for (ShaderDuration& entry : shader_durations_) {
    ss << entry.kernel_name << "\t";
    ss << stringize(entry.gwg.extents()) << "\t";
    ss << stringize(static_cast<utils::uvec3>(entry.lwg)) << "\t";
    ss << entry.barrier_duration_ns << "\t";
    ss << entry.shader_exec_duration_ns << "\t";
    ss << std::endl;
  }

  return ss.str();
}

void QueryPool::print_results(const bool tsv_format) {
  EARLY_RETURN_IF_UNINITIALIZED();
  if (tsv_format) {
    std::cout << generate_tsv_string_report() << std::endl;
  } else {
    std::cout << generate_string_report() << std::endl;
  }
}

unsigned long QueryPool::get_total_shader_ns(std::string kernel_name) {
  for (ShaderDuration& entry : shader_durations_) {
    if (entry.kernel_name == kernel_name) {
      std::chrono::duration<size_t, std::nano> exec_duration_ns(
          entry.shader_exec_duration_ns);
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
          entry.shader_exec_duration_ns);
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
