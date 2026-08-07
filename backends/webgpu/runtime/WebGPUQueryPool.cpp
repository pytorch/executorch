/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUCompat.h>
#include <executorch/backends/webgpu/runtime/WebGPUQueryPool.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace executorch::backends::webgpu {

#ifdef WGPU_BACKEND_ENABLE_PROFILING

namespace {

struct MapCallbackData {
  WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
};

using MapCallbackDataPtr = std::shared_ptr<MapCallbackData>;

void map_callback(
    WGPUMapAsyncStatus status,
    WGPUStringView /*message*/,
    void* userdata1,
    void* /*userdata2*/) {
  std::unique_ptr<MapCallbackDataPtr> data_owner(
      static_cast<MapCallbackDataPtr*>(userdata1));
  (*data_owner)->status = status;
}

constexpr uint64_t kTimestampBytes = sizeof(uint64_t);

} // namespace

WebGPUQueryPool::~WebGPUQueryPool() {
  if (readback_buf_) {
    wgpuBufferRelease(readback_buf_);
  }
  if (resolve_buf_) {
    wgpuBufferRelease(resolve_buf_);
  }
  if (qset_) {
    wgpuQuerySetRelease(qset_);
  }
}

void WebGPUQueryPool::initialize(WGPUDevice device, uint32_t max_pairs) {
  if (max_pairs == 0) {
    return;
  }
  // Re-init guard; mirrors Vulkan QueryPool (avoids leaking a prior QuerySet).
  if (qset_ != nullptr) {
    return;
  }
  capacity_pairs_ = max_pairs;
  const uint32_t count = 2 * max_pairs;
  const uint64_t bytes = static_cast<uint64_t>(count) * kTimestampBytes;

  WGPUQuerySetDescriptor qsd = {};
  qsd.type = WGPUQueryType_Timestamp;
  qsd.count = count;
  qset_ = wgpuDeviceCreateQuerySet(device, &qsd);

  WGPUBufferDescriptor rbd = {};
  rbd.size = bytes;
  rbd.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
  resolve_buf_ = wgpuDeviceCreateBuffer(device, &rbd);

  WGPUBufferDescriptor mbd = {};
  mbd.size = bytes;
  mbd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
  readback_buf_ = wgpuDeviceCreateBuffer(device, &mbd);
  // WebGPU timestamps are already nanoseconds, so ns_per_tick_ stays 1.0.
}

void WebGPUQueryPool::reset(uint32_t num_dispatches) {
  // Fail loud on overrun; mirrors Vulkan QueryPool VK_CHECK_COND guard.
  if (num_dispatches > capacity_pairs_) {
    throw std::runtime_error(
        "WebGPUQueryPool: num_dispatches " + std::to_string(num_dispatches) +
        " exceeds capacity " + std::to_string(capacity_pairs_));
  }
  num_pairs_ = num_dispatches;
  durations_.clear();
  result_state_.invalidate();
}

WGPUPassTimestampWrites WebGPUQueryPool::writes_for(uint32_t i) {
  WGPUPassTimestampWrites tw = {};
  tw.querySet = qset_;
  tw.beginningOfPassWriteIndex = 2 * i;
  tw.endOfPassWriteIndex = 2 * i + 1;
  return tw;
}

void WebGPUQueryPool::record(
    uint32_t i,
    const std::string& name,
    std::array<uint32_t, 3> gwg,
    std::array<uint32_t, 3> lwg) {
  ShaderDuration d;
  d.idx = i;
  d.kernel_name = name;
  d.global_wg = gwg;
  d.local_wg = lwg;
  durations_.push_back(d);
}

void WebGPUQueryPool::resolve(WGPUCommandEncoder encoder) {
  if (num_pairs_ == 0) {
    return;
  }
  const uint32_t count = 2 * num_pairs_;
  wgpuCommandEncoderResolveQuerySet(encoder, qset_, 0, count, resolve_buf_, 0);
  wgpuCommandEncoderCopyBufferToBuffer(
      encoder,
      resolve_buf_,
      0,
      readback_buf_,
      0,
      static_cast<uint64_t>(count) * kTimestampBytes);
}

// Mali can pin begin-of-pass timestamps. Preserve raw start/end timestamps but
// derive durations from consecutive ends. State is local to this extraction;
// the first dispatch can still include work before its recorded pass.
void fill_shader_durations(
    std::vector<ShaderDuration>& durations,
    const uint64_t* ticks,
    double ns_per_tick) {
  std::sort(
      durations.begin(),
      durations.end(),
      [](const ShaderDuration& a, const ShaderDuration& b) {
        return a.idx < b.idx;
      });
  uint64_t prev_end = 0;
  bool have_prev = false;
  for (auto& d : durations) {
    const uint64_t t0 = ticks[2 * d.idx];
    const uint64_t t1 = ticks[2 * d.idx + 1];
    const uint64_t base = have_prev ? std::max(t0, prev_end) : t0;
    d.start_time_ns = static_cast<uint64_t>(t0 * ns_per_tick);
    d.end_time_ns = static_cast<uint64_t>(t1 * ns_per_tick);
    d.execution_duration_ns =
        (t1 > base) ? static_cast<uint64_t>((t1 - base) * ns_per_tick) : 0;
    prev_end = have_prev ? std::max(prev_end, t1) : t1;
    have_prev = true;
  }
}

bool WebGPUQueryPool::finalize_extraction(
    WGPUWaitStatus wait_status,
    WGPUMapAsyncStatus map_status,
    const uint64_t* ticks) {
  result_state_.invalidate();
  if (!detail::query_result_extraction_succeeded(
          num_pairs_, wait_status, map_status, ticks)) {
    return false;
  }
  if (num_pairs_ != 0) {
    fill_shader_durations(durations_, ticks, ns_per_tick_);
  }
  result_state_.complete();
  return true;
}

void WebGPUQueryPool::extract_results(WGPUInstance instance) {
  result_state_.invalidate();
  if (num_pairs_ == 0) {
    (void)finalize_extraction(
        WGPUWaitStatus_Success, WGPUMapAsyncStatus_Success, nullptr);
    return;
  }
  const uint32_t count = 2 * num_pairs_;
  const uint64_t bytes = static_cast<uint64_t>(count) * kTimestampBytes;

  const auto cb_data = std::make_shared<MapCallbackData>();
  WGPUBufferMapCallbackInfo cb_info = {};
  cb_info.mode = WGPUCallbackMode_WaitAnyOnly;
  cb_info.callback = map_callback;
  cb_info.userdata1 = new MapCallbackDataPtr(cb_data);
  const WGPUFuture map_future =
      wgpuBufferMapAsync(readback_buf_, WGPUMapMode_Read, 0, bytes, cb_info);
  const WGPUWaitStatus wait_status = webgpu_wait(instance, map_future);

  if (wait_status != WGPUWaitStatus_Success) {
    wgpuBufferUnmap(readback_buf_);
    (void)webgpu_wait(instance, map_future);
    (void)finalize_extraction(wait_status, cb_data->status, nullptr);
    printf(
        "WebGPUQueryPool: readback wait failed (status %d)\n",
        (int)wait_status);
    return;
  }
  if (cb_data->status != WGPUMapAsyncStatus_Success) {
    (void)finalize_extraction(wait_status, cb_data->status, nullptr);
    printf(
        "WebGPUQueryPool: readback map failed (status %d)\n",
        (int)cb_data->status);
    return;
  }
  const uint64_t* ticks = static_cast<const uint64_t*>(
      wgpuBufferGetConstMappedRange(readback_buf_, 0, bytes));
  const bool valid =
      finalize_extraction(wait_status, cb_data->status, ticks);
  wgpuBufferUnmap(readback_buf_);
  if (!valid) {
    printf("WebGPUQueryPool: readback mapped range is null\n");
  }
}

void WebGPUQueryPool::print_results(bool tsv) const {
  const char* sep = tsv ? "\t" : "  ";
  if (tsv) {
    printf("idx%skernel%sgwg%sduration_us\n", sep, sep, sep);
  } else {
    printf("=== WebGPUQueryPool: per-dispatch GPU time ===\n");
  }
  for (const auto& d : durations_) {
    const double us = d.execution_duration_ns / 1000.0;
    printf(
        "%u%s%s%s(%u,%u,%u)%s%.3f\n",
        d.idx,
        sep,
        d.kernel_name.empty() ? "dispatch" : d.kernel_name.c_str(),
        sep,
        d.global_wg[0],
        d.global_wg[1],
        d.global_wg[2],
        sep,
        us);
  }
  if (tsv) {
    return;
  }
  std::map<std::string, std::pair<uint64_t, uint32_t>> totals;
  for (const auto& d : durations_) {
    auto& t = totals[d.kernel_name.empty() ? "dispatch" : d.kernel_name];
    t.first += d.execution_duration_ns;
    t.second += 1;
  }
  printf("--- per-kernel mean / total (us) ---\n");
  for (const auto& kv : totals) {
    const double mean_us = kv.second.first / kv.second.second / 1000.0;
    const double total_us = kv.second.first / 1000.0;
    printf(
        "%s%smean %.3f%stotal %.3f (n=%u)\n",
        kv.first.c_str(),
        sep,
        mean_us,
        sep,
        total_us,
        kv.second.second);
  }
}

uint64_t WebGPUQueryPool::get_mean_shader_ns(
    const std::string& kernel_name) const {
  uint64_t sum = 0;
  uint32_t n = 0;
  for (const auto& d : durations_) {
    if (d.kernel_name == kernel_name) {
      sum += d.execution_duration_ns;
      n += 1;
    }
  }
  return n == 0 ? 0 : sum / n;
}

#endif // WGPU_BACKEND_ENABLE_PROFILING

} // namespace executorch::backends::webgpu
