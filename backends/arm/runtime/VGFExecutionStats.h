/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstring>

#if defined(EXECUTORCH_VGF_IO_STATS) && EXECUTORCH_VGF_IO_STATS
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace executorch::backends::vgf {

// One record per VGFBackend::execute(), not per whole-model invocation.
struct VgfExecutionStats {
  uint64_t in_copy_bytes = 0;
  uint64_t out_copy_bytes = 0;
  uint64_t in_memcpy_ns = 0;
  uint64_t out_memcpy_ns = 0;
  uint64_t submit_wait_ns = 0;
  uint64_t execute_ns = 0;
  uint64_t imports = 0;
  uint64_t import_ns = 0;
  uint64_t binding_refresh_ns = 0;
  uint64_t in_zero_copy_bytes = 0;
  uint64_t out_zero_copy_bytes = 0;
  // Future I/O zero-copy fallbacks, not undelegated CPU graph operators.
  uint64_t fallback_count = 0;
  uint64_t rejection_count = 0;

  void reset() {
    *this = {};
  }
};

struct VgfExecutionRecord {
  VgfExecutionStats stats;
  uintptr_t delegate_handle = 0;
  bool success = false;
};

// The caller owns this preallocated storage. No allocation, logging or file
// I/O is performed by execute() to publish a record. Overflow is an error for
// the benchmark runner, never silent truncation of measurements.
struct VgfStatsBuffer {
  VgfExecutionRecord* records = nullptr;
  size_t capacity = 0;
  size_t size = 0;
  bool overflow = false;
};

struct VgfDeviceInfo {
  bool valid = false;
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  uint32_t api_version = 0;
  uint32_t driver_version = 0;
  char device_name[256] = {};
  char driver_name[256] = {};
  char driver_info[256] = {};
};

VgfStatsBuffer* set_vgf_stats_buffer(VgfStatsBuffer* buffer) noexcept;
VgfExecutionStats* current_vgf_execution_stats() noexcept;
void set_vgf_device_info(const VgfDeviceInfo& info);
VgfDeviceInfo get_vgf_device_info();

using VgfStatsClock = std::chrono::steady_clock;
inline uint64_t vgf_elapsed_ns(VgfStatsClock::time_point start) noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          VgfStatsClock::now() - start)
          .count());
}

class ScopedVgfStatsTimer final {
 public:
  explicit ScopedVgfStatsTimer(uint64_t* destination) noexcept
      : destination_(destination), start_(VgfStatsClock::now()) {}
  ~ScopedVgfStatsTimer() {
    if (destination_)
      *destination_ += vgf_elapsed_ns(start_);
  }
  ScopedVgfStatsTimer(const ScopedVgfStatsTimer&) = delete;
  ScopedVgfStatsTimer& operator=(const ScopedVgfStatsTimer&) = delete;

 private:
  uint64_t* destination_;
  VgfStatsClock::time_point start_;
};

class ScopedVgfExecutionStats final {
 public:
  explicit ScopedVgfExecutionStats(const void* handle) noexcept;
  ~ScopedVgfExecutionStats();
  void mark_success() noexcept {
    record_.success = true;
  }
  ScopedVgfExecutionStats(const ScopedVgfExecutionStats&) = delete;
  ScopedVgfExecutionStats& operator=(const ScopedVgfExecutionStats&) = delete;

 private:
  VgfExecutionRecord record_{};
  VgfExecutionStats* previous_;
  VgfStatsClock::time_point start_;
};

inline void vgf_stats_memcpy(void* dst, const void* src, size_t n, bool input) {
  auto* stats = current_vgf_execution_stats();
  if (!stats) {
    std::memcpy(dst, src, n);
    return;
  }
  {
    ScopedVgfStatsTimer timer(
        input ? &stats->in_memcpy_ns : &stats->out_memcpy_ns);
    std::memcpy(dst, src, n);
  }
  // Count the exact memcpy length, after the timed region. This deliberately
  // excludes map_io/unmap_io, which do not map/unmap Vulkan memory per call.
  (input ? stats->in_copy_bytes : stats->out_copy_bytes) += n;
}

} // namespace executorch::backends::vgf

#define VGF_STATS_CAT_INNER(a, b) a##b
#define VGF_STATS_CAT(a, b) VGF_STATS_CAT_INNER(a, b)
#define VGF_STATS_EXECUTION(handle)                    \
  ::executorch::backends::vgf::ScopedVgfExecutionStats \
  vgf_execution_stats_scope(handle)
#define VGF_STATS_SUCCESS() vgf_execution_stats_scope.mark_success()
#define VGF_STATS_TIME(field)                                           \
  ::executorch::backends::vgf::ScopedVgfStatsTimer VGF_STATS_CAT(       \
      vgf_stats_timer_, __LINE__)(                                      \
      ::executorch::backends::vgf::current_vgf_execution_stats()        \
          ? &::executorch::backends::vgf::current_vgf_execution_stats() \
                 -> field                                               \
          : nullptr)
#define VGF_STATS_MEMCPY_IN(dst, src, n) \
  ::executorch::backends::vgf::vgf_stats_memcpy(dst, src, n, true)
#define VGF_STATS_MEMCPY_OUT(dst, src, n) \
  ::executorch::backends::vgf::vgf_stats_memcpy(dst, src, n, false)
#else
#define VGF_STATS_EXECUTION(handle) \
  do {                              \
  } while (false)
#define VGF_STATS_SUCCESS() \
  do {                      \
  } while (false)
#define VGF_STATS_TIME(field) \
  do {                        \
  } while (false)
#define VGF_STATS_MEMCPY_IN(dst, src, n) memcpy(dst, src, n)
#define VGF_STATS_MEMCPY_OUT(dst, src, n) memcpy(dst, src, n)
#endif
