/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <limits>

#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

/**
 * @class CudaMemoryTracker
 * @brief Tracks CUDA memory usage and logs memory state at key points
 *
 * This class provides utilities to query and track CUDA memory usage,
 * including peak memory usage and detailed memory state logging.
 */
class CudaMemoryTracker {
 public:
  /**
   * @brief Constructor - initializes tracker and logs startup memory state
   */
  CudaMemoryTracker() {
    if (!query(&last_free_bytes_, &total_bytes_)) {
      return;
    }
    available_ = true;
    // Record the initial free bytes observed at startup. We'll use this as a
    // baseline so reported "peak usage" reflects additional memory used
    // since the tracker was created (instead of the absolute device usage,
    // which may include other processes).
    initial_free_bytes_ = last_free_bytes_;
    min_free_bytes_ = last_free_bytes_;
    log_state("startup", last_free_bytes_, total_bytes_);
  }

  /**
   * @brief Logs current memory state at a tagged checkpoint
   * @param tag Descriptive tag for this memory sample (e.g., "after_load")
   */
  void log_sample(const char* tag) {
    if (!available_) {
      return;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!query(&free_bytes, &total_bytes)) {
      return;
    }
    min_free_bytes_ = std::min(min_free_bytes_, free_bytes);
    total_bytes_ = total_bytes;
    last_free_bytes_ = free_bytes;
    log_state(tag, free_bytes, total_bytes);
  }

  /**
   * @brief Destructor - logs final memory state and peak usage summary
   */
  ~CudaMemoryTracker() {
    if (!available_) {
      return;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!query(&free_bytes, &total_bytes)) {
      return;
    }
    min_free_bytes_ = std::min(min_free_bytes_, free_bytes);
    total_bytes_ = total_bytes;
    last_free_bytes_ = free_bytes;
    // Compute peak usage relative to the initial free baseline so that
    // allocations by other processes present at startup are not attributed
    // to this process. If for some reason initial_free_bytes_ was not set,
    // fall back to absolute device usage.
    double peak_mb = 0.0;
    if (initial_free_bytes_ != std::numeric_limits<size_t>::max()) {
      size_t used_delta = 0;
      if (initial_free_bytes_ > min_free_bytes_) {
        used_delta = initial_free_bytes_ - min_free_bytes_;
      }
      peak_mb = static_cast<double>(used_delta) / (1024.0 * 1024.0);
    } else {
      peak_mb = static_cast<double>(total_bytes_ - min_free_bytes_) /
          (1024.0 * 1024.0);
    }
    const double total_mb =
        static_cast<double>(total_bytes_) / (1024.0 * 1024.0);
    ET_LOG(
        Info,
        "CUDA memory peak usage (since startup): %.2f MB, device total: %.2f MB",
        peak_mb,
        total_mb);
  }

 private:
  /**
   * @brief Queries current CUDA memory info
   * @param free_bytes Output parameter for free memory in bytes
   * @param total_bytes Output parameter for total memory in bytes
   * @return true if query succeeded, false otherwise
   */
  bool query(size_t* free_bytes, size_t* total_bytes) {
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    if (err != cudaSuccess) {
      if (!error_logged_) {
        error_logged_ = true;
        ET_LOG(
            Error,
            "cudaMemGetInfo failed with error: %s",
            cudaGetErrorString(err));
      }
      available_ = false;
      return false;
    }
    return true;
  }

  /**
   * @brief Logs the current memory state
   * @param tag Tag describing this log point
   * @param free_bytes Current free memory in bytes
   * @param total_bytes Current total memory in bytes
   */
  void log_state(const char* tag, size_t free_bytes, size_t total_bytes) const {
    const double used_mb =
        static_cast<double>(total_bytes - free_bytes) / (1024.0 * 1024.0);
    const double free_mb = static_cast<double>(free_bytes) / (1024.0 * 1024.0);
    const double total_mb =
        static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    ET_LOG(
        Info,
        "CUDA memory (%s): used %.2f MB, free %.2f MB, total %.2f MB",
        tag,
        used_mb,
        free_mb,
        total_mb);
  }

  bool available_{false};
  bool error_logged_{false};
  size_t last_free_bytes_{0};
  size_t total_bytes_{0};
  size_t min_free_bytes_{std::numeric_limits<size_t>::max()};
  // Baseline free bytes observed at tracker construction. Used to compute
  // peak usage attributable to this process since the tracker started.
  size_t initial_free_bytes_{std::numeric_limits<size_t>::max()};

 public:
  // Simple accessors to allow other components to read last-sampled values.
  // These are safe to call after a successful log_sample() invocation.
  uint64_t last_free_bytes() const {
    return static_cast<uint64_t>(last_free_bytes_);
  }
  uint64_t total_bytes() const {
    return static_cast<uint64_t>(total_bytes_);
  }
  uint64_t min_free_bytes() const {
    return static_cast<uint64_t>(min_free_bytes_);
  }
  uint64_t initial_free_bytes() const {
    return static_cast<uint64_t>(initial_free_bytes_);
  }
  double peak_usage_mb() const {
    // Prefer peak relative to the initial free baseline; fall back to
    // absolute device peak if baseline isn't available.
    if (min_free_bytes_ == std::numeric_limits<size_t>::max()) {
      return 0.0;
    }
    if (initial_free_bytes_ != std::numeric_limits<size_t>::max()) {
      size_t used_delta = 0;
      if (initial_free_bytes_ > min_free_bytes_) {
        used_delta = initial_free_bytes_ - min_free_bytes_;
      }
      return static_cast<double>(used_delta) / (1024.0 * 1024.0);
    }
    if (total_bytes_ == 0) {
      return 0.0;
    }
    return static_cast<double>(total_bytes_ - min_free_bytes_) /
        (1024.0 * 1024.0);
  }
};

} // namespace executorch::backends::cuda
