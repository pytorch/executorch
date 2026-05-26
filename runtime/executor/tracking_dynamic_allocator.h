/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/executor/dynamic_allocator.h>

#ifdef ET_DYNAMIC_ALLOCATOR_ENABLED

#include <algorithm>
#include <cstddef>

namespace executorch {
namespace runtime {

/**
 * A DynamicAllocator wrapper that tracks allocation statistics.
 * Delegates actual allocation to an inner DynamicAllocator and records
 * current bytes, peak bytes, and allocation count.
 */
class TrackingDynamicAllocator : public DynamicAllocator {
 public:
  explicit TrackingDynamicAllocator(DynamicAllocator* inner) : inner_(inner) {}

  void* allocate(size_t size, size_t alignment, size_t* actual_size) override {
    size_t actual = 0;
    void* ptr = inner_->allocate(size, alignment, &actual);
    if (ptr != nullptr) {
      current_bytes_ += actual;
      peak_bytes_ = std::max(peak_bytes_, current_bytes_);
      num_allocations_++;
      if (actual_size) {
        *actual_size = actual;
      }
    }
    return ptr;
  }

  void* reallocate(
      void* ptr,
      size_t old_size,
      size_t new_size,
      size_t alignment,
      size_t* actual_size) override {
    size_t actual = 0;
    void* new_ptr =
        inner_->reallocate(ptr, old_size, new_size, alignment, &actual);
    if (new_ptr != nullptr) {
      current_bytes_ -= old_size;
      current_bytes_ += actual;
      peak_bytes_ = std::max(peak_bytes_, current_bytes_);
      num_allocations_++;
      if (actual_size) {
        *actual_size = actual;
      }
    }
    return new_ptr;
  }

  void free(void* ptr) override {
    // Note: we don't track which allocation was which size, so current_bytes_
    // can only be accurately decremented if the caller tracks capacity_bytes.
    inner_->free(ptr);
  }

  /// Notify the tracker that `bytes` have been freed.
  void record_free(size_t bytes) {
    current_bytes_ -= std::min(bytes, current_bytes_);
  }

  size_t current_bytes() const {
    return current_bytes_;
  }
  size_t peak_bytes() const {
    return peak_bytes_;
  }
  size_t num_allocations() const {
    return num_allocations_;
  }

 private:
  DynamicAllocator* inner_;
  size_t current_bytes_ = 0;
  size_t peak_bytes_ = 0;
  size_t num_allocations_ = 0;
};

} // namespace runtime
} // namespace executorch

#endif // ET_DYNAMIC_ALLOCATOR_ENABLED
