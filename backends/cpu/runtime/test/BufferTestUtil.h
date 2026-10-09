// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/cpu/runtime/KernelProvider.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>

namespace executorch::backends::cpu {

class GuardedBuffer {
 public:
  explicit GuardedBuffer(size_t logical_bytes) {
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t capacity = (logical_bytes + kReadableTail + 63) & ~size_t{63};
    const size_t usable = (capacity + page_size - 1) / page_size * page_size;
    mapping_bytes_ = usable + page_size;
    mapping_ = mmap(
        nullptr,
        mapping_bytes_,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0);
    ET_CHECK(mapping_ != MAP_FAILED);
    ET_CHECK(
        mprotect(
            static_cast<uint8_t*>(mapping_) + usable, page_size, PROT_NONE) ==
        0);
    buffer = {
        static_cast<uint8_t*>(mapping_) + usable - capacity,
        capacity,
        logical_bytes,
        kBufferAlignment};
    std::memset(buffer.data, 0, capacity);
  }
  ~GuardedBuffer() {
    munmap(mapping_, mapping_bytes_);
  }
  GuardedBuffer(const GuardedBuffer&) = delete;
  GuardedBuffer& operator=(const GuardedBuffer&) = delete;
  Buffer buffer;

 private:
  void* mapping_;
  size_t mapping_bytes_;
};

} // namespace executorch::backends::cpu
