/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// MetalStream wrapper + buffer C ABI. AOTI dispatch logic lives in the
// aoti_* files in this directory.

#import <Metal/Metal.h>

#include <executorch/backends/apple/metal/runtime/shims/v2/runtime.h>
#include <executorch/backends/metal/core/MetalTypes.h>
#include <executorch/backends/metal/core/MetalStream.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <mutex>
#include <unordered_set>

namespace executorch {
namespace backends {
namespace metal {

using metal_v2::MetalStream;

namespace {

// Pointers known to be GPU-accessible (allocated by metal_allocate_buffer
// or registered by metal_buffer_nocopy). Source of truth for
// metal_is_device_pointer().
std::mutex g_device_ptrs_mutex;
std::unordered_set<void*> g_device_ptrs;

void track_device_ptr(void* ptr) {
  if (!ptr) return;
  std::lock_guard<std::mutex> lock(g_device_ptrs_mutex);
  g_device_ptrs.insert(ptr);
}

void untrack_device_ptr(void* ptr) {
  if (!ptr) return;
  std::lock_guard<std::mutex> lock(g_device_ptrs_mutex);
  g_device_ptrs.erase(ptr);
}

bool is_tracked_device_ptr(void* ptr) {
  if (!ptr) return false;
  std::lock_guard<std::mutex> lock(g_device_ptrs_mutex);
  return g_device_ptrs.count(ptr) != 0;
}

}  // namespace

MetalStream* getMetalStream() {
  // Thread-local stream avoids races on the per-thread command buffer
  // when execute() is invoked concurrently. The kernel cache (and
  // PSO/MTLLibrary cache inside MetalKernelCompiler) is process-wide
  // — kernels are NOT recompiled per thread.
  thread_local std::unique_ptr<MetalStream> tls = MetalStream::create();
  return tls.get();
}

void metal_set_flush_interval(int dispatches) {
  getMetalStream()->recorder().setFlushInterval(dispatches);
}

MTLDevice_t getMetalDevice() {
  return getMetalStream()->device();
}

extern "C" {

void* metal_allocate_buffer(long bytes) {
  if (bytes <= 0) return nullptr;
  void* ptr = getMetalStream()->allocator().alloc(static_cast<size_t>(bytes));
  if (ptr) track_device_ptr(ptr);
  return ptr;
}

void metal_deallocate_buffer(void* ptr) {
  if (!ptr) return;
  getMetalStream()->allocator().free(ptr);
  untrack_device_ptr(ptr);
}

bool metal_is_device_pointer(void* ptr) {
  return is_tracked_device_ptr(ptr);
}

int metal_copy_memory(
    void* dst,
    const void* src,
    size_t nbytes,
    bool src_is_device,
    bool /*dst_is_device*/) {
  if (nbytes == 0) return 0;  // 0-byte memcpy is a well-defined no-op.
  if (!src || !dst) return -1;

  // Apple Silicon unified memory: CPU and GPU share an address space.
  // wait() drains in-flight GPU work so its writes are visible to the
  // following memcpy; it does not trigger any new submission.
  if (src_is_device) {
    getMetalStream()->wait();
  }
  std::memcpy(dst, src, nbytes);
  return 0;
}

void metal_cleanup_resources() {
  std::lock_guard<std::mutex> lock(g_device_ptrs_mutex);
  g_device_ptrs.clear();
}

bool metal_buffer_nocopy(void* ptr, size_t nbytes) {
  if (!ptr || nbytes == 0) return false;
  bool ok = getMetalStream()->allocator().registerExternalBuffer(ptr, nbytes);
  if (ok) track_device_ptr(ptr);
  return ok;
}

void* get_metal_device() {
  return (__bridge void*)getMetalDevice();
}

}  // extern "C"

void synchronize_metal_stream() {
  getMetalStream()->wait();
}

}  // namespace metal
}  // namespace backends
}  // namespace executorch
