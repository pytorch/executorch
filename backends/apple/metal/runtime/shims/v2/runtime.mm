/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// MetalStream wrapper + buffer C ABI for v2.
//
// Buffer management and stream access. All AOTI dispatch logic lives in
// the aoti_* files in this directory.
//
// Device-pointer tracking is kept here (rather than in MetalStream) so we
// don't have to extend the portable MetalStream API. We track every
// pointer we hand back from metal_allocate_buffer or successfully register
// via metal_buffer_nocopy and use that set as the source of truth for
// metal_is_device_pointer.

#import <Metal/Metal.h>

#include <executorch/backends/apple/metal/runtime/shims/v2/runtime.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalTypes.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <mutex>
#include <unordered_set>

namespace executorch {
namespace backends {
namespace metal {

using metal_v2::MetalStream;

namespace {

// Pointers we know are GPU-accessible (allocated via alloc() or
// successfully registered via registerExternalBuffer()).
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

} // namespace

MetalStream* getMetalStream() {
  // Thread-local stream: each thread that calls into the v2 shim layer gets
  // its own MetalStream. Avoids races on the shared command buffer when
  // execute() is invoked concurrently from multiple threads. Trade-off:
  // kernel cache and buffer pool are per-thread, so shaders are recompiled
  // on each new thread.
  return MetalStream::getThreadLocal();
}

void metal_set_flush_interval(int dispatches) {
  getMetalStream()->setFlushInterval(dispatches);
}

MTLDevice_t getMetalDevice() {
  return getMetalStream()->device();
}

extern "C" {

void* metal_allocate_buffer(long bytes) {
  if (bytes <= 0) return nullptr;
  void* ptr = getMetalStream()->alloc(static_cast<size_t>(bytes));
  if (ptr) track_device_ptr(ptr);
  return ptr;
}

void* metal_allocate_buffer_untracked(long bytes) {
  if (bytes <= 0) return nullptr;
  return getMetalStream()->alloc(static_cast<size_t>(bytes));
}

void metal_deallocate_buffer(void* ptr) {
  if (!ptr) return;
  getMetalStream()->free(ptr);
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
  if (!src || !dst || nbytes == 0) return -1;

  // Apple Silicon unified memory: CPU and GPU share the same address space.
  // Just need to ensure GPU writes are visible before the CPU reads them.
  // wait() = "drain whatever's in flight"; we don't want to *trigger* any
  // new submission here, just block until the GPU is caught up.
  if (src_is_device) {
    getMetalStream()->wait();
  }
  std::memcpy(dst, src, nbytes);
  return 0;
}

void metal_cleanup_resources() {
  // MetalStream manages its own pool. Clear our local tracking so a fresh
  // process state doesn't leak entries.
  std::lock_guard<std::mutex> lock(g_device_ptrs_mutex);
  g_device_ptrs.clear();
}

bool metal_buffer_nocopy(void* ptr, size_t nbytes, bool /*map_ptr_to_buffer*/) {
  if (!ptr || nbytes == 0) return false;
  bool ok = getMetalStream()->registerExternalBuffer(ptr, nbytes);
  if (ok) track_device_ptr(ptr);
  return ok;
}

bool metal_register_external_buffer_only(void* ptr, size_t nbytes) {
  if (!ptr || nbytes == 0) return false;
  return getMetalStream()->registerExternalBuffer(ptr, nbytes);
}

// Public C-ABI: drain the stream so GPU writes are visible to the CPU.
// Used by metal_backend_v2::execute() after handle->run(); also exported
// for any AOTI-emitted code that wants an explicit sync point. Marks the
// end of an AOTI execute — calls endExecute() to reset per-execute state
// (currentDispatchIdx_, icbRecordedThisIter_) needed for replay
// correctness across iterations.
void synchronize_metal_stream() {
  getMetalStream()->wait();
  getMetalStream()->endExecute();
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
