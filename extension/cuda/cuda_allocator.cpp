/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/cuda/cuda_allocator.h>

#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>

#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>

#if !defined(EXECUTORCH_USE_HIP)
#include <vector>
#endif

namespace executorch::extension::cuda {

using executorch::runtime::Error;
using executorch::runtime::Result;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;

namespace {

struct PalInitializer final {
  PalInitializer() {
    // A static-runtime build gives this library its own platform state.
    et_pal_init();
  }
};

const PalInitializer kPalInitializer{};

// deallocate() must free the address cudaMalloc returned, not the rounded one.
struct RoundedAllocations {
  std::mutex mutex;
  std::unordered_map<void*, void*> original_by_rounded;
  std::atomic<size_t> count{0};
};

RoundedAllocations& rounded_allocations() {
  // Static destructors may still return rounded allocations.
  static auto* state = new RoundedAllocations();
  return *state;
}

#if !defined(EXECUTORCH_USE_HIP)
// The stream ordered allocator hands physical memory back to the driver
// whenever a synchronization observes a pending free, so with the default
// release threshold of zero a pool is emptied repeatedly during one inference
// and every allocation has to map memory again, which measured three orders of
// magnitude slower on an embedded board.
//
// A separate pool leaves the device default pool's policy unchanged. Delegates
// using this allocator share both its retention threshold and its cache trims.
constexpr uint64_t kMemPoolReleaseThreshold = UINT64_MAX;

struct MemPoolState {
  std::mutex mutex;
  std::unordered_map<int, cudaMemPool_t> pools;
};

MemPoolState& mem_pool_state() {
  // Delegate teardown may trim pools during static destruction.
  static auto* state = new MemPoolState();
  return *state;
}

// Resolves the "current device" sentinel that callers are allowed to pass.
// Returns a negative value when the device cannot be determined.
int resolve_device(DeviceIndex index) {
  if (index >= 0) {
    return static_cast<int>(index);
  }
  int current = 0;
  const cudaError_t err = cudaGetDevice(&current);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaGetDevice failed: %s. Using pool defaults.",
        cudaGetErrorString(err));
    (void)cudaGetLastError();
    return -1;
  }
  return current;
}

// The shared allocator pool on a device, creating it on first use.
// Returns nullptr when the pool cannot be created, in which case the caller
// falls back to the device default pool and only loses speed.
cudaMemPool_t mem_pool_for(int device) {
  auto& state = mem_pool_state();
  const std::lock_guard<std::mutex> lock(state.mutex);
  const auto it = state.pools.find(device);
  if (it != state.pools.end()) {
    return it->second;
  }

  cudaMemPoolProps props{};
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypeNone;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = device;

  cudaMemPool_t pool = nullptr;
  cudaError_t err = cudaMemPoolCreate(&pool, &props);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaMemPoolCreate failed for device %d: %s. Using the default pool.",
        device,
        cudaGetErrorString(err));
    (void)cudaGetLastError();
    // Recorded so a permanent failure is not retried and re-logged on every
    // allocation.
    state.pools.emplace(device, nullptr);
    return nullptr;
  }

  uint64_t threshold = kMemPoolReleaseThreshold;
  err = cudaMemPoolSetAttribute(
      pool, cudaMemPoolAttrReleaseThreshold, &threshold);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "Setting the pool release threshold failed for device %d: %s. Keeping "
        "the pool at its defaults.",
        device,
        cudaGetErrorString(err));
    (void)cudaGetLastError();
  }

  state.pools.emplace(device, pool);
  return pool;
}

// Whether `device` supports the stream-ordered allocator. Devices without
// memory pools, such as data-center GPUs in TCC mode on Windows, reject
// cudaMallocAsync and cudaMemPoolCreate with cudaErrorNotSupported, so their
// allocations take the synchronous path instead. Asked on every allocation,
// so the answer is cached per device in atomics rather than behind a lock. An
// attribute query that fails is treated as supported, which keeps the
// stream-ordered path and its own error reporting.
constexpr int kPoolSupportCacheSize = 64;
enum PoolSupport : int { kPoolSupportUnknown = 0, kPoolsSupported, kNoPools };
std::atomic<int> g_pool_support[kPoolSupportCacheSize] = {};
// Set once any device is found without memory pools, so a free can skip
// looking up which device its pointer is on while every device has them.
std::atomic<bool> g_device_without_pools_seen{false};

bool device_supports_memory_pools(int device) {
  const bool cached = device >= 0 && device < kPoolSupportCacheSize;
  if (cached) {
    const int known = g_pool_support[device].load(std::memory_order_acquire);
    if (known != kPoolSupportUnknown) {
      return known == kPoolsSupported;
    }
  }
  int value = 0;
  const cudaError_t err =
      cudaDeviceGetAttribute(&value, cudaDevAttrMemoryPoolsSupported, device);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
  }
  const bool supported = err != cudaSuccess || value != 0;
  if (!supported) {
    g_device_without_pools_seen.store(true, std::memory_order_release);
  }
  if (cached) {
    int expected = kPoolSupportUnknown;
    const int answer = supported ? kPoolsSupported : kNoPools;
    if (g_pool_support[device].compare_exchange_strong(expected, answer) &&
        !supported) {
      ET_LOG(
          Info,
          "CUDA device %d does not support memory pools; stream-ordered "
          "allocations use cudaMalloc and cudaFree instead, and CUDA graphs "
          "are not used.",
          device);
    }
  }
  return supported;
}

bool stream_is_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return status != cudaStreamCaptureStatusNone;
}

// The synchronous fallback for a device without memory pools. Refused while
// the stream is capturing: in relaxed capture mode cudaMalloc and cudaFree
// succeed, so a block freed during capture would be handed out again while
// the graph still writes to it on every replay. Failing here keeps the loud
// failure the stream-ordered call gave before.
Result<void*>
allocate_without_pools(size_t nbytes, DeviceIndex index, cudaStream_t stream) {
  if (stream_is_capturing(stream)) {
    ET_LOG(
        Error,
        "Cannot allocate %zu bytes while capturing a CUDA graph on a device "
        "without memory pools",
        nbytes);
    return Error::NotSupported;
  }
  return CudaAllocator::instance().allocate(nbytes, index);
}

// cudaFree waits for the work already submitted to the device, so work still
// using the block on `stream` finishes before it is released. During capture
// that work has only been recorded, so the block is kept rather than freed, and
// the caller is told: Error::NotSupported for a kept block, Error::Internal for
// a failed cudaFree. Freed here rather than through deallocate(), which reports
// nothing, so the result is cudaFree's own and not whatever else last failed.
Error deallocate_without_pools(
    void* ptr,
    DeviceIndex index,
    cudaStream_t stream) {
  if (stream_is_capturing(stream)) {
    ET_LOG(
        Error,
        "Not freeing %p while capturing a CUDA graph on a device without "
        "memory pools; the graph may still use it",
        ptr);
    return Error::NotSupported;
  }
  // On the block's device, as deallocate() frees it, so cudaFree waits for the
  // work submitted there. A failed switch still frees rather than leaking; only
  // cudaFree's own result is reported. cudaFree directly is right because the
  // fallback allocates at the default alignment, which allocate() never rounds,
  // so this is the pointer cudaMalloc returned.
  int previous = -1;
  bool switched = false;
  if (index >= 0 && cudaGetDevice(&previous) == cudaSuccess &&
      previous != static_cast<int>(index)) {
    switched = cudaSetDevice(index) == cudaSuccess;
  }
  const cudaError_t err = cudaFree(ptr);
  if (switched) {
    (void)cudaSetDevice(previous);
  }
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaFree failed: %s (ptr=%p, device %d)",
        cudaGetErrorString(err),
        ptr,
        static_cast<int>(index));
    return Error::Internal;
  }
  return Error::Ok;
}

#endif // !EXECUTORCH_USE_HIP

Error copy_impl(
    void* dst,
    const void* src,
    size_t nbytes,
    DeviceIndex index,
    cudaMemcpyKind kind) {
  ET_CHECK_OR_RETURN_ERROR(
      kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost,
      InvalidArgument,
      "CudaAllocator::copy_impl: unsupported cudaMemcpyKind %d",
      static_cast<int>(kind));
  const char* method = kind == cudaMemcpyHostToDevice
      ? "CudaAllocator::copy_host_to_device"
      : "CudaAllocator::copy_device_to_host";
  ET_CHECK_OR_RETURN_ERROR(
      dst != nullptr, InvalidArgument, "%s: dst is null", method);
  ET_CHECK_OR_RETURN_ERROR(
      src != nullptr, InvalidArgument, "%s: src is null", method);
  ET_CHECK_OR_RETURN_ERROR(
      index >= -1,
      InvalidArgument,
      "%s: invalid device index %d (must be >= -1)",
      method,
      static_cast<int>(index));
  const auto caller_stream = executorch::extension::cuda::getCallerStream();
  if (caller_stream) {
    // TODO: validate caller stream device matches index.
    // For now assert index is -1 or 0.
    ET_CHECK_OR_RETURN_ERROR(
        index == -1 || index == 0,
        InvalidArgument,
        "%s: with caller stream, only supports device 0 or -1 (current), got %d",
        method,
        static_cast<int>(index));
  }
  if (nbytes == 0) {
    return Error::Ok;
  }

  int prev_device = 0;
  bool switched_device = false;

  if (index >= 0) {
    // Without the current device there is no way to switch to `index` and no
    // way to restore afterwards, so copying would run against whatever device
    // happens to be current and report success. Fail instead, as
    // cuda_mutable_state.cpp does for the same call.
    cudaError_t prev_device_err = cudaGetDevice(&prev_device);
    if (prev_device_err != cudaSuccess) {
      ET_LOG(
          Error,
          "%s: cudaGetDevice failed: %s",
          method,
          cudaGetErrorString(prev_device_err));
      return Error::Internal;
    }
    if (static_cast<int>(index) != prev_device) {
      cudaError_t set_err = cudaSetDevice(index);
      if (set_err != cudaSuccess) {
        // Nothing was switched, so there is nothing to restore. Copying now
        // would silently run against whatever device is still current.
        ET_LOG(
            Error,
            "%s: cudaSetDevice(%d) failed: %s",
            method,
            static_cast<int>(index),
            cudaGetErrorString(set_err));
        return Error::Internal;
      }
      switched_device = true;
    }
  }
  cudaError_t err = cudaSuccess;
  if (caller_stream) {
    err = cudaMemcpyAsync(dst, src, nbytes, kind, *caller_stream);
    if (err == cudaSuccess && kind == cudaMemcpyDeviceToHost) {
      err = cudaStreamSynchronize(*caller_stream);
    }
  } else {
    err = cudaMemcpy(dst, src, nbytes, kind);
  }

  if (switched_device) {
    (void)cudaSetDevice(prev_device);
  }

  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaMemcpy %s failed: %s (%zu bytes, device %d)",
        kind == cudaMemcpyHostToDevice ? "H2D" : "D2H",
        cudaGetErrorString(err),
        nbytes,
        static_cast<int>(index));
    return Error::Internal;
  }
  return Error::Ok;
}

} // namespace

Result<void*>
CudaAllocator::allocate(size_t nbytes, DeviceIndex index, size_t alignment) {
  // index == -1 means "use the current CUDA device"; any value < -1 is invalid.
  ET_CHECK_OR_RETURN_ERROR(
      index >= -1,
      InvalidArgument,
      "CudaAllocator::allocate: invalid device index %d (must be >= -1)",
      static_cast<int>(index));

  // Alignment must be a non-zero power of 2.
  ET_CHECK_OR_RETURN_ERROR(
      alignment != 0 && (alignment & (alignment - 1)) == 0,
      InvalidArgument,
      "CudaAllocator::allocate: alignment must be a power of 2, got %zu",
      alignment);

  // cudaMalloc guarantees 256 bytes. A larger alignment pads one allocation
  // instead of retrying, because freeing a misaligned block would wait for all
  // work queued on the device.
  constexpr size_t kCudaMallocAlignment = 256;
  const size_t padding = alignment > kCudaMallocAlignment ? alignment - 1 : 0;
  ET_CHECK_OR_RETURN_ERROR(
      nbytes <= std::numeric_limits<size_t>::max() - padding,
      InvalidArgument,
      "CudaAllocator::allocate: size %zu with alignment %zu overflows",
      nbytes,
      alignment);

  int prev_device = 0;
  bool switch_device = false;

  // If index == -1, fall back to the current device and skip the set/restore
  // round-trip.
  if (index >= 0) {
    // Without the current device there is no way to switch to `index` and no
    // way to restore afterwards, so the allocation would land on whatever
    // device happens to be current while the caller records it as living on
    // `index`. Fail instead.
    cudaError_t prev_device_err = cudaGetDevice(&prev_device);
    if (prev_device_err != cudaSuccess) {
      ET_LOG(
          Error,
          "CudaAllocator::allocate: cudaGetDevice failed: %s",
          cudaGetErrorString(prev_device_err));
      // Every failure return in allocate() clears the CUDA error it leaves:
      // the return value reports the failure, and a pending error would
      // surface in the next cudaGetLastError(), e.g. a kernel launch check, as
      // if that call had failed.
      (void)cudaGetLastError();
      return Error::Internal;
    }
    switch_device = static_cast<int>(index) != prev_device;
  }

  if (switch_device) {
    cudaError_t set_err = cudaSetDevice(index);
    if (set_err != cudaSuccess) {
      // Allocating now would return a pointer on the current device while the
      // caller records it as living on the requested one. cudaSetDevice reports
      // more than a bad ordinal here (a valid device can be unavailable or in
      // prohibited mode), so report it the way the rest of the CUDA runtime
      // code does rather than blaming the caller's argument.
      ET_LOG(
          Error,
          "CudaAllocator::allocate: cudaSetDevice(%d) failed: %s",
          static_cast<int>(index),
          cudaGetErrorString(set_err));
      (void)cudaGetLastError();
      return Error::Internal;
    }
  }

  void* block = nullptr;
  const cudaError_t err = cudaMalloc(&block, nbytes + padding);

  if (switch_device) {
    (void)cudaSetDevice(prev_device);
  }

  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "CudaAllocator::allocate: cudaMalloc of %zu bytes failed: %s "
        "(requested %zu bytes with alignment %zu on device %d)",
        nbytes + padding,
        cudaGetErrorString(err),
        nbytes,
        alignment,
        static_cast<int>(index));
    (void)cudaGetLastError();
    return Error::MemoryAllocationFailed;
  }

  if (padding != 0) {
    const size_t offset = -reinterpret_cast<uintptr_t>(block) & padding;
    if (offset == 0) {
      return block;
    }
    void* const ptr = static_cast<char*>(block) + offset;
    auto& rounded = rounded_allocations();
    const std::lock_guard<std::mutex> lock(rounded.mutex);
    rounded.original_by_rounded.emplace(ptr, block);
    rounded.count.fetch_add(1, std::memory_order_relaxed);
    return ptr;
  }

  if ((reinterpret_cast<uintptr_t>(block) & (alignment - 1)) != 0) {
    ET_LOG(
        Error,
        "cudaMalloc returned pointer %p not aligned to %zu bytes",
        block,
        alignment);
    (void)cudaFree(block);
    (void)cudaGetLastError();
    return Error::NotSupported;
  }

  return block;
}

void CudaAllocator::deallocate(void* ptr, DeviceIndex index) {
  if (ptr == nullptr) {
    return;
  }

  auto& rounded = rounded_allocations();
  // Skips the lock while no rounded pointer is live. Relaxed is enough: a
  // pointer reaches deallocate() through a handoff that orders it after the
  // allocate() that counted it.
  if (rounded.count.load(std::memory_order_relaxed) != 0) {
    const std::lock_guard<std::mutex> lock(rounded.mutex);
    const auto it = rounded.original_by_rounded.find(ptr);
    if (it != rounded.original_by_rounded.end()) {
      ptr = it->second;
      rounded.original_by_rounded.erase(it);
      rounded.count.fetch_sub(1, std::memory_order_relaxed);
    }
  }

  int prev_device = 0;
  bool switched_device = false;

  if (index >= 0) {
    cudaError_t prev_device_err = cudaGetDevice(&prev_device);
    if (prev_device_err != cudaSuccess) {
      // cudaFree accepts a pointer from any device under unified addressing, so
      // free it anyway rather than leak, but do not try to restore a device we
      // never read.
      ET_LOG(
          Error,
          "CudaAllocator::deallocate: cudaGetDevice failed: %s",
          cudaGetErrorString(prev_device_err));
    } else if (static_cast<int>(index) != prev_device) {
      cudaError_t set_err = cudaSetDevice(index);
      if (set_err != cudaSuccess) {
        // Same reasoning: keep going rather than leak it, but do not stay
        // silent about it.
        ET_LOG(
            Error,
            "CudaAllocator::deallocate: cudaSetDevice(%d) failed: %s",
            static_cast<int>(index),
            cudaGetErrorString(set_err));
      } else {
        switched_device = true;
      }
    }
  }

  cudaError_t err = cudaFree(ptr);

  if (switched_device) {
    (void)cudaSetDevice(prev_device);
  }

  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaFree failed: %s (ptr=%p, device %d)",
        cudaGetErrorString(err),
        ptr,
        static_cast<int>(index));
  }
}

Error CudaAllocator::copy_host_to_device(
    void* dst,
    const void* src,
    size_t nbytes,
    DeviceIndex index) {
  return copy_impl(dst, src, nbytes, index, cudaMemcpyHostToDevice);
}

Error CudaAllocator::copy_device_to_host(
    void* dst,
    const void* src,
    size_t nbytes,
    DeviceIndex index) {
  return copy_impl(dst, src, nbytes, index, cudaMemcpyDeviceToHost);
}

DeviceType CudaAllocator::device_type() const {
  return DeviceType::CUDA;
}

CudaAllocator& CudaAllocator::instance() {
  // Registered allocators can be called until the process exits.
  static auto* allocator = new CudaAllocator();
  return *allocator;
}

Result<void*> CudaAllocator::allocate_async(
    size_t nbytes,
    DeviceIndex index,
    cudaStream_t stream) {
  void* ptr = nullptr;
  cudaError_t err;
  // Named for the log below, so a failure points at the call that actually ran.
  const char* allocator_name = "cudaMallocAsync";
  int log_device = static_cast<int>(index);
#if defined(EXECUTORCH_USE_HIP)
  err = cudaMallocAsync(&ptr, nbytes, stream);
#else
  // Allocating from this allocator's pool keeps its retained memory out of
  // the device default pool, which other users of the async allocator share.
  //
  // The pool has to belong to the device the stream runs on. A pool from
  // another device returns a pointer that stream cannot touch, and the failure
  // surfaces later as an illegal access rather than here. The caller's index
  // and the stream can disagree, so the plain async allocation is used unless
  // the index names the device that is current for this stream.
  const int device = resolve_device(index);
  int stream_device = -1;
  if (cudaGetDevice(&stream_device) != cudaSuccess) {
    (void)cudaGetLastError();
    stream_device = -1;
  }
  const int target = device >= 0 ? device : stream_device;
  if (target >= 0 && !device_supports_memory_pools(target)) {
    // Without memory pools there is no stream-ordered allocation to make. A
    // synchronous allocation is usable from any stream at once, so the caller's
    // ordering still holds; deallocate_async frees it to match.
    return allocate_without_pools(nbytes, index, stream);
  }
  cudaMemPool_t pool =
      (device >= 0 && device == stream_device) ? mem_pool_for(device) : nullptr;
  if (device >= 0) {
    log_device = device;
  }
  if (pool != nullptr) {
    allocator_name = "cudaMallocFromPoolAsync";
    err = cudaMallocFromPoolAsync(&ptr, nbytes, pool, stream);
  } else {
    err = cudaMallocAsync(&ptr, nbytes, stream);
  }
#endif
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "%s failed: %s (requested %zu bytes on device %d)",
        allocator_name,
        cudaGetErrorString(err),
        nbytes,
        log_device);
    // Reported through the return value, as in allocate().
    (void)cudaGetLastError();
    return Error::MemoryAllocationFailed;
  }

  return ptr;
}

void CudaAllocator::deallocate_async(
    void* ptr,
    DeviceIndex index,
    cudaStream_t stream) {
  if (ptr == nullptr) {
    return;
  }

#if !defined(EXECUTORCH_USE_HIP)
  // Decided by the device the memory lives on, the same answer allocate_async
  // reached for it. Looked up only once some device without memory pools has
  // been seen, so a process whose devices all have them frees as before.
  if (g_device_without_pools_seen.load(std::memory_order_acquire)) {
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, ptr) == cudaSuccess) {
      if (attributes.type == cudaMemoryTypeDevice && attributes.device >= 0 &&
          !device_supports_memory_pools(attributes.device)) {
        (void)deallocate_without_pools(
            ptr, static_cast<DeviceIndex>(attributes.device), stream);
        return;
      }
    } else {
      (void)cudaGetLastError();
    }
  }
#endif

  cudaError_t err = cudaFreeAsync(ptr, stream);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaFreeAsync failed: %s (ptr=%p, device %d)",
        cudaGetErrorString(err),
        ptr,
        static_cast<int>(index));
  }
}

bool CudaAllocator::memory_pools_supported(DeviceIndex index) {
#if defined(EXECUTORCH_USE_HIP)
  (void)index;
  return true;
#else
  const int device = resolve_device(index);
  return device < 0 || device_supports_memory_pools(device);
#endif
}

Result<void*> CudaAllocator::allocate_stream_ordered(
    size_t nbytes,
    DeviceIndex index,
    cudaStream_t stream) {
#if !defined(EXECUTORCH_USE_HIP)
  const int device = resolve_device(index);
  if (device >= 0 && !device_supports_memory_pools(device)) {
    return allocate_without_pools(nbytes, index, stream);
  }
#endif
  void* ptr = nullptr;
  const cudaError_t err = cudaMallocAsync(&ptr, nbytes, stream);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaMallocAsync failed: %s (requested %zu bytes on device %d)",
        cudaGetErrorString(err),
        nbytes,
        static_cast<int>(index));
    // Reported through the return value, as in allocate().
    (void)cudaGetLastError();
    return Error::MemoryAllocationFailed;
  }
  return ptr;
}

executorch::runtime::Error CudaAllocator::deallocate_stream_ordered(
    void* ptr,
    DeviceIndex index,
    cudaStream_t stream) {
  if (ptr == nullptr) {
    return Error::Ok;
  }
#if !defined(EXECUTORCH_USE_HIP)
  const int device = resolve_device(index);
  if (device >= 0 && !device_supports_memory_pools(device)) {
    return deallocate_without_pools(ptr, index, stream);
  }
#endif
  const cudaError_t err = cudaFreeAsync(ptr, stream);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaFreeAsync failed: %s (ptr=%p, device %d)",
        cudaGetErrorString(err),
        ptr,
        static_cast<int>(index));
    return Error::Internal;
  }
  return Error::Ok;
}

#if !defined(EXECUTORCH_USE_HIP)
cudaMemPool_t CudaAllocator::pool_for_device(DeviceIndex index) {
  const int device = resolve_device(index);
  if (device < 0) {
    return nullptr;
  }
  auto& state = mem_pool_state();
  const std::lock_guard<std::mutex> lock(state.mutex);
  const auto it = state.pools.find(device);
  return it == state.pools.end() ? nullptr : it->second;
}
#endif // !EXECUTORCH_USE_HIP

void CudaAllocator::release_cached_memory(DeviceIndex index) {
#if defined(EXECUTORCH_USE_HIP)
  (void)index;
#else
  std::vector<std::pair<int, cudaMemPool_t>> targets;
  {
    auto& state = mem_pool_state();
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (index >= 0) {
      const auto it = state.pools.find(static_cast<int>(index));
      if (it == state.pools.end()) {
        return;
      }
      targets.emplace_back(it->first, it->second);
    } else {
      // A caller asking for everything gets every pool this allocator created,
      // not whichever device the calling thread happens to be current on, since
      // the delegate that ran is often not on that device.
      targets.assign(state.pools.begin(), state.pools.end());
    }
  }

  // The pools stay in the map. Trimming empties one without invalidating it, so
  // a later load reuses it rather than paying to create it again.
  for (const auto& [device, pool] : targets) {
    if (pool != nullptr) {
      // Only frees the driver has already observed can be released, and a free
      // still pending gives back nothing rather than less. The backend waits on
      // its stream during teardown for that reason, before dropping it. This
      // cannot wait on anything itself: it holds no stream, and a caller
      // reaching it directly is responsible for having synchronized.
      const cudaError_t err = cudaMemPoolTrimTo(pool, 0);
      if (err != cudaSuccess) {
        ET_LOG(
            Error,
            "cudaMemPoolTrimTo failed for device %d: %s.",
            device,
            cudaGetErrorString(err));
        (void)cudaGetLastError();
      }
    }

    // Memory a method allocated while its CUDA graph was being captured belongs
    // to the device graph pool, which the pool trim cannot reach, so a
    // graph-enabled method would otherwise hold its footprint for the life of
    // the process. Outside the branch above because graph memory is a device
    // resource and exists whether or not this allocator has a pool here.
    //
    // Device scoped, unlike everything else in this function: it releases
    // unused graph memory cached by every user of the device, so another
    // library in this process pays to map its own graph allocations again.
    // Nothing breaks, since only unused blocks go.
    const cudaError_t graph_err = cudaDeviceGraphMemTrim(device);
    if (graph_err != cudaSuccess) {
      ET_LOG(
          Error,
          "cudaDeviceGraphMemTrim failed for device %d: %s.",
          device,
          cudaGetErrorString(graph_err));
      (void)cudaGetLastError();
    }
  }
#endif
}

Error CudaAllocator::memcpy_async(
    void* dst,
    const void* src,
    size_t nbytes,
    cudaMemcpyKind direction,
    cudaStream_t stream) {
  cudaError_t err = cudaMemcpyAsync(dst, src, nbytes, direction, stream);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaMemcpyAsync failed: %s (%zu bytes)",
        cudaGetErrorString(err),
        nbytes);
    return Error::Internal;
  }
  return Error::Ok;
}

} // namespace executorch::extension::cuda
