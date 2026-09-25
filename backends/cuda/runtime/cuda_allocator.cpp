/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_allocator.h>

#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/log.h>

#if !defined(EXECUTORCH_USE_HIP)
#include <mutex>
#include <unordered_map>
#include <vector>
#endif

namespace executorch::backends::cuda {

using executorch::runtime::Error;
using executorch::runtime::Result;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;

namespace {

#if !defined(EXECUTORCH_USE_HIP)
// The stream ordered allocator hands physical memory back to the driver
// whenever a synchronization observes a pending free, so with the default
// release threshold of zero a pool is emptied repeatedly during one inference
// and every allocation has to map memory again, which measured three orders of
// magnitude slower on an embedded board.
//
// The delegate allocates from a pool it creates rather than the device default
// pool, because the default one is shared with every other user of the async
// allocator in this process. Raising the threshold there would make that shared
// pool hold on to memory on their behalf, and trimming it on teardown would
// throw their cached blocks away. Owning the pool means the threshold and the
// trim only ever affect this backend, with no attempt to remember and restore
// somebody else's setting.
constexpr uint64_t kMemPoolReleaseThreshold = UINT64_MAX;

struct MemPoolState {
  std::mutex mutex;
  std::unordered_map<int, cudaMemPool_t> pools;
};

MemPoolState& mem_pool_state() {
  static MemPoolState state;
  return state;
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

// The pool this backend allocates from on a device, creating it on first use.
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

  // cudaMalloc is documented to return memory aligned to at least 256 bytes,
  // which trivially satisfies kDefaultAlignment (alignof(void*)). For any
  // requested alignment <= 256 bytes, the returned pointer is already aligned.
  // Stricter alignment would require over-allocation plus bookkeeping that
  // deallocate() does not currently support, so reject that case.
  constexpr size_t kCudaMallocAlignment = 256;
  ET_CHECK_OR_RETURN_ERROR(
      alignment <= kCudaMallocAlignment,
      NotSupported,
      "CudaAllocator::allocate: requested alignment %zu exceeds cudaMalloc's "
      "guaranteed alignment of %zu bytes; stricter alignment is not supported",
      alignment,
      kCudaMallocAlignment);

  void* ptr = nullptr;
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
      return Error::Internal;
    }
  }

  cudaError_t err = cudaMalloc(&ptr, nbytes);

  if (switch_device) {
    (void)cudaSetDevice(prev_device);
  }

  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaMalloc failed: %s (requested %zu bytes on device %d)",
        cudaGetErrorString(err),
        nbytes,
        static_cast<int>(index));
    return Error::MemoryAllocationFailed;
  }

  // Sanity check: the pointer returned by cudaMalloc should already meet the
  // requested alignment. If a future CUDA runtime weakens this guarantee, we
  // want to fail loudly rather than silently return a misaligned pointer.
  if ((reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) != 0) {
    ET_LOG(
        Error,
        "cudaMalloc returned pointer %p not aligned to %zu bytes",
        ptr,
        alignment);
    (void)cudaFree(ptr);
    return Error::MemoryAllocationFailed;
  }

  return ptr;
}

void CudaAllocator::deallocate(void* ptr, DeviceIndex index) {
  if (ptr == nullptr) {
    return;
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
  static CudaAllocator allocator;
  return allocator;
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
  // Allocating from this backend's own pool keeps its retained memory out of
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
      // A caller asking for everything gets every pool this backend created,
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
    // resource and exists whether or not this backend has a pool here.
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

} // namespace executorch::backends::cuda
