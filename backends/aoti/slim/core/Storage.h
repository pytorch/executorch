/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>

#ifdef CUDA_AVAILABLE
#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>
#include <executorch/backends/cuda/runtime/guard.h>
#endif

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/util/ArrayRefUtil.h>
#include <executorch/backends/aoti/slim/util/SharedPtr.h>
#include <executorch/backends/aoti/slim/util/SizeUtil.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::aoti::slim {

/// Type alias for deleter function pointer.
using DeleterFn = void (*)(void*);

namespace detail {
/// No-op deleter for non-owning storage.
inline void noop(void*) {}
} // namespace detail

/// Default CPU device constant.
inline const c10::Device CPU_DEVICE = c10::Device(c10::DeviceType::CPU, 0);

/// Default CUDA device constant.
inline const c10::Device DEFAULT_CUDA_DEVICE =
    c10::Device(c10::DeviceType::CUDA, 0);

/// DeviceTraits template for device-specific operations.
/// Device-specific implementations provide allocate(), free(), and memcpy().
template <c10::DeviceType D>
struct DeviceTraits;

/// CPU specialization of DeviceTraits.
/// Provides CPU memory allocation and copy operations using malloc/free/memcpy.
template <>
struct DeviceTraits<c10::DeviceType::CPU> {
  /// Allocates CPU memory using malloc.
  /// @param nbytes Number of bytes to allocate.
  /// @param device The target device (unused for CPU).
  /// @return Pointer to allocated memory.
  static void* allocate(size_t nbytes, const c10::Device& device = CPU_DEVICE) {
    (void)device;
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    return malloc(nbytes);
  }

  /// Frees CPU memory using free.
  /// @param ptr Pointer to memory to free.
  static void free(void* ptr) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    std::free(ptr);
  }

  /// Copies memory between CPU locations.
  /// @param dst Destination pointer.
  /// @param src Source pointer.
  /// @param nbytes Number of bytes to copy.
  /// @param dst_device Destination device (unused for CPU-to-CPU).
  /// @param src_device Source device (unused for CPU-to-CPU).
  static void memcpy(
      void* dst,
      const void* src,
      size_t nbytes,
      const c10::Device& dst_device,
      const c10::Device& src_device) {
    (void)dst_device;
    (void)src_device;
    std::memcpy(dst, src, nbytes);
  }
};

#ifdef CUDA_AVAILABLE
/// CUDA specialization of DeviceTraits.
/// Provides CUDA memory allocation and copy operations using
/// cudaMallocAsync/cudaFreeAsync with proper stream handling.
///
/// IMPORTANT: Callers are expected to set the correct CUDA device and stream
/// using CUDAStreamGuard before calling these methods. This is consistent
/// with PyTorch's CUDACachingAllocator design pattern where the allocator
/// assumes the caller has already set the correct device context.
template <>
struct DeviceTraits<c10::DeviceType::CUDA> {
  /// Allocates CUDA device memory on the current stream.
  /// Uses cudaMallocAsync for asynchronous allocation on the stream
  /// that is currently set via CUDAStreamGuard, similar to how
  /// PyTorch's CUDACachingAllocator works.
  ///
  /// NOTE: Caller must ensure the correct device is already set via
  /// CUDAStreamGuard. This function does NOT create a device guard internally.
  ///
  /// @param nbytes Number of bytes to allocate.
  /// @param device The target CUDA device (used to get the stream).
  /// @return Pointer to allocated device memory.
  static void* allocate(size_t nbytes, const c10::Device& device) {
    // Get the current stream for this device (set by CUDAStreamGuard if any)
    // This follows PyTorch's pattern where the allocator assumes the caller
    // has already set the correct device via CUDAStreamGuard.
    auto stream_result =
        executorch::backends::cuda::getCurrentCUDAStream(device.index());
    ET_CHECK_MSG(
        stream_result.ok(),
        "Failed to get current CUDA stream for device %d",
        static_cast<int>(device.index()));

    cudaStream_t stream = stream_result.get();
    void* data = nullptr;
    ET_CUDA_CHECK(cudaMallocAsync(&data, nbytes, stream));
    return data;
  }

  /// Frees CUDA device memory on the current stream.
  /// @param ptr Pointer to device memory to free.
  static void free(void* ptr) {
    // Get the current stream for the current device
    auto stream_result = executorch::backends::cuda::getCurrentCUDAStream(-1);
    if (stream_result.ok()) {
      ET_CUDA_LOG_WARN(cudaFreeAsync(ptr, stream_result.get()));
    } else {
      // Fallback to synchronous free if we can't get the stream
      ET_CUDA_LOG_WARN(cudaFree(ptr));
    }
  }

  /// Copies memory between CPU and CUDA or CUDA and CUDA.
  /// @param dst Destination pointer.
  /// @param src Source pointer.
  /// @param nbytes Number of bytes to copy.
  /// @param dst_device Destination device.
  /// @param src_device Source device.
  static void memcpy(
      void* dst,
      const void* src,
      size_t nbytes,
      const c10::Device& dst_device,
      const c10::Device& src_device) {
    cudaMemcpyKind direction = cudaMemcpyDeviceToDevice;

    if (src_device.is_cpu()) {
      direction = cudaMemcpyHostToDevice;
    } else if (dst_device.is_cpu()) {
      direction = cudaMemcpyDeviceToHost;
    } else {
      ET_CHECK_MSG(
          src_device.index() == dst_device.index(),
          "CUDA memcpy across different device indices not supported: %d != %d",
          static_cast<int>(src_device.index()),
          static_cast<int>(dst_device.index()));
    }

    ET_CUDA_CHECK(cudaMemcpy(dst, src, nbytes, direction));
  }
};
#else
/// CUDA stub when CUDA_AVAILABLE is not defined.
/// All operations abort with an error message.
template <>
struct DeviceTraits<c10::DeviceType::CUDA> {
  static void* allocate(size_t nbytes, const c10::Device& device) {
    (void)nbytes;
    (void)device;
    ET_CHECK_MSG(false, "Build with CUDA_AVAILABLE=1 to enable CUDA support");
  }

  static void free(void* ptr) {
    (void)ptr;
    ET_LOG(Error, "Build with CUDA_AVAILABLE=1 to enable CUDA support");
  }

  static void memcpy(
      void* dst,
      const void* src,
      size_t nbytes,
      const c10::Device& dst_device,
      const c10::Device& src_device) {
    (void)dst;
    (void)src;
    (void)nbytes;
    (void)dst_device;
    (void)src_device;
    ET_CHECK_MSG(false, "Build with CUDA_AVAILABLE=1 to enable CUDA support");
  }
};
#endif // CUDA_AVAILABLE

/**
 * MaybeOwningStorage - A storage class that manages tensor data memory.
 *
 * This class provides owning memory storage for tensor data on CPU.
 * Owning storage allocates and manages its own memory, freeing it upon
 * destruction.
 *
 * Current limitations:
 * - CPU device only
 * - Owning mode only
 * The future diffs will add support for non-owning storage and other devices.
 *
 * Thread Safety: NOT THREAD-SAFE
 * - Uses NonAtomicSharedPtr for reference counting
 * - Must only be used in single-threaded contexts
 */
class MaybeOwningStorage {
 public:
  /// Constructs owning storage with allocated memory.
  /// @param device The device for storage (CPU or CUDA).
  /// @param nbytes Number of bytes to allocate.
  MaybeOwningStorage(const c10::Device& device, size_t nbytes)
      : device_(device), capacity_(nbytes), is_owning_(true) {
    if (device.is_cpu()) {
      data_ = DeviceTraits<c10::DeviceType::CPU>::allocate(nbytes, device);
      deleter_ = DeviceTraits<c10::DeviceType::CPU>::free;
    } else if (device.is_cuda()) {
      data_ = DeviceTraits<c10::DeviceType::CUDA>::allocate(nbytes, device);
      deleter_ = DeviceTraits<c10::DeviceType::CUDA>::free;
    } else {
      ET_CHECK_MSG(false, "Unsupported device type: %s", device.str().c_str());
    }
  }

  /// Default constructor is deleted - storage must have a device.
  MaybeOwningStorage() = delete;

  /// Copy constructor is deleted - use SharedPtr for shared ownership.
  MaybeOwningStorage(const MaybeOwningStorage&) = delete;

  /// Copy assignment is deleted - use SharedPtr for shared ownership.
  MaybeOwningStorage& operator=(const MaybeOwningStorage&) = delete;

  /// Move constructor.
  MaybeOwningStorage(MaybeOwningStorage&& other) noexcept
      : device_(other.device_),
        data_(other.data_),
        capacity_(other.capacity_),
        deleter_(other.deleter_),
        is_owning_(other.is_owning_) {
    other.data_ = nullptr;
    other.capacity_ = 0;
    other.deleter_ = detail::noop;
    other.is_owning_ = false;
  }

  /// Move assignment operator.
  MaybeOwningStorage& operator=(MaybeOwningStorage&& other) noexcept {
    if (this != &other) {
      free_data();

      device_ = other.device_;
      data_ = other.data_;
      capacity_ = other.capacity_;
      deleter_ = other.deleter_;
      is_owning_ = other.is_owning_;

      other.data_ = nullptr;
      other.capacity_ = 0;
      other.deleter_ = detail::noop;
      other.is_owning_ = false;
    }
    return *this;
  }

  /// Destructor - frees owned memory.
  ~MaybeOwningStorage() {
    free_data();
  }

  /// Copies data between storage locations.
  /// @param dst_data_ptr Destination data pointer.
  /// @param src_data_ptr Source data pointer.
  /// @param nbytes Number of bytes to copy.
  /// @param src_device Source device.
  void copy_(
      void* dst_data_ptr,
      void* src_data_ptr,
      size_t nbytes,
      const c10::Device& src_device) {
    ET_CHECK_MSG(
        dst_data_ptr, "Storage copy failed: dst_data_ptr cannot be nullptr");
    ET_CHECK_MSG(
        src_data_ptr, "Storage copy failed: src_data_ptr cannot be nullptr");

    if (dst_data_ptr == src_data_ptr) {
      return;
    }

    if (device_.is_cpu() && src_device.is_cpu()) {
      // CPU to CPU copy
      DeviceTraits<c10::DeviceType::CPU>::memcpy(
          dst_data_ptr, src_data_ptr, nbytes, device_, src_device);
    } else {
      // At least one of the devices is CUDA
      DeviceTraits<c10::DeviceType::CUDA>::memcpy(
          dst_data_ptr, src_data_ptr, nbytes, device_, src_device);
    }
  }

  /// Creates a clone of this storage on the specified device.
  /// @param device Target device for the clone (must be CPU).
  /// @return A new MaybeOwningStorage with copied data.
  MaybeOwningStorage clone(const c10::Device& device) const {
    ET_CHECK_MSG(data_, "Storage clone failed: source data cannot be nullptr");
    ET_CHECK_MSG(
        device.is_cpu(), "Only CPU device is currently supported for clone");

    MaybeOwningStorage cloned_storage(device, capacity_);

    DeviceTraits<c10::DeviceType::CPU>::memcpy(
        cloned_storage.data_, data_, capacity_, device, device_);

    return cloned_storage;
  }

  /// Returns the data pointer, or nullptr for zero-sized storage.
  void* data() const {
    if (capacity_ == 0) {
      return nullptr;
    }
    return data_;
  }

  /// Returns the device this storage is on.
  const c10::Device& device() const {
    return device_;
  }

  /// Returns the capacity in bytes.
  size_t nbytes() const {
    return capacity_;
  }

  /// Returns true if this storage owns its memory.
  bool is_owning() const {
    return is_owning_;
  }

  /// Returns true if the storage can be resized (must be owning).
  bool is_resizable() const {
    return is_owning_;
  }

 private:
  c10::Device device_ = CPU_DEVICE;
  void* data_ = nullptr;
  size_t capacity_ = 0;
  DeleterFn deleter_ = detail::noop;
  bool is_owning_ = false;

  /// Frees the data if non-null.
  void free_data() {
    if (data_ != nullptr) {
      deleter_(data_);
      data_ = nullptr;
    }
  }
};

/// Storage is a shared pointer to MaybeOwningStorage.
/// Multiple tensors can share the same underlying storage.
using Storage = SharedPtr<MaybeOwningStorage>;

/// Creates a new owning storage with the given parameters.
/// @param sizes The sizes of each dimension.
/// @param strides The strides of each dimension.
/// @param dtype The scalar type of tensor elements.
/// @param device The target device (must be CPU).
/// @return A shared pointer to the newly allocated storage.
inline Storage new_storage(
    IntArrayRef sizes,
    IntArrayRef strides,
    c10::ScalarType dtype,
    const c10::Device& device = CPU_DEVICE) {
  size_t nbytes =
      compute_storage_nbytes(sizes, strides, c10::elementSize(dtype), 0);
  return Storage(new MaybeOwningStorage(device, nbytes));
}

} // namespace executorch::backends::aoti::slim
