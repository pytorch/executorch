#pragma once
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

#ifdef USE_CUDA
#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>
#include <executorch/backends/aoti/slim/cuda/Guard.h>
#endif

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/c10/util/ArrayRef.h>
#include <executorch/backends/aoti/slim/c10/util/Exception.h>
#include <executorch/backends/aoti/slim/util/SharedPtr.h>
#include <executorch/backends/aoti/slim/util/SizeUtil.h>

namespace standalone::slim {
using DeleterFn = void (*)(void*);

namespace detail {
inline void noop(void*) {}
} // namespace detail

const standalone::c10::Device CPU_DEVICE =
    standalone::c10::Device(standalone::c10::DeviceType::CPU, 0);

const standalone::c10::Device DEFAULT_CUDA_DEVICE =
    standalone::c10::Device(standalone::c10::DeviceType::CUDA, 0);

// standalone::c10::Device traits template for device-specific operations
template <standalone::c10::DeviceType D>
struct DeviceTraits;

// CPU specialization
template <>
struct DeviceTraits<standalone::c10::DeviceType::CPU> {
  static void* allocate(
      size_t nbytes,
      const standalone::c10::Device& device = CPU_DEVICE) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    return malloc(nbytes);
  }

  static void free(void* ptr) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    std::free(ptr);
  }

  static void memcpy(
      void* dst,
      const void* src,
      size_t nbytes,
      const standalone::c10::Device& dst_device,
      const standalone::c10::Device& src_device) {
    std::memcpy(dst, src, nbytes);
  }
};

// CUDA specialization
#ifdef USE_CUDA
template <>
struct DeviceTraits<standalone::c10::DeviceType::CUDA> {
  static void* allocate(size_t nbytes, const standalone::c10::Device& device) {
    standalone::slim::cuda::CUDAGuard guard(device);
    void* data = nullptr;
    STANDALONE_CUDA_CHECK(cudaMalloc(&data, nbytes));
    return data;
  }

  static void free(void* ptr) {
    STANDALONE_CUDA_CHECK_WARN(cudaFree(ptr));
  }

  static void memcpy(
      void* dst,
      const void* src,
      size_t nbytes,
      const standalone::c10::Device& dst_device,
      const standalone::c10::Device& src_device) {
    // Determine the direction
    cudaMemcpyKind direction = cudaMemcpyDeviceToDevice;
    standalone::c10::Device cuda_device =
        dst_device; // Default to destination device

    if (src_device.is_cpu()) {
      direction = cudaMemcpyHostToDevice;
    } else if (dst_device.is_cpu()) {
      direction = cudaMemcpyDeviceToHost;
      cuda_device = src_device; // Use source CUDA device
    } else {
      STANDALONE_CHECK(
          src_device.index() == dst_device.index(),
          "CUDA memcpy failed across different device indices: ",
          src_device.index(),
          "!=",
          dst_device.index());
    }
    // Set up CUDA context for the appropriate device
    standalone::slim::cuda::CUDAGuard guard(cuda_device);
    STANDALONE_CUDA_CHECK(cudaMemcpy(dst, src, nbytes, direction));
  }
};
#else
template <>
struct DeviceTraits<standalone::c10::DeviceType::CUDA> {
  static void* allocate(size_t nbytes, const standalone::c10::Device& device) {
    STANDALONE_CHECK(false, "Build with USE_CUDA=1 to enable CUDA support");
  }

  static void free(void* ptr) {
    STANDALONE_WARN("Build with USE_CUDA=1 to enable CUDA support");
  }

  static void memcpy(
      void* dst,
      const void* src,
      size_t nbytes,
      const standalone::c10::Device& dst_device,
      const standalone::c10::Device& src_device) {
    STANDALONE_CHECK(false, "Build with USE_CUDA=1 to enable CUDA support");
  }
};
#endif

// Storage can be either owning or non-owning. For AOTI-generated intermediate
// tensors, the storage is always owning. For constant tensors, the storage is
// non-owning.
class MaybeOwningStorage {
 public:
  MaybeOwningStorage(const standalone::c10::Device& device, size_t nbytes)
      : device_(device), capacity_(nbytes), is_owning_(true) {
    // Allocating memory here so owning_ has to be true.
    if (device.is_cpu()) {
      data_ = DeviceTraits<standalone::c10::DeviceType::CPU>::allocate(
          nbytes, device);
      deleter_ = DeviceTraits<standalone::c10::DeviceType::CPU>::free;
    } else if (device.is_cuda()) {
      data_ = DeviceTraits<standalone::c10::DeviceType::CUDA>::allocate(
          nbytes, device);
      deleter_ = DeviceTraits<standalone::c10::DeviceType::CUDA>::free;
    } else {
      STANDALONE_CHECK(false, "Unsupported device type");
    }
  }

  MaybeOwningStorage(
      const standalone::c10::Device& device,
      void* data,
      size_t nbytes)
      : device_(device), data_(data), capacity_(nbytes), is_owning_(false) {
    // data pointer is not owned by this object
  }

  MaybeOwningStorage() = delete;
  MaybeOwningStorage& operator=(const MaybeOwningStorage&) = delete;
  MaybeOwningStorage(const MaybeOwningStorage&) = delete;

  // Move constructor
  MaybeOwningStorage(MaybeOwningStorage&& other) noexcept
      : device_(other.device_),
        data_(other.data_),
        capacity_(other.capacity_),
        deleter_(other.deleter_),
        is_owning_(other.is_owning_) {
    // Leave the moved-from object in a safe state
    other.data_ = nullptr;
    other.capacity_ = 0;
    other.deleter_ = detail::noop;
    other.is_owning_ = false;
  }

  // Move assignment operator
  MaybeOwningStorage& operator=(MaybeOwningStorage&& other) noexcept {
    if (this != &other) {
      // Free current resources
      free_data();

      // Transfer ownership from other
      device_ = other.device_;
      data_ = other.data_;
      capacity_ = other.capacity_;
      deleter_ = other.deleter_;
      is_owning_ = other.is_owning_;

      // Leave the moved-from object in a safe state
      other.data_ = nullptr;
      other.capacity_ = 0;
      other.deleter_ = detail::noop;
      other.is_owning_ = false;
    }
    return *this;
  }

  ~MaybeOwningStorage() {
    free_data();
  }

  void copy_(
      void* dst_data_ptr,
      void* src_data_ptr,
      size_t nbytes,
      const standalone::c10::Device& src_device) {
    STANDALONE_CHECK(
        dst_data_ptr, "Storage clone failed: dst_data_ptr can not be nullptr")
    STANDALONE_CHECK(
        src_data_ptr, "Storage clone failed: src_data_ptr can not be nullptr")
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

  MaybeOwningStorage clone(const standalone::c10::Device& device) const {
    STANDALONE_CHECK(
        data_, "Storage clone failed: source data can not be nullptr")
    // Create a new owning storage with the specified device and same capacity
    MaybeOwningStorage cloned_storage(device, capacity_);

    // Copy the data from the current storage to the new storage
    if (device_.is_cpu() && device.is_cpu()) {
      // CPU to CPU copy
      DeviceTraits<standalone::c10::DeviceType::CPU>::memcpy(
          cloned_storage.data_, data_, capacity_, device, device_);
    } else {
      // At least one of the devices is CUDA
      DeviceTraits<standalone::c10::DeviceType::CUDA>::memcpy(
          cloned_storage.data_, data_, capacity_, device, device_);
    }

    return cloned_storage;
  }

  void* data() const {
    // Always return nullptr for zero-sized storage
    if (capacity_ == 0) {
      return nullptr;
    }
    return data_;
  }

  const standalone::c10::Device& device() const {
    return device_;
  }

  size_t nbytes() const {
    return this->capacity_;
  }

  void unsafe_set_to_non_owning() {
    // This is only used when interacting with at::Tensor. When testing
    // standalone AOTI from pytorch, we need to convert the output SlimTensor
    // into at::Tensor, which means the storage ownership should be stolen by
    // at::Tensor. When all the SlimTensors referencing the storage are
    // destroyed, the storage should NOT be freed.
    deleter_ = detail::noop;
    is_owning_ = false;
  }

  bool is_resizable() const {
    return is_owning_;
  }

  void free_data() {
    if (data_ != nullptr) {
      deleter_(data_);
    }
  }

  void set_data_ptr_noswap(void* new_data) {
    data_ = new_data;
  }

  void set_nbytes(size_t new_nbytes) {
    capacity_ = new_nbytes;
  }

 private:
  standalone::c10::Device device_ = CPU_DEVICE;
  void* data_ = nullptr;
  size_t capacity_ = 0;
  DeleterFn deleter_ = detail::noop;
  bool is_owning_ = false;
};

using Storage = SharedPtr<MaybeOwningStorage>;

inline Storage new_storage(
    standalone::c10::IntArrayRef sizes,
    standalone::c10::IntArrayRef strides,
    standalone::c10::ScalarType dtype,
    const standalone::c10::Device& device = CPU_DEVICE) {
  size_t nbytes = compute_storage_nbytes(
      sizes, strides, standalone::c10::elementSize(dtype), 0);
  return Storage(new MaybeOwningStorage(device, nbytes));
}
} // namespace standalone::slim
