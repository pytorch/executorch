#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/Contiguity.h>
#include <executorch/backends/aoti/slim/c10/core/MemoryFormat.h>
#include <executorch/backends/aoti/slim/c10/core/Scalar.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/c10/core/SizesAndStrides.h>
#include <executorch/backends/aoti/slim/c10/core/WrapDimMinimal.h>
#include <executorch/backends/aoti/slim/c10/util/safe_numerics.h>
#include <executorch/backends/aoti/slim/core/Storage.h>
#include <executorch/backends/aoti/slim/util/SizeUtil.h>

namespace executorch::backends::aoti::slim {

class SlimTensor {
 public:
  SlimTensor(
      Storage&& storage,
      executorch::backends::aoti::slim::c10::IntArrayRef sizes,
      executorch::backends::aoti::slim::c10::IntArrayRef strides,
      executorch::backends::aoti::slim::c10::ScalarType dtype,
      int64_t storage_offset = 0)
      : storage_(std::move(storage)),
        storage_offset_(storage_offset),
        dtype_(dtype) {
    set_sizes_and_strides(sizes, strides);
  }

  // Default constructor - creates an undefined tensor
  SlimTensor()
      : storage_(Storage()),
        storage_offset_(0),
        numel_(0),
        dtype_(executorch::backends::aoti::slim::c10::ScalarType::Float),
        is_contiguous_(true) {
    sizes_and_strides_.set_sizes({0});
    sizes_and_strides_.set_strides({1});
  }

  SlimTensor(const SlimTensor&) = default;
  SlimTensor& operator=(const SlimTensor&) = default;
  SlimTensor(SlimTensor&&) = default;
  SlimTensor& operator=(SlimTensor&&) = default;

  ~SlimTensor() = default;

  void reset() {
    // Decrement the refcount of the storage
    storage_.reset();
  }

  // Accessors
  Storage storage() const {
    return storage_;
  }

  size_t nbytes() const {
    return numel() * itemsize();
  }

  size_t itemsize() const {
    return executorch::backends::aoti::slim::c10::elementSize(dtype_);
  }

  executorch::backends::aoti::slim::c10::IntArrayRef sizes() const {
    return sizes_and_strides_.sizes_arrayref();
  }

  int64_t size(int64_t dim) const {
    int64_t wrapped_dim = executorch::backends::aoti::slim::c10::maybe_wrap_dim(
        dim, static_cast<int64_t>(this->dim()));
    return sizes_and_strides_.size_at(static_cast<size_t>(wrapped_dim));
  }

  executorch::backends::aoti::slim::c10::IntArrayRef strides() const {
    return sizes_and_strides_.strides_arrayref();
  }

  int64_t stride(int64_t dim) const {
    int64_t wrapped_dim = executorch::backends::aoti::slim::c10::maybe_wrap_dim(
        dim, static_cast<int64_t>(this->dim()));
    return sizes_and_strides_.stride_at(static_cast<size_t>(wrapped_dim));
  }

  executorch::backends::aoti::slim::c10::ScalarType dtype() const {
    return dtype_;
  }

  const executorch::backends::aoti::slim::c10::Device& device() const {
    return storage_->device();
  }

  executorch::backends::aoti::slim::c10::DeviceType device_type() const {
    return storage_->device().type();
  }

  executorch::backends::aoti::slim::c10::DeviceIndex device_index() const {
    return storage_->device().index();
  }

  int64_t storage_offset() const {
    return storage_offset_;
  }

  size_t numel() const {
    return numel_;
  }

  size_t dim() const {
    return sizes_and_strides_.size();
  }

  void* data_ptr() const {
    return static_cast<char*>(storage_->data()) + storage_offset_ * itemsize();
  }

  bool is_contiguous() const {
    return is_contiguous_;
  }

  bool is_empty() const {
    return numel_ == 0;
  }

  bool is_cuda() const {
    return device().is_cuda();
  }

  bool is_cpu() const {
    return device().is_cpu();
  }

  // Check if tensor is defined (not default-constructed)
  bool defined() const {
    return storage_.get() != nullptr;
  }

  // Setters
  void set_storage(Storage&& new_storage) {
    storage_ = std::move(new_storage);
  }

  void set_sizes_and_strides(
      executorch::backends::aoti::slim::c10::IntArrayRef sizes,
      executorch::backends::aoti::slim::c10::IntArrayRef strides,
      std::optional<int64_t> storage_offset = std::nullopt) {
    const int64_t new_dim = static_cast<int64_t>(sizes.size());
    STANDALONE_CHECK(
        new_dim == static_cast<int64_t>(strides.size()),
        "dimensionality of sizes (",
        new_dim,
        ") must match dimensionality of strides (",
        strides.size(),
        ")");

    std::vector<int64_t> new_sizes = sizes.vec();
    std::vector<int64_t> new_strides = strides.vec();

    // stride calculation logic
    bool overflowed = false;
    if (new_dim > 0) {
      for (int64_t dim = new_dim - 1; dim >= 0; dim--) {
        if (strides[dim] >= 0) {
          new_strides[dim] = strides[dim];
        } else {
          // for negative strides
          if (dim == new_dim - 1) {
            new_strides[dim] = 1;
          } else {
            overflowed |= executorch::backends::aoti::slim::c10::mul_overflows(
                new_strides[dim + 1],
                std::max<int64_t>(new_sizes[dim + 1], 1),
                &new_strides[dim]);
          }
        }
      }
    }
    STANDALONE_CHECK(!overflowed, "Stride calculation overflowed");

    sizes_and_strides_.set_sizes(new_sizes);
    sizes_and_strides_.set_strides(new_strides);
    if (storage_offset.has_value()) {
      storage_offset_ = *storage_offset;
    }

    refresh_numel();
    refresh_contiguous();
  }

  void set_sizes_contiguous(
      executorch::backends::aoti::slim::c10::IntArrayRef new_size) {
    sizes_and_strides_.set_sizes(new_size);
    refresh_numel();
    empty_tensor_restride(
        executorch::backends::aoti::slim::c10::MemoryFormat::Contiguous);
  }

  void empty_tensor_restride(
      executorch::backends::aoti::slim::c10::MemoryFormat memory_format);

  SlimTensor resize_(
      executorch::backends::aoti::slim::c10::IntArrayRef sizes,
      std::optional<c10::MemoryFormat> optional_memory_format);

  // Conversion operations
  SlimTensor to(
      const executorch::backends::aoti::slim::c10::Device& device) const {
    if (device == storage_->device()) {
      return *this;
    }
    // Does not mutate the current tensor. Returns a new tensor
    Storage new_storage(new MaybeOwningStorage(storage_->clone(device)));
    return SlimTensor(
        std::move(new_storage),
        sizes_and_strides_.sizes_arrayref(),
        sizes_and_strides_.strides_arrayref(),
        dtype_,
        storage_offset_);
  }

  SlimTensor cpu() const {
    return to(CPU_DEVICE);
  }

  SlimTensor cuda() const {
    return to(DEFAULT_CUDA_DEVICE);
  }

  SlimTensor to(executorch::backends::aoti::slim::c10::ScalarType dtype) const {
    STANDALONE_CHECK(false, "TBD: to(dtype)");
  }

  SlimTensor& copy_(const SlimTensor& other) {
    STANDALONE_CHECK(
        this->numel() == other.numel(), "copy_: numel of tensors must match");
    STANDALONE_CHECK(this->dtype() == other.dtype(), "copy_: dtype must match");

    if (this->numel() == 0) {
      return *this;
    }

    // Case 1: Both tensors are contiguous. We can do a fast bulk copy.
    if (this->is_contiguous() && other.is_contiguous()) {
      storage_->copy_(
          this->data_ptr(), other.data_ptr(), other.nbytes(), other.device());
      return *this;
    }

    // Case 2: At least one tensor is non-contiguous, perform element-wise copy
    // that respects both source and destination strides.
    const size_t elem_size =
        executorch::backends::aoti::slim::c10::elementSize(dtype_);
    char* dst_data = static_cast<char*>(this->data_ptr());
    const char* src_data = static_cast<const char*>(other.data_ptr());

    std::vector<int64_t> counter(this->dim(), 0);
    for (size_t i = 0; i < this->numel(); i++) {
      // Compute src offset in elements
      int64_t src_offset = 0;
      for (size_t d = 0; d < other.dim(); d++) {
        src_offset += counter[d] * other.stride(d);
      }

      // Compute dst offset in elements
      int64_t dst_offset = 0;
      for (size_t d = 0; d < this->dim(); d++) {
        dst_offset += counter[d] * this->stride(d);
      }

      // Copy elem_size bytes from src to dst
      if (this->device().is_cpu() && other.device().is_cpu()) {
        std::memcpy(
            dst_data + dst_offset * elem_size,
            src_data + src_offset * elem_size,
            elem_size);
      } else if (this->device().is_cuda() || other.device().is_cuda()) {
#if defined(USE_CUDA)
        DeviceTraits<c10::DeviceType::CUDA>::memcpy(
            dst_data + dst_offset * elem_size,
            src_data + src_offset * elem_size,
            elem_size,
            device(), // dst device
            other.device() // src device
        );
#else
        STANDALONE_CHECK(false, "copy_: no CUDA support");
#endif
      }
      // Increment the multi-dimensional counter
      for (int64_t d = static_cast<int64_t>(this->dim()) - 1; d >= 0; --d) {
        counter[d]++;
        if (counter[d] < this->size(d)) {
          break;
        }
        counter[d] = 0;
      }
    }
    return *this;
  }

  SlimTensor& fill_(const c10::Scalar& value) {
    // Fast path for byte patterns on contiguous tensors - use memset
    if (value.equal(0) && this->is_contiguous()) {
      if (this->device().is_cpu()) {
        std::memset(this->data_ptr(), 0, this->nbytes());
        return *this;
      } else if (this->device().is_cuda()) {
#ifdef USE_CUDA
        cudaError_t err = cudaMemset(this->data_ptr(), 0, this->nbytes());
        STANDALONE_CHECK(
            err == cudaSuccess,
            "CUDA memset failed: ",
            cudaGetErrorString(err));
        return *this;
#else
        STANDALONE_CHECK(false, "CUDA support not available");
#endif
      }
    }

    // Fallback to type-specific fill implementation
    auto fill_value = [&](auto typed_value) {
      using SType = decltype(typed_value);
      if (this->device().is_cuda()) {
#ifdef USE_CUDA
        if (this->is_contiguous()) {
          // Fast path for contiguous tensors
          if constexpr (std::is_same_v<SType, bool>) {
            // Special handling for bool since std::vector<bool> doesn't have
            // data()
            std::vector<uint8_t> host_data(this->numel(), typed_value ? 1 : 0);
            cudaError_t err = cudaMemcpy(
                this->data_ptr(),
                host_data.data(),
                this->nbytes(),
                cudaMemcpyHostToDevice);
            STANDALONE_CHECK(
                err == cudaSuccess,
                "CUDA memcpy failed: ",
                cudaGetErrorString(err));
          } else {
            std::vector<SType> host_data(this->numel(), typed_value);
            cudaError_t err = cudaMemcpy(
                this->data_ptr(),
                host_data.data(),
                this->nbytes(),
                cudaMemcpyHostToDevice);
            STANDALONE_CHECK(
                err == cudaSuccess,
                "CUDA memcpy failed: ",
                cudaGetErrorString(err));
          }
        } else {
          // Handle non-contiguous tensors by copying to CPU, filling, then
          // copying back
          SlimTensor cpu_tensor = this->to(CPU_DEVICE);
          cpu_tensor.fill_(typed_value);
          this->copy_(cpu_tensor);
        }
#else
        STANDALONE_CHECK(false, "CUDA support not available");
#endif
      } else if (this->device().is_cpu()) {
        if (this->is_contiguous()) {
          // Fast path for contiguous tensors
          SType* data = static_cast<SType*>(this->data_ptr());
          for (size_t i = 0; i < this->numel(); ++i) {
            data[i] = typed_value;
          }
        } else {
          // Handle non-contiguous tensors by respecting strides
          const size_t elem_size =
              executorch::backends::aoti::slim::c10::elementSize(this->dtype_);
          char* base_data = static_cast<char*>(this->data_ptr());

          std::vector<int64_t> counter(this->dim(), 0);
          for (size_t i = 0; i < this->numel(); ++i) {
            // Compute offset in elements based on strides
            int64_t offset = 0;
            for (size_t d = 0; d < this->dim(); d++) {
              offset += counter[d] * this->stride(d);
            }

            // Set the value at the computed offset
            SType* element_ptr =
                reinterpret_cast<SType*>(base_data + offset * elem_size);
            *element_ptr = typed_value;

            // Increment the multi-dimensional counter
            for (int64_t d = static_cast<int64_t>(this->dim()) - 1; d >= 0;
                 --d) {
              counter[d]++;
              if (counter[d] < this->size(d)) {
                break;
              }
              counter[d] = 0;
            }
          }
        }
      }
    };

    switch (this->dtype()) {
      case executorch::backends::aoti::slim::c10::ScalarType::Double:
        fill_value(value.to<double>());
        break;
      case executorch::backends::aoti::slim::c10::ScalarType::Float:
        fill_value(value.to<float>());
        break;
      case executorch::backends::aoti::slim::c10::ScalarType::Half:
        fill_value(value.to<executorch::backends::aoti::slim::c10::Half>());
        break;
      case executorch::backends::aoti::slim::c10::ScalarType::BFloat16:
        fill_value(value.to<executorch::backends::aoti::slim::c10::BFloat16>());
        break;
      case executorch::backends::aoti::slim::c10::ScalarType::Long:
        fill_value(value.to<int64_t>());
        break;
      case executorch::backends::aoti::slim::c10::ScalarType::Int:
        fill_value(value.to<int32_t>());
        break;
      case executorch::backends::aoti::slim::c10::ScalarType::Short:
        fill_value(value.to<int16_t>());
        break;
      case executorch::backends::aoti::slim::c10::ScalarType::Char:
        fill_value(value.to<int8_t>());
        break;
      case executorch::backends::aoti::slim::c10::ScalarType::Byte:
        fill_value(value.to<uint8_t>());
        break;
      case executorch::backends::aoti::slim::c10::ScalarType::Bool:
        fill_value(value.to<bool>());
        break;
      case executorch::backends::aoti::slim::c10::ScalarType::ComplexFloat:
        fill_value(
            value.to<executorch::backends::aoti::slim::c10::complex<float>>());
        break;
      case executorch::backends::aoti::slim::c10::ScalarType::ComplexDouble:
        fill_value(
            value.to<executorch::backends::aoti::slim::c10::complex<double>>());
        break;
      default:
        STANDALONE_CHECK(false, "fill_: Unsupported dtype");
    }
    return *this;
  }

  SlimTensor clone() const {
    return _clone_impl(
        this->sizes(), this->strides(), this->dtype(), this->device());
  }

  SlimTensor clone_contiguous() const {
    std::vector<int64_t> contig_strides =
        executorch::backends::aoti::slim::compute_contiguous_strides(
            this->sizes());
    return _clone_impl(
        this->sizes(), contig_strides, this->dtype(), this->device());
  }

  // View operations
  SlimTensor as_strided(
      executorch::backends::aoti::slim::c10::IntArrayRef sizes,
      executorch::backends::aoti::slim::c10::IntArrayRef strides,
      int64_t storage_offset) const;
  SlimTensor as_strided_(
      executorch::backends::aoti::slim::c10::IntArrayRef sizes,
      executorch::backends::aoti::slim::c10::IntArrayRef strides,
      int64_t storage_offset);

  SlimTensor permute(
      executorch::backends::aoti::slim::c10::IntArrayRef dims) const;

  // Transpose operations
  SlimTensor transpose() const;
  SlimTensor transpose(int64_t dim0, int64_t dim1) const;
  SlimTensor t() const;

  SlimTensor reshape(
      executorch::backends::aoti::slim::c10::IntArrayRef proposed_shape) const;

  SlimTensor narrow(int64_t dim, int64_t start, int64_t length) const;

  // Generic element access returning SlimTensor
  SlimTensor operator[](
      executorch::backends::aoti::slim::c10::IntArrayRef indices) const {
    STANDALONE_CHECK(
        indices.size() <= this->dim(),
        "Number of indices (",
        indices.size(),
        ") cannot exceed tensor dimensions (",
        this->dim(),
        ")");

    if (indices.size() == this->dim()) {
      // Full indexing - return 0-dimensional tensor
      int64_t linear_index = 0;
      for (size_t i = 0; i < indices.size(); ++i) {
        int64_t idx = indices[i];
        int64_t size = this->size(i);
        idx = executorch::backends::aoti::slim::c10::maybe_wrap_dim(idx, size);
        linear_index += idx * this->stride(i);
      }
      // Create 0-dimensional tensor pointing to the indexed element
      int64_t new_storage_offset = this->storage_offset_ + linear_index;
      return SlimTensor(
          Storage(this->storage_), {}, {}, this->dtype_, new_storage_offset);
    } else {
      // Partial indexing - return tensor with reduced dimensions
      std::vector<int64_t> new_sizes;
      std::vector<int64_t> new_strides;
      int64_t offset_adjustment = 0;

      // Calculate offset from the provided indices
      for (size_t i = 0; i < indices.size(); ++i) {
        int64_t idx = indices[i];
        int64_t size = this->size(i);
        idx = executorch::backends::aoti::slim::c10::maybe_wrap_dim(idx, size);
        offset_adjustment += idx * this->stride(i);
      }

      // Copy remaining dimensions
      for (size_t i = indices.size(); i < this->dim(); ++i) {
        new_sizes.push_back(this->size(i));
        new_strides.push_back(this->stride(i));
      }

      int64_t new_storage_offset = this->storage_offset_ + offset_adjustment;
      return SlimTensor(
          Storage(this->storage_),
          new_sizes,
          new_strides,
          this->dtype_,
          new_storage_offset);
    }
  }

  // Convenience overload for single index
  SlimTensor operator[](int64_t index) const {
    return (*this)[executorch::backends::aoti::slim::c10::IntArrayRef{index}];
  }

  // Convenience overloads for common multi-dimensional cases
  SlimTensor operator[](std::initializer_list<int64_t> indices) const {
    return (*this)[executorch::backends::aoti::slim::c10::IntArrayRef(indices)];
  }

  // Extract scalar value from 0-dimensional tensor
  executorch::backends::aoti::slim::c10::Scalar item() const {
    switch (this->dtype()) {
      case executorch::backends::aoti::slim::c10::ScalarType::Double:
        return this->item<double>();
      case executorch::backends::aoti::slim::c10::ScalarType::Float:
        return this->item<float>();
      case executorch::backends::aoti::slim::c10::ScalarType::Half:
        return this->item<executorch::backends::aoti::slim::c10::Half>();
      case executorch::backends::aoti::slim::c10::ScalarType::BFloat16:
        return this->item<executorch::backends::aoti::slim::c10::BFloat16>();
      case executorch::backends::aoti::slim::c10::ScalarType::Long:
        return this->item<int64_t>();
      case executorch::backends::aoti::slim::c10::ScalarType::Int:
        return this->item<int32_t>();
      case executorch::backends::aoti::slim::c10::ScalarType::Short:
        return this->item<int16_t>();
      case executorch::backends::aoti::slim::c10::ScalarType::Char:
        return this->item<int8_t>();
      case executorch::backends::aoti::slim::c10::ScalarType::Byte:
        return this->item<uint8_t>();
      case executorch::backends::aoti::slim::c10::ScalarType::Bool:
        return this->item<bool>();
      case executorch::backends::aoti::slim::c10::ScalarType::ComplexFloat:
        return this
            ->item<executorch::backends::aoti::slim::c10::complex<float>>();
      case executorch::backends::aoti::slim::c10::ScalarType::ComplexDouble:
        return this
            ->item<executorch::backends::aoti::slim::c10::complex<double>>();
      default:
        STANDALONE_CHECK(false, "item(): Unsupported dtype");
    }
  }

  // Templated version to access 0-dimensional tensor
  template <typename T>
  T item() const {
    STANDALONE_CHECK(
        this->dim() == 0, "item() can only be called on 0-dimensional tensors");
    STANDALONE_CHECK(
        this->numel() == 1, "item() requires tensor to have exactly 1 element");

    // For 0-dimensional tensors, directly access the single element at
    // data_ptr() No need to compute linear index since there's only one element
    const T* data = static_cast<const T*>(this->data_ptr());
    return *data;
  }

 private:
  SlimTensor _clone_impl(
      executorch::backends::aoti::slim::c10::IntArrayRef sizes,
      executorch::backends::aoti::slim::c10::IntArrayRef strides,
      executorch::backends::aoti::slim::c10::ScalarType dtype,
      const executorch::backends::aoti::slim::c10::Device& device) const {
    Storage storage = new_storage(sizes, strides, dtype, device);
    SlimTensor result =
        SlimTensor(std::move(storage), sizes, strides, dtype, 0);
    result.copy_(*this);
    return result;
  }

  void refresh_numel() {
    numel_ = compute_numel(sizes_and_strides_.sizes_arrayref());
  }

  bool compute_is_contiguous() const {
    return executorch::backends::aoti::slim::c10::_compute_contiguous<int64_t>(
        sizes_and_strides_.sizes_arrayref(),
        sizes_and_strides_.strides_arrayref(),
        numel_);
  }

  void refresh_contiguous() {
    // In SlimTensor, we only care about the single is_contiguous_ flag.
    // (because TensorImpl (aten) implementation has other stuff)
    is_contiguous_ = compute_is_contiguous();
  }

  Storage storage_; // device_type_ and device_index_ are stored in storage_
  int64_t storage_offset_{0};
  executorch::backends::aoti::slim::c10::SizesAndStrides sizes_and_strides_;
  // If sizes and strides are empty, the numel is 1!!  However, most of the
  // time, we will immediately set sizes to {0} and reset numel to 0.
  // (Can't do that in the default initializers, because there's no way to
  // spell "allocate a one-element array" for strides_).
  size_t numel_{1};
  executorch::backends::aoti::slim::c10::ScalarType dtype_;
  bool is_contiguous_{true};
  // NOLINTNEXTLINE(clang-diagnostic-unused-private-field)
  std::array<int8_t, 6> reserved_{0}; // padding to align to 8 bytes
};

} // namespace executorch::backends::aoti::slim

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::backends::aoti::slim::SlimTensor;
} // namespace executor
} // namespace torch

#include <executorch/backends/aoti/slim/core/SlimTensorResize-incl.h>
#include <executorch/backends/aoti/slim/core/SlimTensorView-incl.h>
