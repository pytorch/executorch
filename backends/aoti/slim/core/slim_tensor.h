/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include <c10/util/safe_numerics.h>

#include <executorch/backends/aoti/slim/c10/core/Contiguity.h>
#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/c10/core/SizesAndStrides.h>
#include <executorch/backends/aoti/slim/core/storage.h>
#include <executorch/backends/aoti/slim/util/array_ref_util.h>
#include <executorch/backends/aoti/slim/util/size_util.h>
#include <executorch/runtime/platform/assert.h>

namespace executorch::backends::aoti::slim {

/**
 * SlimTensor - A lightweight tensor class for AOTI-driven ET backends runtime.
 *
 */
class SlimTensor {
 public:
  /**
   * Construct a SlimTensor with the given storage, sizes, strides, and dtype.
   *
   * @param storage The underlying storage
   * @param sizes The sizes of each dimension
   * @param strides The strides of each dimension
   * @param dtype The scalar type of tensor elements
   * @param storage_offset Offset into storage in number of elements
   */
  SlimTensor(
      Storage&& storage,
      IntArrayRef sizes,
      IntArrayRef strides,
      c10::ScalarType dtype,
      int64_t storage_offset = 0)
      : storage_(std::move(storage)),
        storage_offset_(storage_offset),
        dtype_(dtype) {
    set_sizes_and_strides(sizes, strides);
  }

  /**
   * Default constructor - creates an undefined tensor.
   */
  SlimTensor()
      : storage_(Storage()),
        storage_offset_(0),
        numel_(0),
        dtype_(c10::ScalarType::Float),
        is_contiguous_(true) {
    sizes_and_strides_.set_sizes({0});
    sizes_and_strides_.set_strides({1});
  }

  // Default copy/move operations
  SlimTensor(const SlimTensor&) = default;
  SlimTensor& operator=(const SlimTensor&) = default;
  SlimTensor(SlimTensor&&) = default;
  SlimTensor& operator=(SlimTensor&&) = default;
  ~SlimTensor() = default;

  /**
   * Reset the tensor, releasing the storage reference.
   */
  void reset() {
    storage_.reset();
  }

  // =========================================================================
  // Property Accessors
  // =========================================================================

  /**
   * Get the underlying storage.
   */
  Storage storage() const {
    return storage_;
  }

  /**
   * Get the total number of bytes for this tensor's data.
   */
  size_t nbytes() const {
    return numel() * itemsize();
  }

  /**
   * Get the size of a single element in bytes.
   */
  size_t itemsize() const {
    return c10::elementSize(dtype_);
  }

  /**
   * Get the sizes of all dimensions.
   */
  IntArrayRef sizes() const {
    return sizes_and_strides_.sizes_arrayref();
  }

  /**
   * Get the size of a specific dimension.
   */
  int64_t size(int64_t dim) const {
    int64_t ndim = static_cast<int64_t>(this->dim());
    ET_CHECK_MSG(
        dim >= -ndim && dim < ndim,
        "Dimension out of range (expected to be in range of [%ld, %ld], but got %ld)",
        -ndim,
        ndim - 1,
        dim);
    if (dim < 0) {
      dim += ndim;
    }
    return sizes_and_strides_.size_at(static_cast<size_t>(dim));
  }

  /**
   * Get the strides of all dimensions.
   */
  IntArrayRef strides() const {
    return sizes_and_strides_.strides_arrayref();
  }

  /**
   * Get the stride of a specific dimension.
   */
  int64_t stride(int64_t dim) const {
    int64_t ndim = static_cast<int64_t>(this->dim());
    ET_CHECK_MSG(
        dim >= -ndim && dim < ndim,
        "Dimension out of range (expected to be in range of [%ld, %ld], but got %ld)",
        -ndim,
        ndim - 1,
        dim);
    if (dim < 0) {
      dim += ndim;
    }
    return sizes_and_strides_.stride_at(static_cast<size_t>(dim));
  }

  /**
   * Get the scalar type of tensor elements.
   */
  c10::ScalarType dtype() const {
    return dtype_;
  }

  /**
   * Get the device where the tensor data resides.
   */
  const c10::Device& device() const {
    return storage_->device();
  }

  /**
   * Get the device type.
   */
  c10::DeviceType device_type() const {
    return storage_->device().type();
  }

  /**
   * Get the device index.
   */
  c10::DeviceIndex device_index() const {
    return storage_->device().index();
  }

  /**
   * Get the storage offset in number of elements.
   */
  int64_t storage_offset() const {
    return storage_offset_;
  }

  /**
   * Get the total number of elements.
   */
  size_t numel() const {
    return numel_;
  }

  /**
   * Get the number of dimensions.
   */
  size_t dim() const {
    return sizes_and_strides_.size();
  }

  /**
   * Get a pointer to the tensor data, adjusted for storage offset.
   */
  void* data_ptr() const {
    return static_cast<char*>(storage_->data()) + storage_offset_ * itemsize();
  }

  /**
   * Check if the tensor is contiguous in memory (row-major order).
   */
  bool is_contiguous() const {
    return is_contiguous_;
  }

  /**
   * Check if the tensor has zero elements.
   */
  bool is_empty() const {
    return numel_ == 0;
  }

  /**
   * Check if the tensor is on CPU.
   */
  bool is_cpu() const {
    return device().is_cpu();
  }

  /**
   * Check if the tensor is on CUDA.
   */
  bool is_cuda() const {
    return device().is_cuda();
  }

  /**
   * Check if the tensor is defined (has valid storage).
   */
  bool defined() const {
    return storage_.get() != nullptr;
  }

  // =========================================================================
  // Setters
  // =========================================================================

  /**
   * Set the underlying storage.
   */
  void set_storage(Storage&& new_storage) {
    storage_ = std::move(new_storage);
  }

  /**
   * Set sizes, strides, and storage offset together.
   */
  void set_sizes_and_strides(
      IntArrayRef sizes,
      IntArrayRef strides,
      std::optional<int64_t> storage_offset = std::nullopt) {
    const size_t new_dim = sizes.size();
    ET_CHECK_MSG(
        new_dim == strides.size(),
        "dimensionality of sizes (%zu) must match dimensionality of strides (%zu)",
        new_dim,
        strides.size());

    std::vector<int64_t> new_sizes = toVec(sizes);
    std::vector<int64_t> new_strides = toVec(strides);

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
            overflowed |= ::c10::mul_overflows(
                new_strides[dim + 1],
                std::max<int64_t>(new_sizes[dim + 1], 1),
                &new_strides[dim]);
          }
        }
      }
    }
    ET_CHECK_MSG(!overflowed, "Stride calculation overflowed");

    sizes_and_strides_.set_sizes(makeArrayRef(new_sizes));
    sizes_and_strides_.set_strides(makeArrayRef(new_strides));
    if (storage_offset.has_value()) {
      storage_offset_ = *storage_offset;
    }

    refresh_numel();
    refresh_contiguous();
  }

  /**
   * Set sizes to a contiguous layout (computes strides automatically).
   */
  void set_sizes_contiguous(IntArrayRef sizes) {
    std::vector<int64_t> contig_strides = compute_contiguous_strides(sizes);
    set_sizes_and_strides(sizes, makeArrayRef(contig_strides));
  }

  /**
   * Returns a copy of this tensor.
   *
   * @return A new SlimTensor with same content.
   */
  SlimTensor clone() const {
    return _clone_impl(
        this->sizes(), this->strides(), this->dtype(), this->device());
  }

  /**
   * Returns a contiguous copy of this tensor.
   * If the tensor is already contiguous, returns a copy with independent
   * storage.
   *
   * @return A new contiguous SlimTensor.
   */
  SlimTensor clone_contiguous() const {
    std::vector<int64_t> contig_strides =
        compute_contiguous_strides(this->sizes());
    return _clone_impl(
        this->sizes(),
        makeArrayRef(contig_strides),
        this->dtype(),
        this->device());
  }

  // =========================================================================
  // View Operations
  // =========================================================================

  /**
   * Returns a view of the tensor with the specified sizes, strides, and
   * storage offset. The returned tensor shares the same underlying storage.
   *
   * @param sizes The sizes of the view.
   * @param strides The strides of the view.
   * @param storage_offset Offset into storage in number of elements.
   * @return A new SlimTensor that is a view of this tensor.
   */
  inline SlimTensor as_strided(
      IntArrayRef sizes,
      IntArrayRef strides,
      int64_t storage_offset) const;

  /**
   * Overload for initializer lists.
   */
  inline SlimTensor as_strided(
      std::initializer_list<int64_t> sizes,
      std::initializer_list<int64_t> strides,
      int64_t storage_offset) const {
    return as_strided(
        makeArrayRef(sizes), makeArrayRef(strides), storage_offset);
  }

  /**
   * Modifies this tensor in-place to have the specified sizes, strides, and
   * storage offset. The underlying storage remains unchanged.
   *
   * @param sizes The new sizes.
   * @param strides The new strides.
   * @param storage_offset New offset into storage in number of elements.
   * @return Reference to this tensor.
   */
  inline SlimTensor&
  as_strided_(IntArrayRef sizes, IntArrayRef strides, int64_t storage_offset);

  /**
   * Overload for initializer lists.
   */
  inline SlimTensor& as_strided_(
      std::initializer_list<int64_t> sizes,
      std::initializer_list<int64_t> strides,
      int64_t storage_offset) {
    return as_strided_(
        makeArrayRef(sizes), makeArrayRef(strides), storage_offset);
  }

  /**
   * Returns a new tensor with dimensions permuted according to dims.
   * The returned tensor shares the same underlying storage.
   *
   * @param dims The permutation of dimensions.
   * @return A new SlimTensor with permuted dimensions.
   */
  inline SlimTensor permute(IntArrayRef dims) const;

  /**
   * Overload for initializer lists.
   */
  inline SlimTensor permute(std::initializer_list<int64_t> dims) const {
    return permute(makeArrayRef(dims));
  }

  /**
   * Returns a tensor with the same data and number of elements as this tensor,
   * but with the specified shape. If possible, returns a view; otherwise
   * creates a contiguous copy.
   *
   * @param shape The target shape (may contain one -1 for inference).
   * @return A new SlimTensor with the specified shape.
   */
  inline SlimTensor reshape(IntArrayRef shape) const;

  /**
   * Overload for initializer lists.
   */
  inline SlimTensor reshape(std::initializer_list<int64_t> shape) const {
    return reshape(makeArrayRef(shape));
  }

  // =========================================================================
  // Copy Operation
  // =========================================================================

  /**
   * Copy data from another tensor to this tensor.
   *
   * Both tensors must have the same numel and dtype.
   * Currently only supports CPU-to-CPU copy (contiguous tensors only).
   *
   * @param other The source tensor to copy from
   * @return Reference to this tensor
   */
  SlimTensor& copy_(const SlimTensor& other) {
    ET_CHECK_MSG(
        this->numel() == other.numel(), "copy_: numel of tensors must match");
    ET_CHECK_MSG(this->dtype() == other.dtype(), "copy_: dtype must match");

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
    const size_t elem_size = c10::elementSize(dtype_);
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
#if defined(CUDA_AVAILABLE)
        DeviceTraits<c10::DeviceType::CUDA>::memcpy(
            dst_data + dst_offset * elem_size,
            src_data + src_offset * elem_size,
            elem_size,
            device(), // dst device
            other.device() // src device
        );
#else
        ET_CHECK_MSG(false, "Failed on copy_ cuda tensors: no CUDA support");
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

  /**
   * Extract the scalar value from a tensor with exactly 1 element.
   * Automatically handles CUDA tensors by copying data to CPU.
   *
   * @tparam T The type to extract (must match tensor dtype).
   * @return The scalar value.
   */
  template <typename T>
  T item() const {
    ET_CHECK_MSG(
        this->numel() == 1,
        "item() requires tensor to have exactly 1 element, got %zu",
        this->numel());

    T result;
    if (this->is_cpu()) {
      result = *static_cast<const T*>(this->data_ptr());
    } else {
#if defined(CUDA_AVAILABLE)
      DeviceTraits<c10::DeviceType::CUDA>::memcpy(
          &result, this->data_ptr(), sizeof(T), CPU_DEVICE, this->device());
#else
      ET_CHECK_MSG(false, "item(): CUDA tensor but CUDA support not available");
#endif
    }
    return result;
  }

 private:
  SlimTensor _clone_impl(
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      c10::ScalarType dtype,
      const c10::Device& device) const {
    Storage storage = new_storage(sizes, strides, dtype, device);
    SlimTensor result =
        SlimTensor(std::move(storage), sizes, strides, dtype, 0);
    result.copy_(*this);
    return result;
  }

  void refresh_numel() {
    numel_ = compute_numel(sizes_and_strides_.sizes_arrayref());
  }

  void refresh_contiguous() {
    is_contiguous_ = c10::_compute_contiguous<int64_t>(
        sizes_and_strides_.sizes_arrayref(),
        sizes_and_strides_.strides_arrayref(),
        static_cast<int64_t>(numel_));
  }

  Storage storage_;
  int64_t storage_offset_{0};
  c10::SizesAndStrides sizes_and_strides_;
  size_t numel_{1};
  c10::ScalarType dtype_;
  bool is_contiguous_{true};
};

} // namespace executorch::backends::aoti::slim

// Include view operations implementations (must be after SlimTensor class
// definition)
#include <executorch/backends/aoti/slim/core/slim_tensor_view_incl.h>
