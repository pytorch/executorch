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
#include <utility>

#include <executorch/backends/aoti/slim/c10/core/Contiguity.h>
#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/c10/core/SizesAndStrides.h>
#include <executorch/backends/aoti/slim/core/Storage.h>
#include <executorch/backends/aoti/slim/util/ArrayRefUtil.h>
#include <executorch/backends/aoti/slim/util/SizeUtil.h>
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
   * Set sizes and strides together.
   */
  void set_sizes_and_strides(IntArrayRef sizes, IntArrayRef strides) {
    ET_CHECK_MSG(
        sizes.size() == strides.size(),
        "sizes (%zu) and strides (%zu) must have the same length",
        sizes.size(),
        strides.size());

    sizes_and_strides_.set_sizes(sizes);
    sizes_and_strides_.set_strides(strides);

    refresh_numel();
    refresh_contiguous();
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
        this->numel() == other.numel(),
        "copy_: numel mismatch (dst=%zu, src=%zu)",
        this->numel(),
        other.numel());
    ET_CHECK_MSG(this->dtype() == other.dtype(), "copy_: dtype mismatch");

    if (this->numel() == 0) {
      return *this;
    }

    // Current we only support CPU-only tensors
    // TODO(gasoonjia): support other device types.
    ET_CHECK_MSG(
        this->is_cpu() && other.is_cpu(), "copy_: only CPU tensors supported");

    if (this->is_contiguous() && other.is_contiguous()) {
      // Fast path: both tensors are contiguous, use memcpy
      std::memcpy(this->data_ptr(), other.data_ptr(), other.nbytes());
    } else {
      // Slow path: element-wise copy for non-contiguous tensors
      copy_strided_(other);
    }

    return *this;
  }

 private:
  /**
   * Element-wise copy for non-contiguous tensors.
   */
  void copy_strided_(const SlimTensor& other) {
    const size_t elem_size = c10::elementSize(dtype_);
    char* dst_data = static_cast<char*>(this->data_ptr());
    const char* src_data = static_cast<const char*>(other.data_ptr());

    std::vector<int64_t> counter(this->dim(), 0);
    for (size_t i = 0; i < this->numel(); i++) {
      // Compute source offset
      int64_t src_offset = 0;
      for (size_t d = 0; d < other.dim(); d++) {
        src_offset += counter[d] * other.stride(static_cast<int64_t>(d));
      }

      // Compute destination offset
      int64_t dst_offset = 0;
      for (size_t d = 0; d < this->dim(); d++) {
        dst_offset += counter[d] * this->stride(static_cast<int64_t>(d));
      }

      // Copy single element
      std::memcpy(
          dst_data + dst_offset * static_cast<int64_t>(elem_size),
          src_data + src_offset * static_cast<int64_t>(elem_size),
          elem_size);

      // Increment multi-dimensional counter
      for (int64_t d = static_cast<int64_t>(this->dim()) - 1; d >= 0; --d) {
        counter[d]++;
        if (counter[d] < this->size(d)) {
          break;
        }
        counter[d] = 0;
      }
    }
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
