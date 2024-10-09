/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/portable_type/scalar_type.h>
#include <executorch/runtime/core/tensor_shape_dynamism.h>

// Forward declaration of a helper that provides access to internal resizing
// methods of TensorImpl. Real definition is in
// executorch/runtime/core/exec_aten/tensor_util.h.
namespace executorch {
namespace runtime {
namespace internal {
class TensorResizerFriend;
} // namespace internal
} // namespace runtime
} // namespace executorch

namespace executorch {
namespace runtime {
namespace etensor {

/**
 * Manages the storage behind an ETensor (torch::executor::Tensor).
 *
 * Note that instances of this class do not own the arrays given to it
 * (sizes/strides/data), which means that the caller must guarantee that they
 * live longer than a given instance of this class.
 *
 * Note on types:
 *
 * Code that uses ETensor should also be able to build against at::Tensor. So,
 * although the overlapping APIs don't need to be exactly the same, their types
 * should be semantically similar.
 *
 * Many of the methods in at::Tensor use int64_t for parameter and return types.
 * This can be a waste when building for 32-bit environments. So, TensorImpl and
 * ETensor use ssize_t instead: like int64_t it is signed, but it will match the
 * native word size of the target architecture. This will avoid unnecessarily
 * expensive uses of 64-bit integers on 32-bit machines.
 *
 * But, since the types are not identical, code that uses ETensor needs to be
 * generic about the local types it uses when working with these methods. In
 * most cases, `auto` will do the trick. In the worst case, code can be guarded
 * with `#ifdef USE_ATEN_LIB`.
 */
class TensorImpl {
 public:
  /**
   * The type used for elements of `sizes()`.
   *
   * This must match the size/signedness of the type used for `Tensor.sizes` in
   * //executorch/schema/program.fbs.
   *
   * Note that at::TensorImpl uses `int64_t` for this type. ExecuTorch uses
   * `int32_t` to save memory, since no single size value will ever be larger
   * than 2 billion.
   */
  using SizesType = int32_t;

  /**
   * The type used for elements of `dim_order()`.
   *
   * This must match the size/signedness of the type used for `Tensor.dim_order`
   * in //executorch/schema/program.fbs.
   */
  using DimOrderType = uint8_t;

  /**
   * The type used for elements of `strides()`.
   *
   * This must match the size/signedness of the type used for `Tensor.strides`
   * in //executorch/schema/program.fbs.
   *
   * Note that at::TensorImpl uses `int64_t` for this type. ExecuTorch uses
   * `int32_t` to save memory, since no single stride value will ever be larger
   * than 2 billion.
   */
  using StridesType = int32_t;

  TensorImpl() = delete;

  /**
   * @param type The type of the data (int, float, bool).
   * @param dim Number of dimensions, and the length of the `sizes` array.
   * @param sizes Sizes of the tensor at each dimension. Must contain `dim`
   *     entries.
   * @param data Pointer to the data, whose size is determined by `type`,
   *     `dim`, and `sizes`. The tensor will not own this memory.
   * @param dim_order Order in which dimensions are laid out in memory.
   * @param strides Strides of the tensor at each dimension. Must contain `dim`
   *     entries.
   * @param dynamism The mutability of the shape of the tensor.
   */
  TensorImpl(
      ScalarType type,
      ssize_t dim,
      SizesType* sizes,
      void* data = nullptr,
      DimOrderType* dim_order = nullptr,
      StridesType* strides = nullptr,
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC);

  /**
   * Returns the size of the tensor in bytes.
   *
   * NOTE: This returns the size of the data used by the tensor's current shape,
   * not the capacity of the underlying buffer.
   */
  size_t nbytes() const;

  /**
   * Returns the size of the tensor at the given dimension.
   *
   * NOTE: size() intentionally does not return SizeType even though it
   * returns an element of an array of SizeType. This is to help make calls of
   * this method more compatible with at::Tensor, and more consistent with the
   * rest of the methods on this class and in ETensor.
   */
  ssize_t size(ssize_t dim) const {
    ET_CHECK_MSG(
        dim < dim_ && dim >= 0,
        "Dimension out of range (expected to be in range of [0, %zd], but got %zd",
        dim_ - 1,
        dim);
    return sizes_[dim];
  }

  /// Returns the tensor's number of dimensions.
  ssize_t dim() const {
    return dim_;
  }

  /// Returns the number of elements in the tensor.
  ssize_t numel() const {
    return numel_;
  }

  /// Returns the type of the elements in the tensor (int32, float, bool, etc).
  ScalarType scalar_type() const {
    return type_;
  }

  inline ScalarType dtype() const {
    return scalar_type();
  }

  /// Returns the size in bytes of one element of the tensor.
  ssize_t element_size() const;

  /// Returns the sizes of the tensor at each dimension.
  const ArrayRef<SizesType> sizes() const {
    return ArrayRef<SizesType>{sizes_, static_cast<size_t>(dim_)};
  }

  /// Returns the order the dimensions are laid out in memory.
  const ArrayRef<DimOrderType> dim_order() const {
    return ArrayRef<DimOrderType>{dim_order_, static_cast<size_t>(dim_)};
  }

  /// Returns the strides of the tensor at each dimension.
  const ArrayRef<StridesType> strides() const {
    return ArrayRef<StridesType>{strides_, static_cast<size_t>(dim_)};
  }

  /// Returns the mutability of the shape of the tensor.
  TensorShapeDynamism shape_dynamism() const {
    return shape_dynamism_;
  }

  /// Returns a pointer of type T to the constant underlying data blob.
  template <typename T>
  inline const T* data() const {
    return static_cast<const T*>(data());
  }

  /// Returns a pointer to the constant underlying data blob.
  const void* data() const {
    return data_;
  }

  /// Returns a pointer of type T to the mutable underlying data blob.
  template <typename T>
  inline T* mutable_data() const {
    return static_cast<T*>(mutable_data());
  }

  /// Returns a pointer to the mutable underlying data blob.
  void* mutable_data() const {
    return data_;
  }

  /// Sets the underlying data blob to the passed in pointer.
  void set_data(void* ptr) {
    data_ = ptr;
  }

  /*
   * DEPRECATED: Use torch::executor::resize_tensor() or
   * torch::executor::resize_tensor_impl().
   */
  ET_DEPRECATED
  void set_sizes_contiguous(ArrayRef<SizesType> new_sizes) {
    Error err = internal_resize_contiguous(new_sizes);
    ET_CHECK_MSG(
        err == Error::Ok, "Could not resize Tensor; see logs for details");
  }

 private:
  // For access to internal_resize_contiguous().
  friend class ::executorch::runtime::internal::TensorResizerFriend;

  /**
   * Set the sizes and strides of a tensor assuming contiguous strides.
   * Requires that `new_sizes.size() == this.dim()`.
   *
   * Callers must use torch::executor::resize_tensor() or
   * torch::executor::resize_tensor_impl() instead, defined in TensorUtil.h.
   *
   * Same semantics as at::TensorImpl::set_sizes_contiguous(), but returns an
   * error instead of panicking on failure. This is not part of the at::Tensor
   * API, and can only be used in lean mode.
   */
  ET_NODISCARD Error internal_resize_contiguous(ArrayRef<SizesType> new_sizes);

 private:
  // Keep fields arranged to avoid unnecessary alignment holes.

  /// List of sizes of each dimension in the tensor.
  SizesType* sizes_;

  /// List of the order that dimensions are laid out in memory.
  DimOrderType* dim_order_;

  // TODO(T148356881): Get rid of strides from ETensor
  StridesType* strides_;

  /// Pointer to underlying data blob. NOTE: Can be null.
  void* data_;

  /// Tensor's number of dimensions.
  const ssize_t dim_;

  /// Number of elements in the tensor.
  ssize_t numel_;

  /// Maximum number of elements in the bounded tensor. Used when resizing up
  /// and down.
  size_t numel_bound_;

  /// Scalar type (int, float, bool, etc) of the tensor data.
  const ScalarType type_;

  /// Specifies the mutability of the shape of the tensor.
  const TensorShapeDynamism shape_dynamism_;
};

/**
 * Compute the number of elements based on the sizes of a tensor.
 */
ssize_t compute_numel(
    const ::executorch::runtime::etensor::TensorImpl::SizesType* sizes,
    ssize_t dim);

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::compute_numel;
using ::executorch::runtime::etensor::TensorImpl;
} // namespace executor
} // namespace torch
