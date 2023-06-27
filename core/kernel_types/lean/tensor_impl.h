// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <sys/types.h> // TODO(T126923429): Include size_t, ssize_t

#include <executorch/core/ArrayRef.h>
#include <executorch/core/Error.h>
#include <executorch/core/kernel_types/TensorShapeDynamism.h>
#include <executorch/core/kernel_types/lean/scalar_type.h>

namespace torch {
namespace executor {

// Forward declaration of a helper that provides access to internal resizing
// methods of TensorImpl. Real definition is in
// executorch/core/kernel_types/util/TensorUtil.h.
namespace internal {
class TensorResizerFriend;
} // namespace internal

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
   * //executorch/schema/schema.fbs.
   *
   * Note that at::TensorImpl uses `int64_t` for this type. Executorch uses
   * `int32_t` to save memory, since no single size value will ever be larger
   * than 2 billion.
   */
  using SizesType = int32_t;

  /**
   * The type used for elements of `dim_order()`.
   *
   * This must match the size/signedness of the type used for `Tensor.dim_order`
   * in //executorch/schema/schema.fbs.
   */
  using DimOrderType = uint8_t;

  /**
   * The type used for elements of `strides()`.
   *
   * This must match the size/signedness of the type used for `Tensor.strides`
   * in //executorch/schema/schema.fbs.
   *
   * Note that at::TensorImpl uses `int64_t` for this type. Executorch uses
   * `int32_t` to save memory, since no single stride value will ever be larger
   * than 2 billion.
   */
  using StridesType = int32_t;

  TensorImpl() = delete;

  /**
   * @param type: What scalartype data contains (int, float, bool)
   *
   * @param dim: Rank of the tensor
   *
   * @param sizes: Sizes of the tensor at each dimension
   *
   * @param data: pointer to underlying data blob
   *
   * @param dim_order: Order in which dimensions are laid out in memory
   *
   * @param strides: Strides of the tensor at each dimension
   *
   * @param storage_offset: Offset into data that this tensors data blob starts
   * at. Typically used in views
   *
   * @param dynamism: WILL BE REMOVED DONT RELY ON IT SEE TYPE DECLARATION FOR
   * MORE INFO. Enum describing if this tensor can be resized, and if so how.
   * Ex: StaticShape, data cannot be resized and sizes and strides may point to
   * constant memory.
   */
  TensorImpl(
      ScalarType type,
      ssize_t dim,
      SizesType* sizes,
      void* data = nullptr,
      DimOrderType* dim_order = nullptr,
      StridesType* strides = nullptr,
      ssize_t storage_offset = 0,
      // THIS FIELD (dynamism) WILL BE REMOVED DONT RELY ON IT
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
  ssize_t size(ssize_t dim) const;

  /// Returns the tensor's number of dimensions.
  ssize_t dim() const;

  /// Returns the number of elements in the tensor.
  ssize_t numel() const;

  /// Returns the type of the elements in the tensor (int32, float, bool, etc).
  ScalarType scalar_type() const;

  /// Returns the size in bytes of one element of the tensor.
  ssize_t element_size() const;

  /// Returns the offset from `data_` to the beginning of the actual tensor
  /// data, in units of `type_` elements. E.g., if this is an int32 tensor, an
  /// offset of 1 would be a 4-byte offset from `data_` since `sizeof(int32) ==
  /// 4`.
  ssize_t storage_offset() const;

  /// Returns the sizes of the tensor at each dimension.
  const ArrayRef<SizesType> sizes() const;

  /// Returns the order the dimensions are laid out in memory.
  const ArrayRef<DimOrderType> dim_order() const;

  /// Returns the strides of the tensor at each dimension.
  const ArrayRef<StridesType> strides() const;

  /// Returns a pointer of type T to the constant underlying data blob.
  template <typename T>
  inline const T* data() const {
    return static_cast<const T*>(data());
  }

  /// Returns a pointer to the constant underlying data blob.
  const void* data() const;

  /// Returns a pointer of type T to the mutable underlying data blob.
  template <typename T>
  inline T* mutable_data() const {
    return static_cast<T*>(mutable_data());
  }

  /// Returns a pointer to the mutable underlying data blob.
  void* mutable_data() const;

  /// Sets the underlying data blob to the passed in pointer.
  void set_data(void* ptr);

  /*
   * DEPRECATED: Use torch::executor::resize_tensor() or
   * torch::executor::resize_tensor_impl().
   */
  __ET_DEPRECATED
  void set_sizes_contiguous(ArrayRef<SizesType> new_sizes) {
    Error err = internal_resize_contiguous(new_sizes);
    ET_CHECK_MSG(
        err == Error::Ok, "Could not resize Tensor; see logs for details");
  }

 private:
  // For access to internal_resize_contiguous().
  friend class internal::TensorResizerFriend;

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
  __ET_NODISCARD Error
  internal_resize_contiguous(ArrayRef<SizesType> new_sizes);

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

  /// Offset in elements into data that this tensor's data blob starts at.
  const ssize_t storage_offset_;

  /// Number of elements in the tensor.
  ssize_t numel_;

  /// Underlying capacity of data_ in bytes. Used when resizing up and down.
  size_t capacity_;

  /// Scalar type (int, float, bool, etc) of the tensor data.
  const ScalarType type_;

  /// Specifies the mutability of the shape of the tensor.
  const TensorShapeDynamism shape_dynamism_;
};

} // namespace executor
} // namespace torch
