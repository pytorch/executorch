/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/platform/compiler.h>

#include <executorch/runtime/core/portable_type/tensor_impl.h>

namespace executorch {
namespace runtime {
namespace etensor {

/**
 * A minimal Tensor type whose API is a source compatible subset of at::Tensor.
 *
 * NOTE: Instances of this class do not own the TensorImpl given to it,
 * which means that the caller must guarantee that the TensorImpl lives longer
 * than any Tensor instances that point to it.
 *
 * See the documention on TensorImpl for details about the return/parameter
 * types used here and how they relate to at::Tensor.
 */
class Tensor {
 public:
  /// The type used for elements of `sizes()`.
  using SizesType = TensorImpl::SizesType;
  /// The type used for elements of `dim_order()`.
  using DimOrderType = TensorImpl::DimOrderType;
  /// The type used for elements of `strides()`.
  using StridesType = TensorImpl::StridesType;

  Tensor() = delete;
  explicit Tensor(TensorImpl* impl) : impl_(impl) {}

  /**
   * Returns a pointer to the underlying TensorImpl.
   *
   * NOTE: Clients should be wary of operating on the TensorImpl
   * directly instead of the Tensor. It is easy to break things.
   */
  TensorImpl* unsafeGetTensorImpl() const {
    // TODO(T154114015): See if we can make this api private with friends.
    return impl_;
  }

  /**
   * Returns the size of the tensor in bytes.
   *
   * NOTE: Only the alive space is returned not the total capacity of the
   * underlying data blob.
   */
  size_t nbytes() const {
    return impl_->nbytes();
  }

  /**
   * Returns the size of the tensor at the given dimension.
   *
   * NOTE: that size() intentionally does not return SizeType even though it
   * returns an element of an array of SizeType. This is to help make calls of
   * this method more compatible with at::Tensor, and more consistent with the
   * rest of the methods on this class and in ETensor.
   */
  ssize_t size(ssize_t dim) const {
    return impl_->size(dim);
  }

  /// Returns the tensor's number of dimensions.
  ssize_t dim() const {
    return impl_->dim();
  }

  /// Returns the number of elements in the tensor.
  ssize_t numel() const {
    return impl_->numel();
  }

  /// Returns the type of the elements in the tensor (int32, float, bool, etc).
  ScalarType scalar_type() const {
    return impl_->scalar_type();
  }

  inline ScalarType dtype() const {
    return scalar_type();
  }

  /// Returns the size in bytes of one element of the tensor.
  ssize_t element_size() const {
    return impl_->element_size();
  }

  /// Returns the sizes of the tensor at each dimension.
  const ArrayRef<SizesType> sizes() const {
    return impl_->sizes();
  }

  /// Returns the order the dimensions are laid out in memory.
  const ArrayRef<DimOrderType> dim_order() const {
    return impl_->dim_order();
  }

  /// Returns the strides of the tensor at each dimension.
  const ArrayRef<StridesType> strides() const {
    return impl_->strides();
  }

  /// Returns the mutability of the shape of the tensor.
  TensorShapeDynamism shape_dynamism() const {
    return impl_->shape_dynamism();
  }

  /// Returns a pointer of type T to the constant underlying data blob.
  template <typename T>
  inline const T* const_data_ptr() const {
    return impl_->data<T>();
  }

  /// Returns a pointer to the constant underlying data blob.
  inline const void* const_data_ptr() const {
    return impl_->data();
  }

  /// Returns a pointer of type T to the mutable underlying data blob.
  template <typename T>
  inline T* mutable_data_ptr() const {
    return impl_->mutable_data<T>();
  }

  /// Returns a pointer to the mutable underlying data blob.
  inline void* mutable_data_ptr() const {
    return impl_->mutable_data();
  }

  /// DEPRECATED: Use const_data_ptr or mutable_data_ptr instead.
  template <typename T>
  ET_DEPRECATED inline T* data_ptr() const {
    return impl_->mutable_data<T>();
  }

  /// DEPRECATED: Use const_data_ptr or mutable_data_ptr instead.
  ET_DEPRECATED inline void* data_ptr() const {
    return impl_->mutable_data();
  }

  /**
   * DEPRECATED: Changes the data_ptr the tensor aliases. Does not free the
   * previously pointed to data, does not assume ownership semantics of the new
   * ptr. This api does not exist in at::Tensor so kernel developers should
   * avoid it.
   */
  ET_DEPRECATED void set_data(void* ptr) const {
    impl_->set_data(ptr);
  }

 private:
  TensorImpl* impl_ = nullptr;
};

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::Tensor;
} // namespace executor
} // namespace torch
