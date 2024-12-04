/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>

namespace executorch {
namespace extension {
namespace internal {

/**
 * Base class template storing the underlying data with size and stride helpers.
 * Inherited by TensorAccessor<> which requires specialization on rank.
 */
template <typename T, ssize_t N>
class TensorAccessorBase {
 public:
  /// Returns the size of the underlying tensor at the given dimension.
  executorch::aten::SizesType size(ssize_t i) const {
    ET_CHECK_MSG(
        i < dim_ && i >= 0,
        "Dimension outside of [0, %zd], got %zd",
        dim_ - 1,
        i);
    return sizes_[i];
  }

  /// Returns the stride of the underlying tensor at the given dimension.
  executorch::aten::StridesType stride(ssize_t i) const {
    ET_CHECK_MSG(
        i < dim_ && i >= 0,
        "Dimension outside of [0, %zd], got %zd",
        dim_ - 1,
        i);
    return strides_[i];
  }

 protected:
  TensorAccessorBase(
      T* data,
      const executorch::aten::SizesType* sizes,
      const executorch::aten::StridesType* strides,
      ssize_t dim)
      : data_(data), sizes_(sizes), strides_(strides), dim_(dim) {}

  T* data_;
  const executorch::aten::SizesType* sizes_;
  const executorch::aten::StridesType* strides_;
  ssize_t dim_;
};

} // namespace internal

/**
 * TensorAccessor template with data type and rank as template parameters. No
 * public constructors, can only be created using make_tensor_accessor from a
 * given executorch::aten::Tensor. Use operator[] to index and obtain a lower
 * rank accessor or the underlying scalar value.
 */
template <typename T, ssize_t N>
class TensorAccessor : public internal::TensorAccessorBase<T, N> {
 public:
  /**
   * Index into the the outer most dimension.
   *
   * @param i Index.
   * @return If N > 1, a TensorAccessor with N-1 dimensions. If N == 1, a
   * reference to the underlying scalar. Refer to the TensorAccessor<T, 1>
   * specialization.
   */
  TensorAccessor<T, N - 1> operator[](ssize_t i) {
    return TensorAccessor<T, N - 1>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1,
        N - 1);
  }

  /**
   * Index into the the outer most dimension.
   *
   * @param i Index.
   * @return If N > 1, a constant TensorAccessor with N-1 dimensions. If N == 1,
   * a constant reference to the underlying scalar. Refer to the
   * TensorAccessor<T, 1> specialization.
   */
  const TensorAccessor<T, N - 1> operator[](ssize_t i) const {
    return TensorAccessor<T, N - 1>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1,
        N - 1);
  }

 private:
  TensorAccessor(
      T* data,
      const executorch::aten::SizesType* sizes,
      const executorch::aten::StridesType* strides,
      ssize_t dim)
      : internal::TensorAccessorBase<T, N>(data, sizes, strides, dim) {}

  template <typename T2, ssize_t N2>
  friend class TensorAccessor;

  template <typename T2, ssize_t N2>
  friend executorch::runtime::Result<TensorAccessor<T2, N2>>
  make_tensor_accessor(const executorch::aten::Tensor& t);
};

/**
 * TensorAccessor specialization for N == 1, where operator[] returns a
 * reference to the underlying scalar.
 */
template <typename T>
class TensorAccessor<T, 1> : public internal::TensorAccessorBase<T, 1> {
 public:
  /**
   * Index into the the outer most dimension.
   *
   * @param i Index.
   * @return Reference to the underlying scalar.
   */
  T& operator[](ssize_t i) {
    return this->data_[this->strides_[0] * i];
  }

  /**
   * Index into the the outer most dimension.
   *
   * @param i Index.
   * @return Constant reference to the underlying scalar.
   */
  const T& operator[](ssize_t i) const {
    return this->data_[this->strides_[0] * i];
  }

 private:
  TensorAccessor(
      T* data,
      const executorch::aten::SizesType* sizes,
      const executorch::aten::StridesType* strides,
      ssize_t dim)
      : internal::TensorAccessorBase<T, 1>(data, sizes, strides, dim) {}

  template <typename T2, ssize_t N2>
  friend class TensorAccessor;

  template <typename T2, ssize_t N2>
  friend executorch::runtime::Result<TensorAccessor<T2, N2>>
  make_tensor_accessor(const executorch::aten::Tensor& t);
};

/**
 * Creates a TensorAccessor<T, N> from the given tensor. The number of dimension
 * N and the data type T's size must match those of the input tensor. For
 * Executorch tensors, non-trivial dimension order is not supported.
 *
 * @param tensor Origin tensor. The TensorImpl inside must outlive the returned
 * TensorAccessor.
 * @return TensorAccessor of the input tensor.
 * @retval Error::InvalidArgument Mismatch on data type or number of dimensions.
 * @retval Error::NotSupported Input tensor has non-trivial dimension onrder.
 */
template <typename T, ssize_t N>
executorch::runtime::Result<TensorAccessor<T, N>> make_tensor_accessor(
    const executorch::aten::Tensor& tensor) {
  static_assert(
      N > 0,
      "TensorAccessor is used for indexing tensors, for scalar use *_data_ptr<T>()");

  if (N != tensor.dim()) {
    ET_LOG(
        Error,
        "Expecting %zd dimensions but tensor has %zd.",
        static_cast<ssize_t>(N),
        static_cast<ssize_t>(tensor.dim()));
    return executorch::runtime::Error::InvalidArgument;
  }

  if (sizeof(T) != tensor.element_size()) {
    ET_LOG(
        Error,
        "Size of data type template argument (%zd) not equal to tensor element size (%zd)",
        static_cast<ssize_t>(sizeof(T)),
        static_cast<ssize_t>(tensor.element_size()));
    return executorch::runtime::Error::InvalidArgument;
  }

#ifndef USE_ATEN_LIB
  auto dim_order = tensor.dim_order();
  for (ssize_t i = 0; i < dim_order.size(); i++) {
    if (dim_order[i] != i) {
      ET_LOG(Error, "Non-trival dim_order not supported.");
      return executorch::runtime::Error::NotSupported;
    }
  }
#endif

  T* ptr = nullptr;
  if constexpr (std::is_const_v<T>) {
    ptr = tensor.const_data_ptr<T>();
  } else {
    ptr = tensor.mutable_data_ptr<T>();
  }
  return TensorAccessor<T, N>(
      ptr, tensor.sizes().data(), tensor.strides().data(), N);
}

} // namespace extension
} // namespace executorch
