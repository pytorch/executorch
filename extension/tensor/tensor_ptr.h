/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/tensor/tensor_impl_ptr.h>
#include <executorch/runtime/core/error.h>

namespace executorch {
namespace extension {

#ifndef USE_ATEN_LIB
namespace internal {
/**
 * Custom deleter for TensorPtr that ensures the associated TensorImplPtr
 * is properly managed.
 *
 * Since Tensor does not own its TensorImpl, this deleter is responsible for
 * managing the lifecycle of the TensorImplPtr, ensuring that the dynamic
 * metadata (sizes, dim_order, strides) is properly released when the Tensor is
 * destroyed.
 */
struct TensorPtrDeleter final {
  TensorImplPtr tensor_impl;

  void operator()(exec_aten::Tensor* pointer) {
    // Release all resources immediately since the data held by the
    // TensorPtrDeleter is tied to the managed object, not the smart pointer
    // itself. We need to free this memory when the object is destroyed, not
    // when the smart pointer (and deleter) are eventually destroyed or reset.
    tensor_impl.reset();
    delete pointer;
  }
};
} // namespace internal

/**
 * A smart pointer type for managing the lifecycle of a Tensor.
 *
 * TensorPtr uses a unique pointer to enforce that each Tensor object has
 * distinct ownership. This abstraction serves as a more convenient and safer
 * replacement for the standard Tensor, which does not manage its
 * metadata by design. Using TensorPtr simplifies memory management and ensures
 * that the underlying TensorImpl is safely shared among tensors when needed.
 */
using TensorPtr =
    std::unique_ptr<exec_aten::Tensor, internal::TensorPtrDeleter>;
#else
/**
 * A smart pointer type for managing the lifecycle of a Tensor.
 *
 * When using ATen, this is a standard unique_ptr for exec_aten::Tensor.
 * In ATen, the Tensor class owns its TensorImpl and associated metadata,
 * so no custom deleter is required.
 */
using TensorPtr = std::unique_ptr<exec_aten::Tensor>;
#endif // USE_ATEN_LIB

/**
 * Creates a new TensorPtr that manages a newly created Tensor with the given
 * TensorImplPtr.
 *
 * This function wraps the provided TensorImplPtr in a TensorPtr, ensuring that
 * the Tensor object's lifecycle is properly managed. The TensorPtr will
 * uniquely own the Tensor object, while the underlying TensorImplPtr may be
 * shared with other Tensors.
 *
 * @param tensor_impl A TensorImplPtr to the TensorImpl to be managed.
 * @return A TensorPtr that manages the newly created Tensor.
 */
inline TensorPtr make_tensor_ptr(TensorImplPtr tensor_impl) {
#ifndef USE_ATEN_LIB
  auto tensor = std::make_unique<exec_aten::Tensor>(tensor_impl.get());
  return TensorPtr(
      tensor.release(), internal::TensorPtrDeleter{std::move(tensor_impl)});
#else
  return std::make_unique<exec_aten::Tensor>(std::move(tensor_impl));
#endif // USE_ATEN_LIB
}

/**
 * Creates a new TensorPtr that shares the same TensorImplPtr as an existing
 * TensorPtr.
 *
 * This function creates a new TensorPtr that shares the
 * underlying TensorImpl with the provided TensorPtr, ensuring that the
 * underlying data and metadata are not duplicated but safely shared between the
 * tensor objects.
 *
 * @param tensor A TensorPtr to the existing Tensor from which to create a copy.
 * @return A new TensorPtr that shares the underlying TensorImplPtr with the
 * original.
 */
inline TensorPtr make_tensor_ptr(const TensorPtr& tensor) {
#ifndef USE_ATEN_LIB
  return make_tensor_ptr(tensor.get_deleter().tensor_impl);
#else
  return make_tensor_ptr(tensor->getIntrusivePtr());
#endif // USE_ATEN_LIB
}

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * @param type The scalar type of the tensor elements.
 * @param sizes A vector specifying the size of each dimension.
 * @param data A pointer to the data buffer.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of the tensor.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @param deleter A custom deleter function for managing the lifetime of the
 * data buffer. If provided, this deleter will be called when the managed Tensor
 * object is destroyed.
 * @return A TensorPtr that manages the newly created Tensor.
 */
inline TensorPtr make_tensor_ptr(
    const exec_aten::ScalarType type,
    std::vector<exec_aten::SizesType> sizes,
    void* data,
    std::vector<exec_aten::DimOrderType> dim_order = {},
    std::vector<exec_aten::StridesType> strides = {},
    const exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::STATIC,
    std::function<void(void*)> deleter = nullptr) {
  return make_tensor_ptr(make_tensor_impl_ptr(
      type,
      std::move(sizes),
      data,
      std::move(dim_order),
      std::move(strides),
      dynamism,
      std::move(deleter)));
}

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * This template overload is specialized for cases where the tensor data is
 * provided as a vector of a specific scalar type, rather than a raw pointer.
 * The deleter ensures that the data vector is properly managed and its
 * lifetime is tied to the TensorImpl.
 *
 * @tparam T The scalar type of the tensor elements.
 * @param sizes A vector specifying the size of each dimension.
 * @param data A vector containing the tensor's data.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of each dimension.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorImplPtr managing the newly created TensorImpl.
 */
template <exec_aten::ScalarType T = exec_aten::ScalarType::Float>
TensorPtr make_tensor_ptr(
    std::vector<exec_aten::SizesType> sizes,
    std::vector<typename runtime::ScalarTypeToCppType<T>::type> data,
    std::vector<exec_aten::DimOrderType> dim_order = {},
    std::vector<exec_aten::StridesType> strides = {},
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::STATIC) {
  return make_tensor_ptr(make_tensor_impl_ptr<T>(
      std::move(sizes),
      std::move(data),
      std::move(dim_order),
      std::move(strides),
      dynamism));
}

/**
 * Resizes the Tensor managed by the given TensorPtr to the new sizes provided.
 *
 * @param tensor A TensorPtr managing the Tensor to resize.
 * @param sizes A vector representing the new sizes for each dimension.
 * @return Error::Ok on success, or an appropriate error code otherwise.
 */
ET_NODISCARD
runtime::Error resize_tensor_ptr(
    TensorPtr& tensor,
    const std::vector<exec_aten::SizesType>& sizes);

} // namespace extension
} // namespace executorch
