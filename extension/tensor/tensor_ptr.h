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
 * Creates a TensorPtr that manages a new Tensor with the same properties
 * as the given Tensor, sharing the same data without owning it.
 *
 * @param tensor The Tensor whose properties are to be used to create a new
 * TensorPtr.
 * @return A new TensorPtr that manages a Tensor with the same properties as the
 * original.
 */
inline TensorPtr make_tensor_ptr(const exec_aten::Tensor& tensor) {
  return make_tensor_ptr(make_tensor_impl_ptr(
      tensor.scalar_type(),
      std::vector<exec_aten::SizesType>(
          tensor.sizes().begin(), tensor.sizes().end()),
      tensor.mutable_data_ptr(),
#ifndef USE_ATEN_LIB
      std::vector<exec_aten::DimOrderType>(
          tensor.dim_order().begin(), tensor.dim_order().end()),
      std::vector<exec_aten::StridesType>(
          tensor.strides().begin(), tensor.strides().end()),
      tensor.shape_dynamism()
#else // USE_ATEN_LIB
      {},
      std::vector<exec_aten::StridesType>(
          tensor.strides().begin(), tensor.strides().end())
#endif // USE_ATEN_LIB
          ));
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
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND,
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
 * provided as a vector. The scalar type is automatically deduced from the
 * vector's data type. If the specified `type` differs from the deduced type of
 * the vector's elements, and casting is allowed, the data will be cast to the
 * specified `type`. This allows for flexible creation of tensors with data
 * vectors of one type and a different scalar type.
 *
 * @tparam T The C++ type of the tensor elements, deduced from the vector.
 * @param sizes A vector specifying the size of each dimension.
 * @param data A vector containing the tensor's data.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of each dimension.
 * @param type The scalar type of the tensor elements. If it differs from the
 * deduced type, the data will be cast to this type if allowed.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorPtr that manages the newly created TensorImpl.
 */
template <typename T = float>
TensorPtr make_tensor_ptr(
    std::vector<exec_aten::SizesType> sizes,
    std::vector<T> data,
    std::vector<exec_aten::DimOrderType> dim_order = {},
    std::vector<exec_aten::StridesType> strides = {},
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return make_tensor_ptr(make_tensor_impl_ptr(
      std::move(sizes),
      std::move(data),
      std::move(dim_order),
      std::move(strides),
      dynamism));
}

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * This template overload is specialized for cases where the tensor data is
 * provided as a vector. The scalar type is automatically deduced from the
 * vector's data type. If the specified `type` differs from the deduced type of
 * the vector's elements, and casting is allowed, the data will be cast to the
 * specified `type`. This allows for flexible creation of tensors with data
 * vectors of one type and a different scalar type.
 *
 * @tparam T The C++ type of the tensor elements, deduced from the vector.
 * @param data A vector containing the tensor's data.
 * @param type The scalar type of the tensor elements. If it differs from the
 * deduced type, the data will be cast to this type if allowed.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorPtr that manages the newly created TensorImpl.
 */
template <typename T = float>
TensorPtr make_tensor_ptr(
    std::vector<T> data,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return make_tensor_ptr(make_tensor_impl_ptr(std::move(data), type, dynamism));
}

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * This template overload is specialized for cases where the tensor data is
 * provided as an initializer list. The scalar type is automatically deduced
 * from the initializer list's data type. If the specified `type` differs from
 * the deduced type of the initializer list's elements, and casting is allowed,
 * the data will be cast to the specified `type`. This allows for flexible
 * creation of tensors with data vectors of one type and a different scalar
 * type.
 *
 * @tparam T The C++ type of the tensor elements, deduced from the initializer
 * list.
 * @param sizes A vector specifying the size of each dimension.
 * @param list An initializer list containing the tensor's data.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of each dimension.
 * @param type The scalar type of the tensor elements. If it differs from the
 * deduced type, the data will be cast to this type if allowed.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorPtr that manages the newly created TensorImpl.
 */
template <
    typename T = float,
    exec_aten::ScalarType deduced_type = runtime::CppTypeToScalarType<T>::value>
inline TensorPtr make_tensor_ptr(
    std::vector<exec_aten::SizesType> sizes,
    std::initializer_list<T> list,
    std::vector<exec_aten::DimOrderType> dim_order = {},
    std::vector<exec_aten::StridesType> strides = {},
    exec_aten::ScalarType type = deduced_type,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return make_tensor_ptr(make_tensor_impl_ptr(
      std::move(sizes),
      std::move(list),
      std::move(dim_order),
      std::move(strides),
      type,
      dynamism));
}

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * This template overload allows creating a Tensor from an initializer list
 * of data. The scalar type is automatically deduced from the type of the
 * initializer list's elements. If the specified `type` differs from
 * the deduced type of the initializer list's elements, and casting is allowed,
 * the data will be cast to the specified `type`. This allows for flexible
 * creation of tensors with data vectors of one type and a different scalar
 * type.
 *
 * @tparam T The C++ type of the tensor elements, deduced from the initializer
 * list.
 * @param list An initializer list containing the tensor's data.
 * @param type The scalar type of the tensor elements. If it differs from the
 * deduced type, the data will be cast to this type if allowed.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorPtr that manages the newly created TensorImpl.
 */
template <typename T = float>
TensorPtr make_tensor_ptr(
    std::initializer_list<T> data,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return make_tensor_ptr(std::vector<T>(data), dynamism);
}

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * This overload accepts a raw memory buffer stored in a std::vector<uint8_t>
 * and a scalar type to interpret the data. The vector is managed, and the
 * memory's lifetime is tied to the TensorImpl.
 *
 * @param scalar_type The scalar type of the tensor elements.
 * @param sizes A vector specifying the size of each dimension.
 * @param data A vector containing the raw memory for the tensor's data.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of each dimension.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorPtr managing the newly created Tensor.
 */
inline TensorPtr make_tensor_ptr(
    exec_aten::ScalarType scalar_type,
    std::vector<exec_aten::SizesType> sizes,
    std::vector<uint8_t> data,
    std::vector<exec_aten::DimOrderType> dim_order = {},
    std::vector<exec_aten::StridesType> strides = {},
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return make_tensor_ptr(make_tensor_impl_ptr(
      scalar_type,
      std::move(sizes),
      std::move(data),
      std::move(dim_order),
      std::move(strides),
      dynamism));
}

/**
 * Creates a TensorPtr that manages a new Tensor with the same properties
 * as the given Tensor, but with a copy of the data owned by the returned
 * TensorPtr.
 *
 * @param tensor The Tensor to clone.
 * @return A new TensorPtr that manages a Tensor with the same properties as the
 * original but with copied data.
 */
inline TensorPtr clone_tensor_ptr(const exec_aten::Tensor& tensor) {
  return make_tensor_ptr(make_tensor_impl_ptr(
      tensor.scalar_type(),
      std::vector<exec_aten::SizesType>(
          tensor.sizes().begin(), tensor.sizes().end()),
      std::vector<uint8_t>(
          (uint8_t*)tensor.const_data_ptr(),
          (uint8_t*)tensor.const_data_ptr() + tensor.nbytes()),
#ifndef USE_ATEN_LIB
      std::vector<exec_aten::DimOrderType>(
          tensor.dim_order().begin(), tensor.dim_order().end()),
      std::vector<exec_aten::StridesType>(
          tensor.strides().begin(), tensor.strides().end()),
      tensor.shape_dynamism()
#else // USE_ATEN_LIB
      {},
      std::vector<exec_aten::StridesType>(
          tensor.strides().begin(), tensor.strides().end())
#endif // USE_ATEN_LIB
          ));
}

/**
 * Creates a new TensorPtr by cloning the given TensorPtr, copying the
 * underlying data.
 *
 * @param tensor The TensorPtr to clone.
 * @return A new TensorPtr that manages a Tensor with the same properties as the
 * original but with copied data.
 */
inline TensorPtr clone_tensor_ptr(const TensorPtr& tensor) {
  return clone_tensor_ptr(*tensor);
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
