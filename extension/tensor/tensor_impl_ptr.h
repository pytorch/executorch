/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {

#ifndef USE_ATEN_LIB
/**
 * A smart pointer for managing the lifecycle of a TensorImpl.
 *
 * TensorImplPtr uses a shared pointer since multiple Tensor objects may
 * share the same underlying data and metadata. This shared ownership ensures
 * that the TensorImpl is destroyed only when all references to it are gone,
 * providing a safe and efficient way to manage shared tensor implementations.
 * It serves as a safer, more convenient alternative to the original TensorImpl,
 * which does not manage its metadata by design.
 */
using TensorImplPtr = std::shared_ptr<executorch::aten::TensorImpl>;
#else
/**
 * A smart pointer type for managing the lifecycle of a TensorImpl.
 *
 * TensorImplPtr uses an intrusive pointer when working with ATen, ensuring
 * efficient reference counting and shared ownership of the underlying data and
 * metadata.
 */
using TensorImplPtr =
    c10::intrusive_ptr<executorch::aten::TensorImpl, at::UndefinedTensorImpl>;
#endif // USE_ATEN_LIB

/**
 * Creates a TensorImplPtr that manages a newly created TensorImpl with the
 * specified properties.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param data A pointer to the data buffer.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @param deleter A custom deleter function for managing the lifetime of the
 * data buffer. If provided, this deleter is called when the managed TensorImpl
 * is destroyed.
 * @return A TensorImplPtr managing the newly created TensorImpl.
 */
TensorImplPtr make_tensor_impl_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    void* data,
    std::vector<executorch::aten::DimOrderType> dim_order,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
    std::function<void(void*)> deleter = nullptr);

/**
 * Creates a TensorImplPtr that manages a newly created TensorImpl with the
 * specified properties.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param data A pointer to the data buffer.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @param deleter A custom deleter function for managing the lifetime of the
 * data buffer. If provided, this deleter is called when the managed TensorImpl
 * is destroyed.
 * @return A TensorImplPtr managing the newly created TensorImpl.
 */
inline TensorImplPtr make_tensor_impl_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    void* data,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
    std::function<void(void*)> deleter = nullptr) {
  return make_tensor_impl_ptr(
      std::move(sizes), data, {}, {}, type, dynamism, std::move(deleter));
}

/**
 * Creates a TensorImplPtr that manages a newly created TensorImpl with the
 * specified properties.
 *
 * This template overload is specialized for cases where tensor data is provided
 * as a vector. If the specified `type` differs from the deduced type of the
 * vector's elements, and casting is allowed, the data will be cast to the
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
 * @return A TensorImplPtr that manages the newly created TensorImpl.
 */
template <
    typename T = float,
    executorch::aten::ScalarType deduced_type =
        runtime::CppTypeToScalarType<T>::value>
TensorImplPtr make_tensor_impl_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<T> data,
    std::vector<executorch::aten::DimOrderType> dim_order = {},
    std::vector<executorch::aten::StridesType> strides = {},
    executorch::aten::ScalarType type = deduced_type,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  if (type != deduced_type) {
    ET_CHECK_MSG(
        runtime::canCast(deduced_type, type),
        "Cannot cast deduced type to specified type.");
    std::vector<uint8_t> casted_data(data.size() * runtime::elementSize(type));
    ET_SWITCH_REALHBBF16_TYPES(
        type, nullptr, "make_tensor_impl_ptr", CTYPE, [&] {
          std::transform(
              data.begin(),
              data.end(),
              reinterpret_cast<CTYPE*>(casted_data.data()),
              [](const T& val) { return static_cast<CTYPE>(val); });
        });
    const auto raw_data_ptr = casted_data.data();
    auto data_ptr =
        std::make_shared<std::vector<uint8_t>>(std::move(casted_data));
    return make_tensor_impl_ptr(
        std::move(sizes),
        raw_data_ptr,
        std::move(dim_order),
        std::move(strides),
        type,
        dynamism,
        [data_ptr = std::move(data_ptr)](void*) {});
  }
  const auto raw_data_ptr = data.data();
  auto data_ptr = std::make_shared<std::vector<T>>(std::move(data));
  return make_tensor_impl_ptr(
      std::move(sizes),
      raw_data_ptr,
      std::move(dim_order),
      std::move(strides),
      type,
      dynamism,
      [data_ptr = std::move(data_ptr)](void*) {});
}

/**
 * Creates a TensorImplPtr that manages a newly created TensorImpl with the
 * specified properties.
 *
 * This template overload is specialized for cases where tensor data is provided
 * as a vector. If the specified `type` differs from the deduced type of the
 * vector's elements, and casting is allowed, the data will be cast to the
 * specified `type`. This allows for flexible creation of tensors with data
 * vectors of one type and a different scalar type.
 *
 * @tparam T The C++ type of the tensor elements, deduced from the vector.
 * @param data A vector containing the tensor's data.
 * @param type The scalar type of the tensor elements. If it differs from the
 * deduced type, the data will be cast to this type if allowed.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorImplPtr that manages the newly created TensorImpl.
 */
template <
    typename T = float,
    executorch::aten::ScalarType deduced_type =
        runtime::CppTypeToScalarType<T>::value>
inline TensorImplPtr make_tensor_impl_ptr(
    std::vector<T> data,
    executorch::aten::ScalarType type = deduced_type,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  std::vector<executorch::aten::SizesType> sizes{
      executorch::aten::SizesType(data.size())};
  return make_tensor_impl_ptr(
      std::move(sizes), std::move(data), {0}, {1}, type, dynamism);
}

/**
 * Creates a TensorImplPtr that manages a newly created TensorImpl with the
 * specified properties.
 *
 * This template overload is specialized for cases where tensor data is provided
 * as an initializer list. If the specified `type` differs from the deduced type
 * of the initializer list's elements, and casting is allowed, the data will be
 * cast to the specified `type`. This allows for flexible creation of tensors
 * with data initializer list of one type and a different scalar type.
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
 * @return A TensorImplPtr that manages the newly created TensorImpl.
 */
template <
    typename T = float,
    executorch::aten::ScalarType deduced_type =
        runtime::CppTypeToScalarType<T>::value>
inline TensorImplPtr make_tensor_impl_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    std::initializer_list<T> list,
    std::vector<executorch::aten::DimOrderType> dim_order = {},
    std::vector<executorch::aten::StridesType> strides = {},
    executorch::aten::ScalarType type = deduced_type,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return make_tensor_impl_ptr(
      std::move(sizes),
      std::vector<T>(std::move(list)),
      std::move(dim_order),
      std::move(strides),
      type,
      dynamism);
}

/**
 * Creates a TensorImplPtr that manages a newly created TensorImpl with the
 * specified properties.
 *
 * This template overload is specialized for cases where tensor data is provided
 * as an initializer list. If the specified `type` differs from the deduced type
 * of the initializer list's elements, and casting is allowed, the data will be
 * cast to the specified `type`. This allows for flexible creation of tensors
 * with data initializer list of one type and a different scalar type.
 *
 * @tparam T The C++ type of the tensor elements, deduced from the initializer
 * list.
 * @param list An initializer list containing the tensor's data.
 * @param type The scalar type of the tensor elements. If it differs from the
 * deduced type, the data will be cast to this type if allowed.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorImplPtr that manages the newly created TensorImpl.
 */
template <
    typename T = float,
    executorch::aten::ScalarType deduced_type =
        runtime::CppTypeToScalarType<T>::value>
inline TensorImplPtr make_tensor_impl_ptr(
    std::initializer_list<T> list,
    executorch::aten::ScalarType type = deduced_type,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  std::vector<executorch::aten::SizesType> sizes{
      executorch::aten::SizesType(list.size())};
  return make_tensor_impl_ptr(
      std::move(sizes), std::move(list), {0}, {1}, type, dynamism);
}

/**
 * Creates a TensorImplPtr to manage a Tensor with a single scalar value.
 *
 * @tparam T The C++ type of the scalar value.
 * @param value The scalar value used for the Tensor.
 * @return A TensorImplPtr managing the newly created TensorImpl.
 */
template <typename T>
inline TensorImplPtr make_tensor_impl_ptr(T value) {
  return make_tensor_impl_ptr({}, std::vector<T>{value});
}

/**
 * Creates a TensorImplPtr that manages a newly created TensorImpl with the
 * specified properties.
 *
 * This overload accepts a raw memory buffer stored in a std::vector<uint8_t>
 * and a scalar type to interpret the data. The vector is managed, and its
 * lifetime is tied to the TensorImpl.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param data A vector containing the raw memory buffer for the tensor's data.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorImplPtr managing the newly created TensorImpl.
 */
TensorImplPtr make_tensor_impl_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<uint8_t> data,
    std::vector<executorch::aten::DimOrderType> dim_order,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND);

/**
 * Creates a TensorImplPtr that manages a newly created TensorImpl with the
 * specified properties.
 *
 * This overload accepts a raw memory buffer stored in a std::vector<uint8_t>
 * and a scalar type to interpret the data. The vector is managed, and the
 * memory's lifetime is tied to the TensorImpl.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param data A vector containing the raw memory for the tensor's data.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorImplPtr managing the newly created TensorImpl.
 */
inline TensorImplPtr make_tensor_impl_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<uint8_t> data,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return make_tensor_impl_ptr(
      std::move(sizes), std::move(data), {}, {}, type, dynamism);
}

} // namespace extension
} // namespace executorch
