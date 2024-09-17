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
using TensorImplPtr = std::shared_ptr<exec_aten::TensorImpl>;
#else
/**
 * A smart pointer type for managing the lifecycle of a TensorImpl.
 *
 * TensorImplPtr uses an intrusive pointer when working with ATen, ensuring
 * efficient reference counting and shared ownership of the underlying data and
 * metadata.
 */
using TensorImplPtr =
    c10::intrusive_ptr<exec_aten::TensorImpl, at::UndefinedTensorImpl>;
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
    std::vector<exec_aten::SizesType> sizes,
    void* data,
    std::vector<exec_aten::DimOrderType> dim_order,
    std::vector<exec_aten::StridesType> strides,
    exec_aten::ScalarType type = exec_aten::ScalarType::Float,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND,
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
    std::vector<exec_aten::SizesType> sizes,
    void* data,
    exec_aten::ScalarType type = exec_aten::ScalarType::Float,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND,
    std::function<void(void*)> deleter = nullptr) {
  return make_tensor_impl_ptr(
      std::move(sizes), data, {}, {}, type, dynamism, std::move(deleter));
}

/**
 * Creates a TensorImplPtr that manages a newly created TensorImpl with the
 * specified properties.
 *
 * This template overload is specialized for cases where tensor data is provided
 * as a vector. The scalar type is automatically deduced from the vector's data
 * type. The deleter ensures that the data vector is properly managed, with its
 * lifetime tied to the TensorImpl.
 *
 * @tparam T The C++ type of the tensor elements, deduced from the vector.
 * @param sizes A vector specifying the size of each dimension.
 * @param data A vector containing the tensor's data.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorImplPtr that manages the newly created TensorImpl.
 */
template <
    typename T = float,
    exec_aten::ScalarType deduced_type = runtime::CppTypeToScalarType<T>::value>
inline TensorImplPtr make_tensor_impl_ptr(
    std::vector<exec_aten::SizesType> sizes,
    std::vector<T> data,
    std::vector<exec_aten::DimOrderType> dim_order = {},
    std::vector<exec_aten::StridesType> strides = {},
    exec_aten::ScalarType type = deduced_type,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  ET_CHECK_MSG(type == deduced_type, "Type does not match the deduced type.");
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
 * This template overload is specialized for cases where the tensor data is
 * provided as a vector. The scalar type is automatically deduced from the
 * vector's data type. The deleter ensures that the data vector is properly
 * managed and its lifetime is tied to the TensorImpl.
 *
 * @tparam T The C++ type of the tensor elements, deduced from the vector.
 * @param data A vector containing the tensor's data.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorImplPtr that manages the newly created TensorImpl.
 */
template <
    typename T = float,
    exec_aten::ScalarType deduced_type = runtime::CppTypeToScalarType<T>::value>
inline TensorImplPtr make_tensor_impl_ptr(
    std::vector<T> data,
    exec_aten::ScalarType type = deduced_type,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  ET_CHECK_MSG(type == deduced_type, "Type does not match the deduced type.");
  std::vector<exec_aten::SizesType> sizes{exec_aten::SizesType(data.size())};
  return make_tensor_impl_ptr(
      std::move(sizes), std::move(data), {0}, {1}, type, dynamism);
}

/**
 * Creates a TensorImplPtr that manages a newly created TensorImpl with the
 * specified properties.
 *
 * This template overload is specialized for cases where tensor data is provided
 * as an initializer list. The scalar type is automatically deduced from the
 * initializer list's data type. The deleter ensures that the data is properly
 * managed, with its lifetime tied to the TensorImpl.
 *
 * @tparam T The C++ type of the tensor elements, deduced from the initializer
 * list.
 * @param sizes A vector specifying the size of each dimension.
 * @param list An initializer list containing the tensor's data.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorImplPtr that manages the newly created TensorImpl.
 */
template <
    typename T = float,
    exec_aten::ScalarType deduced_type = runtime::CppTypeToScalarType<T>::value>
inline TensorImplPtr make_tensor_impl_ptr(
    std::vector<exec_aten::SizesType> sizes,
    std::initializer_list<T> list,
    std::vector<exec_aten::DimOrderType> dim_order = {},
    std::vector<exec_aten::StridesType> strides = {},
    exec_aten::ScalarType type = deduced_type,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  ET_CHECK_MSG(type == deduced_type, "Type does not match the deduced type.");
  auto data = std::vector<T>(std::move(list));
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
 * This template overload is specialized for cases where the tensor data is
 * provided as an initializer list. The scalar type is automatically deduced
 * from the initializer list's data type. The deleter ensures that the data is
 * properly managed and its lifetime is tied to the TensorImpl.
 *
 * @tparam T The C++ type of the tensor elements, deduced from the initializer
 * list.
 * @param sizes A vector specifying the size of each dimension.
 * @param list An initializer list containing the tensor's data.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorImplPtr that manages the newly created TensorImpl.
 */
template <
    typename T = float,
    exec_aten::ScalarType deduced_type = runtime::CppTypeToScalarType<T>::value>
inline TensorImplPtr make_tensor_impl_ptr(
    std::initializer_list<T> list,
    exec_aten::ScalarType type = deduced_type,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  ET_CHECK_MSG(type == deduced_type, "Type does not match the deduced type.");
  std::vector<exec_aten::SizesType> sizes{exec_aten::SizesType(list.size())};
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
    std::vector<exec_aten::SizesType> sizes,
    std::vector<uint8_t> data,
    std::vector<exec_aten::DimOrderType> dim_order,
    std::vector<exec_aten::StridesType> strides,
    exec_aten::ScalarType type = exec_aten::ScalarType::Float,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);

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
    std::vector<exec_aten::SizesType> sizes,
    std::vector<uint8_t> data,
    exec_aten::ScalarType type = exec_aten::ScalarType::Float,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return make_tensor_impl_ptr(
      std::move(sizes), std::move(data), {}, {}, type, dynamism);
}

} // namespace extension
} // namespace executorch
