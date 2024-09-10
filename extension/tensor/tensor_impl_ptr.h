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
 * A smart pointer type for managing the lifecycle of a TensorImpl.
 *
 * TensorImplPtr uses a shared pointer because multiple Tensor objects might
 * share the same underlying data and metadata. This shared ownership model
 * ensures that the TensorImpl is only destroyed when all references to it are
 * gone, providing a safe and efficient way to manage shared tensor
 * implementations. This abstraction is designed to be a safer and more
 * convenient alternative to the original TensorImpl, which does not
 * manage metadata by design.
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
 * @param type The scalar type of the tensor elements.
 * @param sizes A vector specifying the size of each dimension.
 * @param data A pointer to the data buffer.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of each dimension.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @param deleter A custom deleter function for managing the lifetime of the
 * data buffer. If provided, this deleter will be called when the managed
 * TensorImpl object is destroyed.
 * @return A TensorImplPtr managing the newly created TensorImpl.
 */
TensorImplPtr make_tensor_impl_ptr(
    exec_aten::ScalarType type,
    std::vector<exec_aten::SizesType> sizes,
    void* data,
    std::vector<exec_aten::DimOrderType> dim_order = {},
    std::vector<exec_aten::StridesType> strides = {},
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::STATIC,
    std::function<void(void*)> deleter = nullptr);

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
 * @param sizes A vector specifying the size of each dimension.
 * @param data A vector containing the tensor's data.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of each dimension.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorImplPtr that manages the newly created TensorImpl.
 */
template <typename T = float>
TensorImplPtr make_tensor_impl_ptr(
    std::vector<exec_aten::SizesType> sizes,
    std::vector<T> data,
    std::vector<exec_aten::DimOrderType> dim_order = {},
    std::vector<exec_aten::StridesType> strides = {},
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::STATIC) {
  constexpr exec_aten::ScalarType scalar_type =
      runtime::CppTypeToScalarType<T>::value;
  const auto raw_data_ptr = data.data();
  auto data_ptr = std::make_shared<std::vector<T>>(std::move(data));
  return make_tensor_impl_ptr(
      scalar_type,
      std::move(sizes),
      raw_data_ptr,
      std::move(dim_order),
      std::move(strides),
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
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorImplPtr that manages the newly created TensorImpl.
 */
template <typename T = float>
TensorImplPtr make_tensor_impl_ptr(
    std::vector<T> data,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::STATIC) {
  constexpr exec_aten::ScalarType scalar_type =
      runtime::CppTypeToScalarType<T>::value;
  std::vector<exec_aten::SizesType> sizes{exec_aten::SizesType(data.size())};
  const auto raw_data_ptr = data.data();
  auto data_ptr = std::make_shared<std::vector<T>>(std::move(data));
  return make_tensor_impl_ptr(
      scalar_type,
      std::move(sizes),
      raw_data_ptr,
      {0},
      {1},
      dynamism,
      [data_ptr = std::move(data_ptr)](void*) {});
}

/**
 * Creates a TensorImplPtr that manages a newly created TensorImpl with the
 * specified properties.
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
 * @return A TensorImplPtr managing the newly created TensorImpl.
 */
TensorImplPtr make_tensor_impl_ptr(
    exec_aten::ScalarType scalar_type,
    std::vector<exec_aten::SizesType> sizes,
    std::vector<uint8_t> data,
    std::vector<exec_aten::DimOrderType> dim_order = {},
    std::vector<exec_aten::StridesType> strides = {},
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::STATIC);

} // namespace extension
} // namespace executorch
