/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace executorch {
namespace extension {

/**
 * A smart pointer type for managing the lifecycle of a Tensor.
 */
using TensorPtr = std::shared_ptr<executorch::aten::Tensor>;

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param data A pointer to the data buffer.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of the tensor.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @param deleter A custom deleter function for managing the lifetime of the
 * data buffer. If provided, this deleter will be called when the managed Tensor
 * object is destroyed.
 * @return A TensorPtr that manages the newly created Tensor.
 */
TensorPtr make_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    void* data,
    std::vector<executorch::aten::DimOrderType> dim_order,
    std::vector<executorch::aten::StridesType> strides,
    const executorch::aten::ScalarType type =
        executorch::aten::ScalarType::Float,
    const executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
    std::function<void(void*)> deleter = nullptr);

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param data A pointer to the data buffer.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @param deleter A custom deleter function for managing the lifetime of the
 * data buffer. If provided, this deleter will be called when the managed Tensor
 * object is destroyed.
 * @return A TensorPtr that manages the newly created Tensor.
 */
inline TensorPtr make_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    void* data,
    const executorch::aten::ScalarType type =
        executorch::aten::ScalarType::Float,
    const executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
    std::function<void(void*)> deleter = nullptr) {
  return make_tensor_ptr(
      std::move(sizes), data, {}, {}, type, dynamism, std::move(deleter));
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
template <
    typename T = float,
    executorch::aten::ScalarType deduced_type =
        runtime::CppTypeToScalarType<T>::value>
inline TensorPtr make_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<T> data,
    std::vector<executorch::aten::DimOrderType> dim_order = {},
    std::vector<executorch::aten::StridesType> strides = {},
    executorch::aten::ScalarType type = deduced_type,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  ET_CHECK_MSG(
      data.size() ==
          executorch::aten::compute_numel(sizes.data(), sizes.size()),
      "Data size does not match tensor size.");
  if (type != deduced_type) {
    ET_CHECK_MSG(
        runtime::canCast(deduced_type, type),
        "Cannot cast deduced type to specified type.");
    std::vector<uint8_t> casted_data(data.size() * runtime::elementSize(type));

    // Create a minimal context for error handling in ET_SWITCH
    struct {
      [[noreturn]] void fail(torch::executor::Error /* error */) {
        ET_CHECK_MSG(false, "Unsupported dtype in make_tensor_ptr");
      }
    } ctx;

    ET_SWITCH_REALHBBF16_TYPES(type, ctx, "make_tensor_ptr", CTYPE, [&] {
      std::transform(
          data.begin(),
          data.end(),
          reinterpret_cast<CTYPE*>(casted_data.data()),
          [](const T& val) { return static_cast<CTYPE>(val); });
    });
    const auto raw_data_ptr = casted_data.data();
    auto data_ptr =
        std::make_shared<std::vector<uint8_t>>(std::move(casted_data));
    return make_tensor_ptr(
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
  return make_tensor_ptr(
      std::move(sizes),
      raw_data_ptr,
      std::move(dim_order),
      std::move(strides),
      type,
      dynamism,
      [data_ptr = std::move(data_ptr)](void*) {});
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
template <
    typename T = float,
    executorch::aten::ScalarType deduced_type =
        runtime::CppTypeToScalarType<T>::value>
inline TensorPtr make_tensor_ptr(
    std::vector<T> data,
    executorch::aten::ScalarType type = deduced_type,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  std::vector<executorch::aten::SizesType> sizes{
      executorch::aten::SizesType(data.size())};
  return make_tensor_ptr(
      std::move(sizes), std::move(data), {0}, {1}, type, dynamism);
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
    executorch::aten::ScalarType deduced_type =
        runtime::CppTypeToScalarType<T>::value>
inline TensorPtr make_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    std::initializer_list<T> list,
    std::vector<executorch::aten::DimOrderType> dim_order = {},
    std::vector<executorch::aten::StridesType> strides = {},
    executorch::aten::ScalarType type = deduced_type,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return make_tensor_ptr(
      std::move(sizes),
      std::vector<T>(std::move(list)),
      std::move(dim_order),
      std::move(strides),
      type,
      dynamism);
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
template <
    typename T = float,
    executorch::aten::ScalarType deduced_type =
        runtime::CppTypeToScalarType<T>::value>
inline TensorPtr make_tensor_ptr(
    std::initializer_list<T> list,
    executorch::aten::ScalarType type = deduced_type,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  std::vector<executorch::aten::SizesType> sizes{
      executorch::aten::SizesType(list.size())};
  return make_tensor_ptr(
      std::move(sizes), std::move(list), {0}, {1}, type, dynamism);
}

/**
 * Creates a TensorPtr that manages a Tensor with a single scalar value.
 *
 * @tparam T The C++ type of the scalar value.
 * @param value The scalar value to be used for the Tensor.
 * @return A TensorPtr that manages the newly created TensorImpl.
 */
template <typename T>
inline TensorPtr make_tensor_ptr(T value) {
  return make_tensor_ptr({}, std::vector<T>{value});
}

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * This overload accepts a raw memory buffer stored in a std::vector<uint8_t>
 * and a scalar type to interpret the data. The vector is managed, and the
 * memory's lifetime is tied to the TensorImpl.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param data A vector containing the raw memory for the tensor's data.
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorPtr managing the newly created Tensor.
 */
TensorPtr make_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<uint8_t> data,
    std::vector<executorch::aten::DimOrderType> dim_order,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND);

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * This overload accepts a raw memory buffer stored in a std::vector<uint8_t>
 * and a scalar type to interpret the data. The vector is managed, and the
 * memory's lifetime is tied to the TensorImpl.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param data A vector containing the raw memory for the tensor's data.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @return A TensorPtr managing the newly created Tensor.
 */
inline TensorPtr make_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<uint8_t> data,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return make_tensor_ptr(
      std::move(sizes), std::move(data), {}, {}, type, dynamism);
}

/**
 * Creates a TensorPtr to manage a new Tensor with the same properties
 * as the given Tensor, sharing the same data without owning it.
 *
 * @param tensor The Tensor whose properties are used to create a new TensorPtr.
 * @return A new TensorPtr managing a Tensor with the same properties as the
 * original.
 */
inline TensorPtr make_tensor_ptr(const executorch::aten::Tensor& tensor) {
  return make_tensor_ptr(
      std::vector<executorch::aten::SizesType>(
          tensor.sizes().begin(), tensor.sizes().end()),
      tensor.mutable_data_ptr(),
#ifndef USE_ATEN_LIB
      std::vector<executorch::aten::DimOrderType>(
          tensor.dim_order().begin(), tensor.dim_order().end()),
#else // USE_ATEN_LIB
      {},
#endif // USE_ATEN_LIB
      std::vector<executorch::aten::StridesType>(
          tensor.strides().begin(), tensor.strides().end()),
      tensor.scalar_type()
#ifndef USE_ATEN_LIB
          ,
      tensor.shape_dynamism()
#endif // USE_ATEN_LIB
  );
}

/**
 * Creates a TensorPtr to manage a new Tensor with the same properties
 * as the Tensor referenced by the given TensorPtr, sharing the same data
 * without owning it.
 *
 * This is a convenience overload equivalent to make_tensor_ptr(*tensor_ptr).
 * It does not extend the lifetime of the underlying buffer; if the original
 * owner releases the storage, all views aliasing it become dangling.
 *
 * @param tensor_ptr The TensorPtr whose underlying Tensor is used to initialize
 *                   the returned view.
 * @return A new TensorPtr managing a Tensor with the same properties as the
 *         original.
 */
inline TensorPtr make_tensor_ptr(const TensorPtr& tensor_ptr) {
  return make_tensor_ptr(*tensor_ptr);
}

/**
 * Creates a TensorPtr that manages a new Tensor with the same properties
 * as the given Tensor, but with a copy of the data owned by the returned
 * TensorPtr, or nullptr if the original data is null.
 *
 * @param tensor The Tensor to clone.
 * @return A new TensorPtr that manages a Tensor with the same properties as the
 * original but with copied data.
 */
TensorPtr clone_tensor_ptr(const executorch::aten::Tensor& tensor);

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
 * Resizes the Tensor managed by the provided TensorPtr to the new sizes.
 *
 * @param tensor A TensorPtr managing the Tensor to resize.
 * @param sizes A vector representing the new sizes for each dimension.
 * @return Error::Ok on success, or an appropriate error code on failure.
 */
ET_NODISCARD
runtime::Error resize_tensor_ptr(
    TensorPtr& tensor,
    const std::vector<executorch::aten::SizesType>& sizes);

} // namespace extension
} // namespace executorch
