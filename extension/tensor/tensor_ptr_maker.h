/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/tensor/tensor_ptr.h>

namespace executorch {
namespace extension {

/**
 * A helper class for creating TensorPtr instances from raw data and tensor
 * properties. Note the the TensorPtr created by this class will not own the
 * data, so it must outlive the TensorPtr.
 *
 * TensorPtrMaker provides a fluent interface for specifying various properties
 * of a tensor, such as its type, sizes, data pointer, dimension order, strides,
 * and shape dynamism. The final tensor is created by invoking make_tensor_ptr()
 * or converting TensorPtrMaker to TensorPtr.
 */
class TensorPtrMaker final {
 public:
  // This class may have non-copyable members in the future.
  TensorPtrMaker(const TensorPtrMaker&) = delete;
  TensorPtrMaker& operator=(const TensorPtrMaker&) = delete;
  // But it is movable.
  TensorPtrMaker(TensorPtrMaker&&) = default;
  TensorPtrMaker& operator=(TensorPtrMaker&&) = default;
  /**
   * Sets the scalar type of the tensor elements.
   *
   * @param type The scalar type (e.g., float, int, bool).
   * @return Rvalue to this TensorPtrMaker for method chaining.
   */
  TensorPtrMaker&& type(exec_aten::ScalarType type) {
    type_ = type;
    return std::move(*this);
  }

  /**
   * Sets the order of dimensions in memory.
   *
   * @param dim_order A vector specifying the dimension order.
   * @return Rvalue to this TensorPtrMaker for method chaining.
   */
  TensorPtrMaker&& dim_order(std::vector<exec_aten::DimOrderType> dim_order) {
    dim_order_ = std::move(dim_order);
    return std::move(*this);
  }

  /**
   * Sets the strides for each dimension of the tensor.
   *
   * @param strides A vector specifying the stride for each dimension.
   * @return Rvalue to this TensorPtrMaker for method chaining.
   */
  TensorPtrMaker&& strides(std::vector<exec_aten::StridesType> strides) {
    strides_ = std::move(strides);
    return std::move(*this);
  }

  /**
   * Sets the shape dynamism of the tensor.
   *
   * @param dynamism Specifies whether the tensor's shape is static, dynamic, or
   * bounded.
   * @return Rvalue to this TensorPtrMaker for method chaining.
   */
  TensorPtrMaker&& dynamism(exec_aten::TensorShapeDynamism dynamism) {
    dynamism_ = dynamism;
    return std::move(*this);
  }

  /**
   * Sets a custom deleter function to manage the lifetime of the data buffer.
   *
   * @param deleter A function that will be called to delete the data buffer
   * when the Tensor object managed by the TensorPtr is destroyed. Explicitly
   * consuming an rvalue to avoid unnecessary copies when the deleter is a
   * lambda that has captured some state.
   * @return Rvalue to this TensorPtrMaker for method chaining.
   */
  TensorPtrMaker&& deleter(std::function<void(void*)>&& deleter) {
    deleter_ = std::move(deleter);
    return std::move(*this);
  }

  /**
   * Creates and returns a TensorPtr instance using the properties set in this
   * TensorPtrMaker.
   *
   * @return A TensorPtr instance that manages the newly created Tensor.
   */
  TensorPtr make_tensor_ptr() && {
    return ::executorch::extension::make_tensor_ptr(
        type_,
        std::move(sizes_),
        data_,
        std::move(dim_order_),
        std::move(strides_),
        dynamism_,
        std::move(deleter_));
  }

  /**
   * Implicit conversion operator to create a TensorPtr.
   *
   * @return A TensorPtr instance that manages the newly created Tensor.
   */
  operator TensorPtr() && {
    return std::move(*this).make_tensor_ptr();
  }

 private:
  TensorPtrMaker(
      void* data,
      std::vector<exec_aten::SizesType> sizes,
      exec_aten::ScalarType type)
      : sizes_(std::move(sizes)), data_(data), type_(type) {}

 private:
  // The following properties are required to create a Tensor.
  friend TensorPtrMaker for_blob(
      void* data,
      std::vector<exec_aten::SizesType> sizes,
      exec_aten::ScalarType type);

 private:
  std::vector<exec_aten::SizesType> sizes_;
  std::vector<exec_aten::StridesType> strides_;
  std::vector<exec_aten::DimOrderType> dim_order_;
  std::function<void(void*)> deleter_ = nullptr;
  void* data_ = nullptr;
  exec_aten::ScalarType type_ = exec_aten::ScalarType::Float;
  exec_aten::TensorShapeDynamism dynamism_ =
      exec_aten::TensorShapeDynamism::STATIC;
};

/**
 * Creates a TensorPtrMaker instance for building a TensorPtr from a raw data
 * pointer and tensor sizes.
 *
 * The TensorPtrMaker returned by this function allows for further customization
 * of the tensor's properties, such as data type, dimension order, strides, and
 * shape dynamism, before finalizing the TensorPtr creation.
 *
 * @param data A pointer to the raw data to be used by the tensor. It must
 * outlive the TensorPtr created by this function.
 * @param sizes A vector specifying the size of each dimension.
 * @param type The scalar type of the tensor elements.
 * @return A TensorPtrMaker instance for creating a TensorPtr.
 */
inline TensorPtrMaker for_blob(
    void* data,
    std::vector<exec_aten::SizesType> sizes,
    exec_aten::ScalarType type = exec_aten::ScalarType::Float) {
  return TensorPtrMaker(data, std::move(sizes), type);
}

/**
 * Creates a TensorPtr from a raw data pointer and tensor sizes, with an
 * optional dynamism setting.
 *
 * This function is a convenient way to create a tensor from existing data, with
 * the option to specify whether the tensor's shape is static, dynamic, or
 * bounded.
 *
 * @param data A pointer to the raw data to be used by the tensor. It must
 * outlive the TensorPtr created by this function.
 * @param sizes A vector specifying the size of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance that manages the newly created Tensor.
 */
inline TensorPtr from_blob(
    void* data,
    std::vector<exec_aten::SizesType> sizes,
    exec_aten::ScalarType type = exec_aten::ScalarType::Float,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::STATIC) {
  return for_blob(data, std::move(sizes), type)
      .dynamism(dynamism)
      .make_tensor_ptr();
}

/**
 * Creates a TensorPtr from a raw data pointer, tensor sizes, and strides, with
 * an optional dynamism setting.
 *
 * This function allows for the creation of a tensor from existing data, with
 * the option to specify custom strides for each dimension and whether the
 * tensor's shape is static, dynamic, or bounded.
 *
 * @param data A pointer to the raw data to be used by the tensor. It must
 * outlive the TensorPtr created by this function.
 * @param sizes A vector specifying the size of each dimension.
 * @param strides A vector specifying the stride for each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance that manages the newly created Tensor.
 */
inline TensorPtr from_blob(
    void* data,
    std::vector<exec_aten::SizesType> sizes,
    std::vector<exec_aten::StridesType> strides,
    exec_aten::ScalarType type = exec_aten::ScalarType::Float,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::STATIC) {
  return for_blob(data, std::move(sizes), type)
      .strides(std::move(strides))
      .dynamism(dynamism)
      .make_tensor_ptr();
}

/**
 * Creates a TensorPtr from a raw data pointer and tensor sizes, with an
 * optional dynamism setting.
 *
 * This function is a convenient way to create a tensor from existing data, with
 * the option to specify whether the tensor's shape is static, dynamic, or
 * bounded.
 *
 * @param data A pointer to the raw data to be used by the tensor. It must
 * outlive the TensorPtr created by this function.
 * @param sizes A vector specifying the size of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param deleter A function to delete the data when it's no longer needed.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance that manages the newly created Tensor.
 */
inline TensorPtr from_blob(
    void* data,
    std::vector<exec_aten::SizesType> sizes,
    exec_aten::ScalarType type,
    std::function<void(void*)>&& deleter,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::STATIC) {
  return for_blob(data, std::move(sizes), type)
      .deleter(std::move(deleter))
      .dynamism(dynamism)
      .make_tensor_ptr();
}

/**
 * Creates a TensorPtr from a raw data pointer, tensor sizes, and strides, with
 * an optional dynamism setting.
 *
 * This function allows for the creation of a tensor from existing data, with
 * the option to specify custom strides for each dimension and whether the
 * tensor's shape is static, dynamic, or bounded.
 *
 * @param data A pointer to the raw data to be used by the tensor. It must
 * outlive the TensorPtr created by this function.
 * @param sizes A vector specifying the size of each dimension.
 * @param strides A vector specifying the stride for each dimension.
 * @param type The scalar type of the tensor elements.
 * @param deleter A function to delete the data when it's no longer needed.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance that manages the newly created Tensor.
 */
inline TensorPtr from_blob(
    void* data,
    std::vector<exec_aten::SizesType> sizes,
    std::vector<exec_aten::StridesType> strides,
    exec_aten::ScalarType type,
    std::function<void(void*)>&& deleter,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::STATIC) {
  return for_blob(data, std::move(sizes), type)
      .strides(std::move(strides))
      .deleter(std::move(deleter))
      .dynamism(dynamism)
      .make_tensor_ptr();
}

} // namespace extension
} // namespace executorch
