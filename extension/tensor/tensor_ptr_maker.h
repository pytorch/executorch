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
 * properties. Note that the TensorPtr created by this class does not own the
 * data, so the data must outlive the TensorPtr.
 *
 * TensorPtrMaker provides a fluent interface for specifying various tensor
 * properties, such as type, sizes, data pointer, dimension order, strides, and
 * shape dynamism. The final tensor is created by invoking make_tensor_ptr() or
 * by converting TensorPtrMaker to TensorPtr.
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
  TensorPtrMaker&& type(executorch::aten::ScalarType type) {
    type_ = type;
    return std::move(*this);
  }

  /**
   * Sets the order of dimensions in memory.
   *
   * @param dim_order A vector specifying the dimension order.
   * @return Rvalue to this TensorPtrMaker for method chaining.
   */
  TensorPtrMaker&& dim_order(
      std::vector<executorch::aten::DimOrderType> dim_order) {
    dim_order_ = std::move(dim_order);
    return std::move(*this);
  }

  /**
   * Sets the strides for each dimension of the tensor.
   *
   * @param strides A vector specifying the stride for each dimension.
   * @return Rvalue to this TensorPtrMaker for method chaining.
   */
  TensorPtrMaker&& strides(std::vector<executorch::aten::StridesType> strides) {
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
  TensorPtrMaker&& dynamism(executorch::aten::TensorShapeDynamism dynamism) {
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
        std::move(sizes_),
        data_,
        std::move(dim_order_),
        std::move(strides_),
        type_,
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
      std::vector<executorch::aten::SizesType> sizes,
      executorch::aten::ScalarType type)
      : sizes_(std::move(sizes)), data_(data), type_(type) {}

 private:
  // The following properties are required to create a Tensor.
  friend TensorPtrMaker for_blob(
      void* data,
      std::vector<executorch::aten::SizesType> sizes,
      executorch::aten::ScalarType type);

 private:
  std::vector<executorch::aten::SizesType> sizes_;
  std::vector<executorch::aten::StridesType> strides_;
  std::vector<executorch::aten::DimOrderType> dim_order_;
  std::function<void(void*)> deleter_ = nullptr;
  void* data_ = nullptr;
  executorch::aten::ScalarType type_ = executorch::aten::ScalarType::Float;
  executorch::aten::TensorShapeDynamism dynamism_ =
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND;
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
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float) {
  return TensorPtrMaker(data, std::move(sizes), type);
}

/**
 * Creates a TensorPtr from a raw data pointer and tensor sizes, with an
 * optional dynamism setting.
 *
 * This function provides a convenient way to create a tensor from existing
 * data, with the option to specify whether the tensor's shape is static or
 * dynamic.
 *
 * @param data A pointer to the raw data used by the tensor. The data must
 * outlive the TensorPtr created by this function.
 * @param sizes A vector specifying the size of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr from_blob(
    void* data,
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
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
 * tensor’s shape is static, dynamic, or bounded.
 *
 * @param data A pointer to the raw data used by the tensor. The data must
 * outlive the TensorPtr created by this function.
 * @param sizes A vector specifying the size of each dimension.
 * @param strides A vector specifying the stride for each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static, dynamic, or
 * bounded.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr from_blob(
    void* data,
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
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
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType type,
    std::function<void(void*)>&& deleter,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
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
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type,
    std::function<void(void*)>&& deleter,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return for_blob(data, std::move(sizes), type)
      .strides(std::move(strides))
      .deleter(std::move(deleter))
      .dynamism(dynamism)
      .make_tensor_ptr();
}

/**
 * Creates a TensorPtr with the specified sizes, strides, and properties.
 *
 * This function allocates memory for the tensor elements but does not
 * initialize them with any specific values. The tensor is created with the
 * specified strides.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param strides A vector specifying the stride for each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
TensorPtr empty_strided(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND);

/**
 * Creates an empty TensorPtr with the same size and properties as the given
 * tensor.
 *
 * This function allocates memory for the tensor elements but does not
 * initialize them with any specific values.
 *
 * @param other A reference to another tensor, whose size and properties are
 * used.
 * @param type The scalar type of the tensor elements. If not provided, the
 * scalar type of the other tensor is used.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr empty_like(
    const TensorPtr& other,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  if (type == executorch::aten::ScalarType::Undefined) {
    type = other->scalar_type();
  }
  return empty_strided(
      {other->sizes().begin(), other->sizes().end()},
      {other->strides().begin(), other->strides().end()},
      type,
      dynamism);
}

/**
 * Creates an empty TensorPtr with the specified sizes and properties.
 *
 * This function allocates memory for the tensor elements but does not
 * initialize them with any specific values.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr empty(
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return empty_strided(std::move(sizes), {}, type, dynamism);
}

/**
 * Creates a TensorPtr filled with the specified value.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param strides A vector specifying the stride for each dimension.
 * @param fill_value The value to fill the tensor with.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
TensorPtr full_strided(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::Scalar fill_value,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND);

/**
 * Creates a TensorPtr filled with the specified value, with the same size and
 * properties as another tensor.
 *
 * @param other A reference to another tensor, whose size and properties will be
 * used.
 * @param fill_value The value to fill the tensor with.
 * @param type The scalar type of the tensor elements. If not specified, the
 * scalar type of the other tensor is used.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr full_like(
    const TensorPtr& other,
    executorch::aten::Scalar fill_value,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  if (type == executorch::aten::ScalarType::Undefined) {
    type = other->scalar_type();
  }
  return full_strided(
      {other->sizes().begin(), other->sizes().end()},
      {other->strides().begin(), other->strides().end()},
      fill_value,
      type,
      dynamism);
}

/**
 * Creates a TensorPtr filled with the specified value.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param fill_value The value used to fill the tensor.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr full(
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::Scalar fill_value,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return full_strided(std::move(sizes), {}, fill_value, type, dynamism);
}

/**
 * Creates a TensorPtr holding a scalar value.
 *
 * @param value The scalar value for the tensor.
 * @param type The scalar type of the tensor elements.
 * @return A TensorPtr instance managing the newly created scalar Tensor.
 */
inline TensorPtr scalar_tensor(
    executorch::aten::Scalar value,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float) {
  return full({}, value, type);
}

/**
 * Creates a TensorPtr filled with ones, with the same size and properties as
 * another tensor.
 *
 * @param other A reference to another tensor, whose size and properties are
 * used.
 * @param type The scalar type of the tensor elements. If not provided, the
 * scalar type of the other tensor is used.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr ones_like(
    const TensorPtr& other,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return full_like(other, 1, type, dynamism);
}

/**
 * Creates a TensorPtr filled with ones.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr ones(
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return full(std::move(sizes), 1, type, dynamism);
}

/**
 * Creates a TensorPtr filled with zeros, with the same size and properties as
 * another tensor.
 *
 * @param other A reference to another tensor, whose size and properties will be
 * used.
 * @param type The scalar type of the tensor elements. If not specified, the
 * scalar type of the `other` tensor is used.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr zeros_like(
    const TensorPtr& other,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return full_like(other, 0, type, dynamism);
}

/**
 * Creates a TensorPtr filled with zeros.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr zeros(
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return full(std::move(sizes), 0, type, dynamism);
}

/**
 * Creates a TensorPtr filled with random values between 0 and 1.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param strides A vector specifying the stride for each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 **/
TensorPtr rand_strided(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND);

/**
 * Creates a TensorPtr filled with random values between 0 and 1.
 *
 * @param other A reference to another tensor, whose size and properties will be
 * used.
 * @param type The scalar type of the tensor elements. If not specified, the
 * scalar type of the other tensor is used.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr rand_like(
    const TensorPtr& other,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  if (type == executorch::aten::ScalarType::Undefined) {
    type = other->scalar_type();
  }
  return rand_strided(
      {other->sizes().begin(), other->sizes().end()},
      {other->strides().begin(), other->strides().end()},
      type,
      dynamism);
}

/**
 * Creates a TensorPtr filled with random values between 0 and 1.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr rand(
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return rand_strided(std::move(sizes), {}, type, dynamism);
}

/**
 * Creates a TensorPtr filled with random values between 0 and 1, with specified
 * strides.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param strides A vector specifying the stride for each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
TensorPtr randn_strided(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND);

/**
 * Creates a TensorPtr filled with random values from a normal distribution.
 *
 * @param other A reference to another tensor, whose size and properties will be
 * used.
 * @param type The scalar type of the tensor elements. If not specified, the
 * scalar type of the other tensor is used.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr randn_like(
    const TensorPtr& other,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  if (type == executorch::aten::ScalarType::Undefined) {
    type = other->scalar_type();
  }
  return randn_strided(
      {other->sizes().begin(), other->sizes().end()},
      {other->strides().begin(), other->strides().end()},
      type,
      dynamism);
}

/**
 * Creates a TensorPtr filled with random values sampled from a normal
 * distribution.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr randn(
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Float,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return randn_strided(std::move(sizes), {}, type, dynamism);
}

/**
 * Creates a TensorPtr filled with random integer values in the given range.
 *
 * @param low The lower bound (inclusive) of the random values.
 * @param high The upper bound (exclusive) of the random values.
 * @param sizes A vector specifying the size of each dimension.
 * @param strides A vector specifying the stride for each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
TensorPtr randint_strided(
    int64_t low,
    int64_t high,
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Int,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND);

/**
 * Creates a TensorPtr filled with random integer values in the given range.
 *
 * @param other A reference to another tensor, whose size and properties will be
 * used.
 * @param low The lower bound (inclusive) of the random values.
 * @param high The upper bound (exclusive) of the random values.
 * @param type The scalar type of the tensor elements. If not specified, the
 * scalar type of the other tensor is used.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr randint_like(
    const TensorPtr& other,
    int64_t low,
    int64_t high,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Undefined,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  if (type == executorch::aten::ScalarType::Undefined) {
    type = other->scalar_type();
  }
  return randint_strided(
      low,
      high,
      {other->sizes().begin(), other->sizes().end()},
      {other->strides().begin(), other->strides().end()},
      type,
      dynamism);
}

/**
 * Creates a TensorPtr filled with random integer values within the specified
 * range.
 *
 * @param low The inclusive lower bound of the random values.
 * @param high The exclusive upper bound of the random values.
 * @param sizes A vector specifying the size of each dimension.
 * @param type The scalar type of the tensor elements.
 * @param dynamism Specifies whether the tensor's shape is static or dynamic.
 * @return A TensorPtr instance managing the newly created Tensor.
 */
inline TensorPtr randint(
    int64_t low,
    int64_t high,
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType type = executorch::aten::ScalarType::Int,
    executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  return randint_strided(low, high, std::move(sizes), {}, type, dynamism);
}

} // namespace extension
} // namespace executorch
