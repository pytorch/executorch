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
class TensorImplPtr {
 public:
  constexpr TensorImplPtr() = default;
  explicit constexpr TensorImplPtr(std::nullptr_t) {}
  TensorImplPtr(
      exec_aten::TensorImpl tensor_impl,
      std::vector<exec_aten::SizesType> sizes,
      std::vector<exec_aten::DimOrderType> dim_order,
      std::vector<exec_aten::StridesType> strides,
      std::function<void(void*)> data_deleter = nullptr)
      : repr_(std::make_shared<HeapData>(
            std::move(tensor_impl),
            std::move(sizes),
            std::move(dim_order),
            std::move(strides),
            std::move(data_deleter))) {}

  operator bool() const {
    return static_cast<bool>(repr_);
  }

  exec_aten::TensorImpl* get() const {
    return repr_ ? &repr_->tensor_impl_ : nullptr;
  }

  exec_aten::TensorImpl* operator->() const {
    return get();
  }

  exec_aten::TensorImpl& operator*() const {
    ET_DCHECK(repr_ != nullptr);
    return *get();
  }

  void reset() {
    repr_.reset();
  }

  void swap(TensorImplPtr& other) noexcept {
    repr_.swap(other.repr_);
  }

  bool operator==(const TensorImplPtr& rhs) const {
    return repr_ == rhs.repr_;
  }

  bool operator!=(const TensorImplPtr& rhs) const {
    return !(*this == rhs);
  }

  bool operator==(std::nullptr_t) const {
    return !operator bool();
  }

  bool operator!=(std::nullptr_t) const {
    return !(*this == nullptr);
  }

  auto use_count() const noexcept {
    return repr_.use_count();
  }

 private:
  struct HeapData {
    exec_aten::TensorImpl tensor_impl_;
    // TODO: consolidate these allocations similar to torch::Tensor's
    // SizesAndStrides?
    std::vector<exec_aten::SizesType> sizes_;
    std::vector<exec_aten::DimOrderType> dim_order_;
    std::vector<exec_aten::StridesType> strides_;
    // TODO: don't pay for the deleter if it wasn't set.
    std::function<void(void*)> data_deleter_;

    HeapData(
        exec_aten::TensorImpl&& ti,
        std::vector<exec_aten::SizesType>&& sizes,
        std::vector<exec_aten::DimOrderType>&& dim_order,
        std::vector<exec_aten::StridesType>&& strides,
        std::function<void(void*)>&& data_deleter)
        : tensor_impl_(std::move(ti)),
          sizes_(std::move(sizes)),
          dim_order_(std::move(dim_order)),
          strides_(std::move(strides)),
          data_deleter_(std::move(data_deleter)) {}

    ~HeapData() {
      if (data_deleter_) {
        data_deleter_(tensor_impl_.mutable_data());
      }
    }
  };
  std::shared_ptr<HeapData> repr_;
};
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
    exec_aten::ScalarType deduced_type = runtime::CppTypeToScalarType<T>::value>
TensorImplPtr make_tensor_impl_ptr(
    std::vector<exec_aten::SizesType> sizes,
    std::vector<T> data,
    std::vector<exec_aten::DimOrderType> dim_order = {},
    std::vector<exec_aten::StridesType> strides = {},
    exec_aten::ScalarType type = deduced_type,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
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
    exec_aten::ScalarType deduced_type = runtime::CppTypeToScalarType<T>::value>
inline TensorImplPtr make_tensor_impl_ptr(
    std::vector<T> data,
    exec_aten::ScalarType type = deduced_type,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
  std::vector<exec_aten::SizesType> sizes{exec_aten::SizesType(data.size())};
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
    exec_aten::ScalarType deduced_type = runtime::CppTypeToScalarType<T>::value>
inline TensorImplPtr make_tensor_impl_ptr(
    std::vector<exec_aten::SizesType> sizes,
    std::initializer_list<T> list,
    std::vector<exec_aten::DimOrderType> dim_order = {},
    std::vector<exec_aten::StridesType> strides = {},
    exec_aten::ScalarType type = deduced_type,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
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
    exec_aten::ScalarType deduced_type = runtime::CppTypeToScalarType<T>::value>
inline TensorImplPtr make_tensor_impl_ptr(
    std::initializer_list<T> list,
    exec_aten::ScalarType type = deduced_type,
    exec_aten::TensorShapeDynamism dynamism =
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND) {
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
