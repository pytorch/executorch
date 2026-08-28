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

#include <c10/macros/Macros.h>
#include <c10/util/safe_numerics.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")

namespace executorch {
namespace extension {

/**
 * A smart pointer type for managing the lifecycle of a Tensor.
 */
using TensorPtr = std::shared_ptr<executorch::aten::Tensor>;

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * The `device` parameter sets the Tensor's device location only — no data is
 * allocated or copied. The caller is responsible for ensuring `data` already
 * lives on the requested device; construct the `executorch::aten::Device` from
 * the runtime environment and pass it in. To copy CPU data to a device, use
 * `clone_tensor_ptr` with a device target instead, or `tensor.to(device)` in a
 * USE_ATEN_LIB build.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param data A pointer to the data buffer (CPU or device, see device).
 * @param dim_order A vector specifying the order of dimensions.
 * @param strides A vector specifying the strides of the tensor.
 * @param type The scalar type of the tensor elements.
 * @param device The device on which `data` resides (default CPU).
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
    executorch::aten::Device device =
        executorch::aten::Device(executorch::aten::DeviceType::CPU),
    const executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
    std::function<void(void*)> deleter = nullptr);

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * Convenience overload for the primary factory; see the primary overload for
 * device semantics.
 *
 * @param sizes A vector specifying the size of each dimension.
 * @param data A pointer to the data buffer (CPU or device, see device).
 * @param type The scalar type of the tensor elements.
 * @param device The device on which `data` resides (default CPU).
 * @param dynamism Specifies the mutability of the tensor's shape.
 * @param deleter A custom deleter function for managing the lifetime of the
 * data buffer.
 * @return A TensorPtr that manages the newly created Tensor.
 */
inline TensorPtr make_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    void* data,
    const executorch::aten::ScalarType type =
        executorch::aten::ScalarType::Float,
    executorch::aten::Device device =
        executorch::aten::Device(executorch::aten::DeviceType::CPU),
    const executorch::aten::TensorShapeDynamism dynamism =
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
    std::function<void(void*)> deleter = nullptr) {
  return make_tensor_ptr(
      std::move(sizes),
      data,
      {},
      {},
      type,
      device,
      dynamism,
      std::move(deleter));
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
 * The result is always a CPU tensor. To move it to a device, use
 * `clone_tensor_ptr` with a device target, or `tensor.to(device)` in a
 * USE_ATEN_LIB build.
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
  auto numel_result = executorch::aten::safe_numel(sizes.data(), sizes.size());
  ET_CHECK_MSG(
      numel_result.ok(),
      "safe_numel failed: %d",
      static_cast<int>(numel_result.error()));
  ET_CHECK_MSG(
      data.size() == static_cast<size_t>(numel_result.get()),
      "Data size does not match tensor size.");
  if (type != deduced_type) {
    ET_CHECK_MSG(
        runtime::canCast(deduced_type, type),
        "Cannot cast deduced type to specified type.");
    size_t casted_bytes = 0;
    ET_CHECK_MSG(
        !c10::mul_overflows(
            data.size(),
            static_cast<size_t>(aten::elementSize(type)),
            &casted_bytes),
        "casted_data size overflow: %zu elements * %zu bytes/element",
        data.size(),
        static_cast<size_t>(aten::elementSize(type)));
    std::vector<uint8_t> casted_data(casted_bytes);

    // Create a minimal context for error handling in ET_SWITCH
    struct {
      [[noreturn]] void fail(torch::executor::Error /* error */) {
        ET_CHECK_MSG(false, "Unsupported dtype in make_tensor_ptr");
      }
    } ctx;

    ET_SWITCH_REALHBBF16_AND_UINT_TYPES(
        type, ctx, "make_tensor_ptr", CTYPE, [&] {
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
        executorch::aten::Device(executorch::aten::DeviceType::CPU),
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
      executorch::aten::Device(executorch::aten::DeviceType::CPU),
      dynamism,
      [data_ptr = std::move(data_ptr)](void*) {});
}

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * This template overload is specialized for cases where the tensor data is
 * provided as a vector. The scalar type is automatically deduced from the
 * vector's data type.
 *
 * The result is always a CPU tensor. To move it to a device, use
 * `clone_tensor_ptr` with a device target, or `tensor.to(device)` in a
 * USE_ATEN_LIB build.
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
 * from the initializer list's data type.
 *
 * The result is always a CPU tensor. To move it to a device, use
 * `clone_tensor_ptr` with a device target, or `tensor.to(device)` in a
 * USE_ATEN_LIB build.
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
 * initializer list's elements.
 *
 * The result is always a CPU tensor. To move it to a device, use
 * `clone_tensor_ptr` with a device target, or `tensor.to(device)` in a
 * USE_ATEN_LIB build.
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
  return make_tensor_ptr(
      std::vector<executorch::aten::SizesType>{}, std::vector<T>{value});
}

/**
 * Creates a TensorPtr that manages a Tensor with the specified properties.
 *
 * This overload accepts a raw memory buffer stored in a std::vector<uint8_t>
 * and a scalar type to interpret the data. The vector is managed, and the
 * memory's lifetime is tied to the TensorImpl. The result is always a CPU
 * tensor.
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
 * Convenience overload for the raw-buffer factory; see above. The result is
 * always a CPU tensor.
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
 * Creates a TensorPtr to manage a new Tensor that aliases the given Tensor's
 * storage, with optional metadata overrides. Shape dynamism is inherited from
 * the source tensor.
 *
 * If an override is provided (non-empty), it is passed as-is. If an override is
 * empty, the corresponding metadata is reused from the source tensor when it
 * fits; otherwise it is left empty for the core factory to derive a valid
 * configuration. If `dim_order` is empty but `strides` is provided, `dim_order`
 * is left empty so the core may infer it from the provided strides.
 *
 * This overload always aliases — it never copies. To copy a tensor's data to
 * a device, use `clone_tensor_ptr` with a device target, or
 * `tensor.to(device)` in a USE_ATEN_LIB build.
 *
 * @param tensor The source tensor to alias.
 * @param sizes Optional sizes override.
 * @param dim_order Optional dimension order override.
 * @param strides Optional strides override.
 * @param deleter A custom deleter function for managing the lifetime of the
 * original Tensor.
 * @return A TensorPtr aliasing the same storage with requested metadata.
 */
inline TensorPtr make_tensor_ptr(
    const executorch::aten::Tensor& tensor,
    std::vector<executorch::aten::SizesType> sizes = {},
    std::vector<executorch::aten::DimOrderType> dim_order = {},
    std::vector<executorch::aten::StridesType> strides = {},
    std::function<void(void*)> deleter = nullptr) {
  if (sizes.empty()) {
    sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
  }
  const auto same_rank = sizes.size() == static_cast<size_t>(tensor.dim());
  const auto same_shape = same_rank &&
      std::equal(sizes.begin(), sizes.end(), tensor.sizes().begin());
  auto element_count_result =
      executorch::aten::safe_numel(sizes.data(), sizes.size());
  ET_CHECK_MSG(
      element_count_result.ok(),
      "safe_numel failed: %d",
      static_cast<int>(element_count_result.error()));
  const auto element_count = element_count_result.get();
  const auto parent_element_count = tensor.numel();
  ET_CHECK_MSG(
      element_count <= parent_element_count,
      "Requested view has %zd elements, but source tensor only has %zd.",
      static_cast<ssize_t>(element_count),
      static_cast<ssize_t>(parent_element_count));
#ifndef USE_ATEN_LIB
  if (dim_order.empty() && strides.empty() && same_rank) {
    dim_order.assign(tensor.dim_order().begin(), tensor.dim_order().end());
  }
#endif // USE_ATEN_LIB
  if (strides.empty() && dim_order.empty() && same_shape) {
    strides.assign(tensor.strides().begin(), tensor.strides().end());
  }
  return make_tensor_ptr(
      std::move(sizes),
      tensor.mutable_data_ptr(),
      std::move(dim_order),
      std::move(strides),
      tensor.scalar_type(),
#ifndef USE_ATEN_LIB
      executorch::aten::Device(tensor.device_type(), tensor.device_index()),
      tensor.shape_dynamism(),
      std::move(deleter));
#else // USE_ATEN_LIB
      tensor.device(),
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
      std::move(deleter));
#endif // USE_ATEN_LIB
}

/**
 * Convenience overload identical to make_tensor_ptr(*tensor_ptr, ...).
 * Keeps the original TensorPtr alive until the returned TensorPtr is destroyed.
 *
 * This overload always aliases — it never copies. To copy a tensor's data to
 * a device, use `clone_tensor_ptr` with a device target, or
 * `tensor.to(device)` in a USE_ATEN_LIB build.
 *
 * @param tensor_ptr The source tensor pointer to alias.
 * @param sizes Optional sizes override.
 * @param dim_order Optional dimension order override.
 * @param strides Optional strides override.
 * @return A TensorPtr aliasing the same storage with requested metadata.
 */
inline TensorPtr make_tensor_ptr(
    const TensorPtr& tensor_ptr,
    std::vector<executorch::aten::SizesType> sizes = {},
    std::vector<executorch::aten::DimOrderType> dim_order = {},
    std::vector<executorch::aten::StridesType> strides = {}) {
  return make_tensor_ptr(
      *tensor_ptr,
      std::move(sizes),
      std::move(dim_order),
      std::move(strides),
      [tensor_ptr](void*) {});
}

/**
 * Creates a TensorPtr that manages a new Tensor with the same properties as the
 * given Tensor, but with a copy of the data owned by the returned TensorPtr.
 *
 * `target` says where the copy lives. The overload without it copies to the
 * source tensor's own device. One end of every copy has to be CPU: an
 * accelerator source with an accelerator target is not supported, not even
 * when both name the same device. So a clone that names no target copies a CPU
 * source and fails for an accelerator source; route that case through CPU. An
 * accelerator target keeps its device index, while a CPU result is plain CPU
 * whatever index was asked for or carried by the source.
 *
 * Between CPU and an accelerator the copy goes through the DeviceAllocator
 * registered for that accelerator, and a device-backed result frees its memory
 * through the same allocator when destroyed. That path is compiled out of
 * USE_ATEN_LIB builds, so both ends have to be CPU there; move data with
 * `tensor.to(device)` instead.
 *
 * A CPU-to-CPU clone of a Tensor whose data is null returns a TensorPtr with
 * null data. A copy that touches an accelerator needs real data and fails
 * without it.
 *
 * The clone keeps the source dtype. To change dtype, use `convert_tensor_ptr`.
 *
 * @param tensor The Tensor to clone.
 * @param target The device the copy should live on.
 * @return A new TensorPtr owning a copy of the data on `target`.
 */
TensorPtr clone_tensor_ptr(
    const executorch::aten::Tensor& tensor,
    executorch::aten::Device target);

/**
 * Overload that copies to the source tensor's own device.
 *
 * @param tensor The Tensor to clone.
 * @return A new TensorPtr owning a copy of the data.
 */
inline TensorPtr clone_tensor_ptr(const executorch::aten::Tensor& tensor) {
  return clone_tensor_ptr(tensor, tensor.device());
}

/**
 * Convenience overload identical to clone_tensor_ptr(*tensor, target).
 *
 * @param tensor The TensorPtr to clone.
 * @param target The device the copy should live on.
 * @return A new TensorPtr owning a copy of the data on `target`.
 */
inline TensorPtr clone_tensor_ptr(
    const TensorPtr& tensor,
    executorch::aten::Device target) {
  return clone_tensor_ptr(*tensor, target);
}

/**
 * Convenience overload identical to clone_tensor_ptr(*tensor).
 *
 * @param tensor The TensorPtr to clone.
 * @return A new TensorPtr owning a copy of the data.
 */
inline TensorPtr clone_tensor_ptr(const TensorPtr& tensor) {
  return clone_tensor_ptr(*tensor);
}

/**
 * Creates a TensorPtr that manages a new Tensor holding the given Tensor's data
 * cast to `type`.
 *
 * Both the source and the result are CPU tensors: move the data to CPU first if
 * it lives on an accelerator. The cast covers Bool, the signed and unsigned
 * integer types, Half, BFloat16, Float and Double; any other type on either
 * side, complex and quantized among them, aborts. The cast also has to be one
 * `canCast` allows, so Float to Int and Int to Bool abort as well.
 * Two requests skip the cast: asking for the type the tensor already has is a
 * plain clone, and a Tensor whose data is null converts to a TensorPtr with
 * null data of the requested type.
 *
 * @param tensor The Tensor to convert.
 * @param type The data type for the new tensor. The data is cast from the
 * source tensor's type.
 * @return A new TensorPtr that manages a Tensor with the specified type and
 * cast data.
 */
TensorPtr convert_tensor_ptr(
    const executorch::aten::Tensor& tensor,
    executorch::aten::ScalarType type);

/**
 * Convenience overload identical to convert_tensor_ptr(*tensor, type).
 *
 * @param tensor The TensorPtr to convert.
 * @param type The data type for the new tensor. The data is cast from the
 * source tensor's type.
 * @return A new TensorPtr that manages a Tensor with the specified type and
 * cast data.
 */
inline TensorPtr convert_tensor_ptr(
    const TensorPtr& tensor,
    executorch::aten::ScalarType type) {
  return convert_tensor_ptr(*tensor, type);
}

/**
 * DEPRECATED: a clone keeps the dtype. Use `convert_tensor_ptr(tensor, type)`
 * to cast instead. May be removed in 1.7.0 or later.
 */
ET_DEPRECATED TensorPtr clone_tensor_ptr(
    const executorch::aten::Tensor& tensor,
    executorch::aten::ScalarType type);

/**
 * DEPRECATED: a clone keeps the dtype. Use `convert_tensor_ptr(tensor, type)`
 * to cast instead. May be removed in 1.7.0 or later.
 */
ET_DEPRECATED inline TensorPtr clone_tensor_ptr(
    const TensorPtr& tensor,
    executorch::aten::ScalarType type) {
  return convert_tensor_ptr(*tensor, type);
}

#ifndef USE_ATEN_LIB
/**
 * DEPRECATED: use `clone_tensor_ptr(tensor, target)` instead. The replacement
 * is also more permissive: a CPU source with a CPU target used to abort here
 * and is a plain clone there. May be removed in 1.7.0 or later.
 */
ET_DEPRECATED TensorPtr
clone_tensor_ptr_to(const TensorPtr& tensor, executorch::aten::Device target);
#endif // USE_ATEN_LIB

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

C10_DIAGNOSTIC_POP()
