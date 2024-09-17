/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_impl_ptr.h>

#include <algorithm>
#include <numeric>

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>

namespace executorch {
namespace extension {
namespace {
#ifndef USE_ATEN_LIB
// No-op deleter that does nothing when called.
static void noop_deleter(void*) {}

/**
 * Custom deleter for TensorImplPtr that ensures the memory associated with
 * dynamic metadata (sizes, dim_order, and strides) is properly managed when the
 * TensorImpl is destroyed.
 *
 * Since TensorImpl does not own the metadata arrays (sizes, dim_order,
 * strides), this deleter is responsible for releasing that memory when the
 * TensorImpl is destroyed.
 */
struct TensorImplPtrDeleter final {
  // A custom deleter of the std::shared_ptr is required to be copyable until
  // C++20, so any data it holds must be copyable too. Hence, we use shared_ptr
  // to hold the data and metadata to avoid unnecessary copies.
  std::shared_ptr<void> data;
  std::shared_ptr<std::vector<exec_aten::SizesType>> sizes;
  std::shared_ptr<std::vector<exec_aten::DimOrderType>> dim_order;
  std::shared_ptr<std::vector<exec_aten::StridesType>> strides;

  void operator()(exec_aten::TensorImpl* pointer) {
    // Release all resources immediately since the data held by the
    // TensorImplPtrDeleter is tied to the managed object, not the smart pointer
    // itself. We need to free this memory when the object is destroyed, not
    // when the smart pointer (and deleter) are eventually destroyed or reset.
    data.reset();
    sizes.reset();
    dim_order.reset();
    strides.reset();
    delete pointer;
  }
};
#endif // USE_ATEN_LIB
} // namespace

TensorImplPtr make_tensor_impl_ptr(
    std::vector<exec_aten::SizesType> sizes,
    void* data,
    std::vector<exec_aten::DimOrderType> dim_order,
    std::vector<exec_aten::StridesType> strides,
    exec_aten::ScalarType type,
    exec_aten::TensorShapeDynamism dynamism,
    std::function<void(void*)> deleter) {
  const auto dim = sizes.size();
  ET_CHECK_MSG(
      dim_order.empty() || dim_order.size() == dim,
      "dim_order size must match sizes or be empty.");
  ET_CHECK_MSG(
      strides.empty() || strides.size() == dim,
      "strides size must match sizes or be empty.");

  if (dim_order.empty()) {
    dim_order.resize(dim);
    std::iota(dim_order.begin(), dim_order.end(), 0);
    if (!strides.empty()) {
      std::sort(dim_order.begin(), dim_order.end(), [&](size_t a, size_t b) {
        return strides[a] > strides[b];
      });
    }
  }
  std::vector<exec_aten::StridesType> computed_strides(dim);
  auto error = runtime::dim_order_to_stride(
      sizes.data(), dim_order.data(), dim, computed_strides.data());
  ET_CHECK_MSG(error == runtime::Error::Ok, "Failed to compute strides.");

  if (!strides.empty()) {
    ET_CHECK_MSG(computed_strides == strides, "Invalid strides provided.");
  } else {
    strides = std::move(computed_strides);
  }
#ifndef USE_ATEN_LIB
  auto tensor_impl = std::make_unique<exec_aten::TensorImpl>(
      type,
      dim,
      sizes.data(),
      data,
      dim_order.data(),
      strides.data(),
      dim > 0 ? dynamism : exec_aten::TensorShapeDynamism::STATIC);
  return TensorImplPtr(
      tensor_impl.release(),
      TensorImplPtrDeleter{
          std::shared_ptr<void>(
              data, deleter ? std::move(deleter) : noop_deleter),
          std::make_shared<std::vector<exec_aten::SizesType>>(std::move(sizes)),
          std::make_shared<std::vector<exec_aten::DimOrderType>>(
              std::move(dim_order)),
          std::make_shared<std::vector<exec_aten::StridesType>>(
              std::move(strides))});
#else
  auto options = c10::TensorOptions()
                     .dtype(c10::scalarTypeToTypeMeta(type))
                     .device(c10::kCPU);
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(
          sizes, strides, options.dtype().itemsize()),
      c10::InefficientStdFunctionContext::makeDataPtr(
          data, std::move(deleter), options.device()),
      nullptr,
      false);
  auto tensor_impl = c10::make_intrusive<at::TensorImpl>(
      std::move(storage),
      c10::DispatchKeySet(c10::DispatchKey::CPU),
      options.dtype());
  tensor_impl->set_sizes_and_strides(sizes, strides);
  return tensor_impl;
#endif // USE_ATEN_LIB
}

TensorImplPtr make_tensor_impl_ptr(
    std::vector<exec_aten::SizesType> sizes,
    std::vector<uint8_t> data,
    std::vector<exec_aten::DimOrderType> dim_order,
    std::vector<exec_aten::StridesType> strides,
    exec_aten::ScalarType type,
    exec_aten::TensorShapeDynamism dynamism) {
  ET_CHECK_MSG(
      data.size() >= exec_aten::compute_numel(sizes.data(), sizes.size()) *
              exec_aten::elementSize(type),
      "Data size is smaller than required by sizes and scalar type.");
  auto raw_data_ptr = data.data();
  auto data_ptr = std::make_shared<std::vector<uint8_t>>(std::move(data));
  return make_tensor_impl_ptr(
      std::move(sizes),
      raw_data_ptr,
      std::move(dim_order),
      std::move(strides),
      type,
      dynamism,
      [data_ptr = std::move(data_ptr)](void*) {});
}

} // namespace extension
} // namespace executorch
