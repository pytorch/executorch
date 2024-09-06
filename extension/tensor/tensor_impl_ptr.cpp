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
  std::unique_ptr<void, std::function<void(void*)>> data;
  std::vector<exec_aten::SizesType> sizes;
  std::vector<exec_aten::DimOrderType> dim_order;
  std::vector<exec_aten::StridesType> strides;

  void operator()(exec_aten::TensorImpl* pointer) {
    // Release all resources immediately since the data held by the
    // TensorImplDeleter is tied to the managed object, not the smart pointer
    // itself. We need to free this memory when the object is destroyed, not
    // when the smart pointer (and deleter) are eventually destroyed or reset.
    data.reset();
    sizes = {};
    dim_order = {};
    strides = {};
    delete pointer;
  }
};
#endif // USE_ATEN_LIB
} // namespace

TensorImplPtr make_tensor_impl_ptr(
    exec_aten::ScalarType type,
    std::vector<exec_aten::SizesType> sizes,
    void* data,
    std::vector<exec_aten::DimOrderType> dim_order,
    std::vector<exec_aten::StridesType> strides,
    exec_aten::TensorShapeDynamism dynamism,
    std::function<void(void*)> deleter) {
  const auto dim = sizes.size();
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
      dynamism);
  return TensorImplPtr(
      tensor_impl.release(),
      TensorImplPtrDeleter{
          std::unique_ptr<void, std::function<void(void*)>>(
              data, deleter ? std::move(deleter) : noop_deleter),
          std::move(sizes),
          std::move(dim_order),
          std::move(strides)});
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

} // namespace extension
} // namespace executorch
