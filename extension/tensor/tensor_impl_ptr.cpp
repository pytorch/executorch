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
  exec_aten::TensorImpl tensor_impl(
      type,
      dim,
      sizes.data(),
      data,
      dim_order.data(),
      strides.data(),
      dim > 0 ? dynamism : exec_aten::TensorShapeDynamism::STATIC);
  return TensorImplPtr(
      std::move(tensor_impl),
      std::move(sizes),
      std::move(dim_order),
      std::move(strides),
      std::move(deleter));
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
