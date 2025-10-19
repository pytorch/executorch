/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr.h>

#include <numeric>

#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace executorch {
namespace extension {
namespace {
#ifndef USE_ATEN_LIB
/**
 * A structure that consolidates the metadata (sizes, dim_order, strides) and
 * the data buffer associated with a Tensor. Since Tensor does not own
 * the memory for these metadata arrays or the data itself, this structure
 * ensures that they are managed together and have the same lifetime as the
 * Tensor. When the Tensor is destroyed, the Storage structure ensures
 * proper cleanup of the associated metadata and data if needed.
 */
struct Storage final {
  executorch::aten::TensorImpl tensor_impl;
  executorch::aten::Tensor tensor;
  std::vector<executorch::aten::SizesType> sizes;
  std::vector<executorch::aten::DimOrderType> dim_order;
  std::vector<executorch::aten::StridesType> strides;
  std::function<void(void*)> deleter;

  Storage(
      executorch::aten::TensorImpl&& tensor_impl,
      std::vector<executorch::aten::SizesType>&& sizes,
      std::vector<executorch::aten::DimOrderType>&& dim_order,
      std::vector<executorch::aten::StridesType>&& strides,
      std::function<void(void*)>&& deleter)
      : tensor_impl(std::move(tensor_impl)),
        tensor(&this->tensor_impl),
        sizes(std::move(sizes)),
        dim_order(std::move(dim_order)),
        strides(std::move(strides)),
        deleter(std::move(deleter)) {}

  ~Storage() {
    if (deleter) {
      deleter(tensor_impl.mutable_data());
    }
  }
};
#endif // USE_ATEN_LIB
} // namespace

TensorPtr make_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    void* data,
    std::vector<executorch::aten::DimOrderType> dim_order,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type,
    executorch::aten::TensorShapeDynamism dynamism,
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
  std::vector<executorch::aten::StridesType> computed_strides(dim);

  auto error = runtime::dim_order_to_stride(
      sizes.data(), dim_order.data(), dim, computed_strides.data());
  ET_CHECK_MSG(error == runtime::Error::Ok, "Failed to compute strides.");

  if (!strides.empty()) {
    for (size_t i = 0; i < dim; i++) {
      ET_CHECK_MSG(
          strides[i] == computed_strides[i] || sizes[i] == 1,
          "invalid strides for dim %zu: %" ET_PRI_SIZES_AND_STRIDES
          "!= %" ET_PRI_SIZES_AND_STRIDES
          " while its size is %" ET_PRI_SIZES_AND_STRIDES " != 1",
          i,
          strides[i],
          computed_strides[i],
          sizes[i]);
    }
  }

  strides = std::move(computed_strides);

#ifndef USE_ATEN_LIB
  executorch::aten::TensorImpl tensor_impl(
      type,
      dim,
      sizes.data(),
      data,
      dim_order.data(),
      strides.data(),
      dim > 0 ? dynamism : executorch::aten::TensorShapeDynamism::STATIC);
  auto storage = std::make_shared<Storage>(
      std::move(tensor_impl),
      std::move(sizes),
      std::move(dim_order),
      std::move(strides),
      std::move(deleter));
  const auto tensor_ptr = &storage->tensor;
  return std::shared_ptr<executorch::aten::Tensor>(
      std::move(storage), tensor_ptr);
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
  auto tensor_impl = c10::make_intrusive<executorch::aten::TensorImpl>(
      std::move(storage),
      c10::DispatchKeySet(c10::DispatchKey::CPU),
      options.dtype());
  tensor_impl->set_sizes_and_strides(sizes, strides);
  return std::make_shared<executorch::aten::Tensor>(std::move(tensor_impl));
#endif // USE_ATEN_LIB
}

TensorPtr make_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<uint8_t> data,
    std::vector<executorch::aten::DimOrderType> dim_order,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type,
    executorch::aten::TensorShapeDynamism dynamism) {
  ET_CHECK_MSG(
      data.size() ==
          executorch::aten::compute_numel(sizes.data(), sizes.size()) *
              executorch::aten::elementSize(type),
      "Data size does not match tensor size.");
  auto data_ptr = data.data();
  return make_tensor_ptr(
      std::move(sizes),
      data_ptr,
      std::move(dim_order),
      std::move(strides),
      type,
      dynamism,
      // Data is moved into the deleter and is destroyed together with Storage.
      [data = std::move(data)](void*) {});
}

TensorPtr clone_tensor_ptr(const executorch::aten::Tensor& tensor) {
  std::vector<executorch::aten::SizesType> sizes(
      tensor.sizes().begin(), tensor.sizes().end());
  std::vector<executorch::aten::DimOrderType> dim_order{
#ifndef USE_ATEN_LIB
      tensor.dim_order().begin(), tensor.dim_order().end()
#endif // USE_ATEN_LIB
  };
  std::vector<executorch::aten::StridesType> strides(
      tensor.strides().begin(), tensor.strides().end());
  auto dynamism = executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND;
#ifndef USE_ATEN_LIB
  dynamism = tensor.shape_dynamism();
#endif // USE_ATEN_LIB
  return tensor.const_data_ptr()
      ? make_tensor_ptr(
            std::move(sizes),
            std::vector<uint8_t>(
                (uint8_t*)tensor.const_data_ptr(),
                (uint8_t*)tensor.const_data_ptr() + tensor.nbytes()),
            std::move(dim_order),
            std::move(strides),
            tensor.scalar_type(),
            dynamism)
      : make_tensor_ptr(
            std::move(sizes),
            nullptr,
            std::move(dim_order),
            std::move(strides),
            tensor.scalar_type(),
            dynamism);
}

runtime::Error resize_tensor_ptr(
    TensorPtr& tensor,
    const std::vector<executorch::aten::SizesType>& sizes) {
  return ET_RUNTIME_NAMESPACE::resize_tensor(
      *tensor,
      executorch::aten::ArrayRef<executorch::aten::SizesType>(
          sizes.data(), sizes.size()));
}

} // namespace extension
} // namespace executorch
