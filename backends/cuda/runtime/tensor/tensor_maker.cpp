/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/tensor/tensor_maker.h>

#include <numeric>

#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace executorch::backends::cuda {

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

TensorPtr make_tensor(
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

  // AOTI backends (like AOTI-CUDA) handle both contiguous and incontiguous
  // tensors, so we skip stride calculation and incontiguous tensor checks.
  // Strides are passed through as-is without validation.

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

} // namespace executorch::backends::cuda
