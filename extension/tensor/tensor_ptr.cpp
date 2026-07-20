/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr.h>

#include <numeric>

#include <c10/util/safe_numerics.h>

#ifndef USE_ATEN_LIB
#include <executorch/runtime/core/device_allocator.h>
#endif // USE_ATEN_LIB
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
 *
 * For device tensors, the data pointer points to device memory; the deleter
 * is responsible for freeing it through the appropriate DeviceAllocator.
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

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;

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
    std::function<void(void*)> deleter,
    executorch::aten::Device device) {
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
      dim > 0 ? dynamism : executorch::aten::TensorShapeDynamism::STATIC,
      device.type(),
      device.index());
  auto storage = std::make_shared<Storage>(
      std::move(tensor_impl),
      std::move(sizes),
      std::move(dim_order),
      std::move(strides),
      std::move(deleter));
  const auto raw_tensor_ptr = &storage->tensor;
  return std::shared_ptr<executorch::aten::Tensor>(
      std::move(storage), raw_tensor_ptr);
#else
  auto options = c10::TensorOptions()
                     .dtype(c10::scalarTypeToTypeMeta(type))
                     .device(device);
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
      c10::DispatchKeySet(options.computeDispatchKey()),
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
  auto numel_result = executorch::aten::safe_numel(sizes.data(), sizes.size());
  ET_CHECK_MSG(
      numel_result.ok(),
      "safe_numel failed: %d",
      static_cast<int>(numel_result.error()));
  const ssize_t numel = numel_result.get();
  size_t nbytes;
  ET_CHECK_MSG(
      !c10::mul_overflows(
          static_cast<size_t>(numel),
          executorch::aten::elementSize(type),
          &nbytes),
      "Overflow computing nbytes: numel=%zd element_size=%zu",
      numel,
      executorch::aten::elementSize(type));
  ET_CHECK_MSG(
      data.size() == nbytes,
      "Data size (%zu) does not match tensor size (%zu).",
      data.size(),
      nbytes);
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

TensorPtr clone_tensor_ptr(
    const executorch::aten::Tensor& tensor,
    executorch::aten::ScalarType type) {
#ifndef USE_ATEN_LIB
  ET_CHECK_MSG(
      tensor.device_type() == runtime::etensor::DeviceType::CPU,
      "clone_tensor_ptr only supports CPU tensors; use clone_tensor_ptr_to with a CPU target first.");
#else // USE_ATEN_LIB
  ET_CHECK_MSG(
      tensor.is_cpu(),
      "clone_tensor_ptr only supports CPU tensors; move it to CPU first (e.g. tensor.to(torch::kCPU)).");
#endif // USE_ATEN_LIB
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
  const auto* tensor_data = tensor.const_data_ptr();
  if (!tensor_data) {
    return make_tensor_ptr(
        std::move(sizes),
        nullptr,
        std::move(dim_order),
        std::move(strides),
        type,
        dynamism);
  }
  const auto tensor_type = tensor.scalar_type();
  if (tensor_type == type) {
    return make_tensor_ptr(
        std::move(sizes),
        std::vector<uint8_t>(
            (uint8_t*)tensor_data, (uint8_t*)tensor_data + tensor.nbytes()),
        std::move(dim_order),
        std::move(strides),
        tensor_type,
        dynamism);
  }
  ET_CHECK_MSG(
      runtime::canCast(tensor_type, type),
      "Cannot cast tensor type to desired type.");
  const auto tensor_numel = static_cast<size_t>(tensor.numel());
  size_t clone_nbytes;
  ET_CHECK_MSG(
      !c10::mul_overflows(tensor_numel, aten::elementSize(type), &clone_nbytes),
      "Overflow computing clone nbytes: numel=%zu element_size=%zu",
      tensor_numel,
      aten::elementSize(type));
  std::vector<uint8_t> data(clone_nbytes);

  // Create a minimal context for error handling in ET_SWITCH
  struct {
    [[noreturn]] void fail(torch::executor::Error /* error */) {
      ET_CHECK_MSG(false, "Unsupported dtype in clone_tensor_ptr");
    }
  } ctx;

  ET_SWITCH_REALHBBF16_AND_UINT_TYPES(
      tensor_type, ctx, "clone_tensor_ptr_cast_from", CTYPE_FROM, [&] {
        const CTYPE_FROM* tensor_data_ptr =
            static_cast<const CTYPE_FROM*>(tensor_data);
        ET_SWITCH_REALHBBF16_AND_UINT_TYPES(
            type, ctx, "clone_tensor_ptr_cast_to", CTYPE_TO, [&] {
              CTYPE_TO* data_ptr = reinterpret_cast<CTYPE_TO*>(data.data());
              std::transform(
                  tensor_data_ptr,
                  tensor_data_ptr + tensor_numel,
                  data_ptr,
                  [](const CTYPE_FROM& val) {
                    return static_cast<CTYPE_TO>(val);
                  });
            });
      });
  return make_tensor_ptr(
      std::move(sizes),
      std::move(data),
      std::move(dim_order),
      std::move(strides),
      type,
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

// ---- Device tensor helper ----
//
// This helper relies on the ExecuTorch DeviceAllocator and the portable tensor
// metadata APIs (dim_order, shape_dynamism, device), which have no equivalent
// in USE_ATEN_LIB builds, so it is compiled out there.

#ifndef USE_ATEN_LIB

TensorPtr clone_tensor_ptr_to(
    const TensorPtr& tensor,
    executorch::aten::Device target) {
  const auto source = tensor->device();
  ET_CHECK_MSG(
      !(source.is_cpu() && target.is_cpu()),
      "clone_tensor_ptr_to does not copy CPU-to-CPU; use clone_tensor_ptr.");
  ET_CHECK_MSG(
      source.is_cpu() || target.is_cpu(),
      "Device-to-device copy is not supported; route through CPU.");

  const auto nbytes = tensor->nbytes();
  const auto* src_data = tensor->const_data_ptr();
  ET_CHECK_MSG(src_data != nullptr, "Source tensor has no data.");

  // Whichever end is not CPU provides the allocator.
  const auto device = target.is_cpu() ? source : target;
  auto* allocator = runtime::get_device_allocator(device.type());
  ET_CHECK_MSG(
      allocator != nullptr,
      "No device allocator registered for device type %d",
      static_cast<int>(device.type()));

  std::vector<executorch::aten::SizesType> sizes(
      tensor->sizes().begin(), tensor->sizes().end());
  std::vector<executorch::aten::DimOrderType> dim_order(
      tensor->dim_order().begin(), tensor->dim_order().end());
  std::vector<executorch::aten::StridesType> strides(
      tensor->strides().begin(), tensor->strides().end());

  if (target.is_cpu()) {
    std::vector<uint8_t> cpu_data(nbytes);
    auto err = allocator->copy_device_to_host(
        cpu_data.data(), src_data, nbytes, source.index());
    ET_CHECK_MSG(
        err == runtime::Error::Ok,
        "Device-to-host copy failed: error %d",
        static_cast<int>(err));
    return make_tensor_ptr(
        std::move(sizes),
        std::move(cpu_data),
        std::move(dim_order),
        std::move(strides),
        tensor->scalar_type(),
        tensor->shape_dynamism());
  }

  auto result = allocator->allocate(nbytes, target.index());
  ET_CHECK_MSG(
      result.ok(),
      "Failed to allocate device memory: error %d",
      static_cast<int>(result.error()));
  void* device_data = result.get();
  auto err = allocator->copy_host_to_device(
      device_data, src_data, nbytes, target.index());
  ET_CHECK_MSG(
      err == runtime::Error::Ok,
      "Host-to-device copy failed: error %d",
      static_cast<int>(err));
  return make_tensor_ptr(
      std::move(sizes),
      device_data,
      std::move(dim_order),
      std::move(strides),
      tensor->scalar_type(),
      tensor->shape_dynamism(),
      [allocator, target](void* ptr) {
        allocator->deallocate(ptr, target.index());
      },
      target);
}

#endif // USE_ATEN_LIB

} // namespace extension
} // namespace executorch
