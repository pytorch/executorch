/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr.h>

#include <numeric>

#include <executorch/runtime/core/device_allocator.h>
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
    std::function<void(void*)> deleter,
    runtime::etensor::DeviceType device_type,
    runtime::etensor::DeviceIndex device_index) {
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

  TensorPtr cpu_tensor;
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
  const auto raw_tensor_ptr = &storage->tensor;
  cpu_tensor = std::shared_ptr<executorch::aten::Tensor>(
      std::move(storage), raw_tensor_ptr);
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
  cpu_tensor =
      std::make_shared<executorch::aten::Tensor>(std::move(tensor_impl));
#endif // USE_ATEN_LIB
  if (device_type != runtime::etensor::DeviceType::CPU) {
    return clone_tensor_ptr_to_device(cpu_tensor, device_type, device_index);
  }
  return cpu_tensor;
}

TensorPtr make_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<uint8_t> data,
    std::vector<executorch::aten::DimOrderType> dim_order,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type,
    executorch::aten::TensorShapeDynamism dynamism,
    runtime::etensor::DeviceType device_type,
    runtime::etensor::DeviceIndex device_index) {
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
      [data = std::move(data)](void*) {},
      device_type,
      device_index);
}

TensorPtr clone_tensor_ptr(
    const executorch::aten::Tensor& tensor,
    executorch::aten::ScalarType type) {
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
  std::vector<uint8_t> data(tensor_numel * aten::elementSize(type));

  // Create a minimal context for error handling in ET_SWITCH
  struct {
    [[noreturn]] void fail(torch::executor::Error /* error */) {
      ET_CHECK_MSG(false, "Unsupported dtype in clone_tensor_ptr");
    }
  } ctx;

  ET_SWITCH_REALHBBF16_AND_UINT_TYPES(
      tensor_type, ctx, "clone_tensor_ptr_from", CTYPE_FROM, [&] {
        const CTYPE_FROM* tensor_data_ptr =
            static_cast<const CTYPE_FROM*>(tensor_data);
        ET_SWITCH_REALHBBF16_AND_UINT_TYPES(
            type, ctx, "clone_tensor_ptr_to", CTYPE_TO, [&] {
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

// ---- Device tensor helpers ----

namespace {

#ifndef USE_ATEN_LIB
struct DeviceStorage final {
  executorch::aten::TensorImpl tensor_impl;
  executorch::aten::Tensor tensor;
  std::vector<executorch::aten::SizesType> sizes;
  std::vector<executorch::aten::DimOrderType> dim_order;
  std::vector<executorch::aten::StridesType> strides;
  std::function<void(void*)> deleter;

  DeviceStorage(
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

  ~DeviceStorage() {
    if (deleter) {
      deleter(tensor_impl.mutable_data());
    }
  }
};
#endif // USE_ATEN_LIB

TensorPtr make_tensor_ptr_with_device(
    std::vector<executorch::aten::SizesType> sizes,
    void* data,
    executorch::aten::ScalarType type,
    runtime::etensor::DeviceType device_type,
    runtime::etensor::DeviceIndex device_index,
    std::function<void(void*)> deleter) {
  const auto dim = sizes.size();
  std::vector<executorch::aten::DimOrderType> dim_order(dim);
  std::iota(dim_order.begin(), dim_order.end(), 0);

  std::vector<executorch::aten::StridesType> strides(dim);
  if (dim > 0) {
    auto error = runtime::dim_order_to_stride(
        sizes.data(), dim_order.data(), dim, strides.data());
    ET_CHECK_MSG(error == runtime::Error::Ok, "Failed to compute strides.");
  }

#ifndef USE_ATEN_LIB
  executorch::aten::TensorImpl tensor_impl(
      type,
      dim,
      sizes.data(),
      data,
      dim_order.data(),
      strides.data(),
      dim > 0 ? executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND
              : executorch::aten::TensorShapeDynamism::STATIC,
      device_type,
      device_index);
  auto storage = std::make_shared<DeviceStorage>(
      std::move(tensor_impl),
      std::move(sizes),
      std::move(dim_order),
      std::move(strides),
      std::move(deleter));
  const auto tensor_ptr = &storage->tensor;
  return std::shared_ptr<executorch::aten::Tensor>(
      std::move(storage), tensor_ptr);
#else
  (void)device_type;
  (void)device_index;
  return make_tensor_ptr(
      std::move(sizes),
      data,
      type,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
      std::move(deleter));
#endif // USE_ATEN_LIB
}

} // namespace

TensorPtr clone_tensor_ptr_to_device(
    const TensorPtr& cpu_tensor,
    runtime::etensor::DeviceType device_type,
    runtime::etensor::DeviceIndex device_index) {
  ET_CHECK_MSG(
      device_type != runtime::etensor::DeviceType::CPU,
      "Target device must not be CPU; use clone_tensor_ptr for CPU-to-CPU copies.");

  auto* allocator = runtime::get_device_allocator(device_type);
  ET_CHECK_MSG(
      allocator != nullptr,
      "No device allocator registered for device type %d",
      static_cast<int>(device_type));

  const auto nbytes = cpu_tensor->nbytes();
  const auto* cpu_data = cpu_tensor->const_data_ptr();
  ET_CHECK_MSG(cpu_data != nullptr, "Source tensor has no data.");

  auto result = allocator->allocate(nbytes, device_index);
  ET_CHECK_MSG(result.ok(), "Failed to allocate device memory.");
  void* device_data = result.get();

  auto err = allocator->copy_host_to_device(
      device_data, cpu_data, nbytes, device_index);
  ET_CHECK_MSG(err == runtime::Error::Ok, "Host-to-device copy failed.");

  std::vector<executorch::aten::SizesType> sizes(
      cpu_tensor->sizes().begin(), cpu_tensor->sizes().end());

  return make_tensor_ptr_with_device(
      std::move(sizes),
      device_data,
      cpu_tensor->scalar_type(),
      device_type,
      device_index,
      [allocator, device_index](void* ptr) {
        allocator->deallocate(ptr, device_index);
      });
}

TensorPtr clone_tensor_ptr_to_cpu(const TensorPtr& device_tensor) {
  const auto nbytes = device_tensor->nbytes();
  const auto* device_data = device_tensor->const_data_ptr();
  ET_CHECK_MSG(device_data != nullptr, "Source device tensor has no data.");

#ifndef USE_ATEN_LIB
  const auto device_type = device_tensor->unsafeGetTensorImpl()->device_type();
  const auto device_index =
      device_tensor->unsafeGetTensorImpl()->device_index();
#else
  const auto& aten_device = device_tensor->device();
  ET_CHECK_MSG(!aten_device.is_cpu(), "Source tensor is already on CPU.");
  auto device_type = runtime::etensor::DeviceType::CPU;
  if (aten_device.is_cuda()) {
    device_type = runtime::etensor::DeviceType::CUDA;
  }
  const auto device_index =
      static_cast<runtime::etensor::DeviceIndex>(aten_device.index());
#endif

  ET_CHECK_MSG(
      device_type != runtime::etensor::DeviceType::CPU,
      "Source tensor is already on CPU.");

  auto* allocator = runtime::get_device_allocator(device_type);
  ET_CHECK_MSG(
      allocator != nullptr,
      "No device allocator registered for device type %d",
      static_cast<int>(device_type));

  std::vector<uint8_t> cpu_data(nbytes);

  auto err = allocator->copy_device_to_host(
      cpu_data.data(), device_data, nbytes, device_index);
  ET_CHECK_MSG(err == runtime::Error::Ok, "Device-to-host copy failed.");

  std::vector<executorch::aten::SizesType> sizes(
      device_tensor->sizes().begin(), device_tensor->sizes().end());

  return make_tensor_ptr(
      std::move(sizes),
      std::move(cpu_data),
      {},
      {},
      device_tensor->scalar_type());
}

} // namespace extension
} // namespace executorch
