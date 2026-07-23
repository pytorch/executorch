/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/tensor_parser.h>

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/schema/program_generated.h>

#include <ATen/ATen.h> // @donotremove @manual=//caffe2/aten:ATen-core

namespace executorch {
// This file is only used in ATen mode, so we use the runtime_aten namespace.
namespace runtime {
namespace aten {
namespace deserialization {

namespace {

void deleteNothing(void*);
void deleteNothing(void*) {}

} // namespace

Result<at::Tensor> parseTensor(
    const Program* program,
    MemoryManager* memory_manager,
    const executorch_flatbuffer::Tensor* s_tensor,
    const NamedDataMap* named_data_map,
    Span<NamedData> external_constants) {
  EXECUTORCH_SCOPE_PROF("TensorParser::parseTensor");

  ET_CHECK_OR_RETURN_ERROR(
      s_tensor->storage_offset() == 0,
      NotSupported,
      "Non-zero storage offset %" PRId32 " not supported",
      s_tensor->storage_offset());

  // get metadata
  at::ScalarType type = static_cast<at::ScalarType>(s_tensor->scalar_type());
  ET_CHECK_OR_RETURN_ERROR(
      isValid(type),
      InvalidProgram,
      "Invalid ScalarType %" PRId8,
      static_cast<int8_t>(type));

  // Defaults to CPU when extra_tensor_info is absent (older PTE files). A
  // device-delegate planned buffer must be tagged with its real device or the
  // runtime treats device memory as host memory.
  c10::DeviceType device_type = c10::DeviceType::CPU;
  c10::DeviceIndex device_index = 0;
  if (s_tensor->extra_tensor_info() != nullptr) {
    // Untrusted byte from the PTE; validate before the cast so a bogus value
    // cannot reach c10::Device as garbage.
    const auto raw_device_type = s_tensor->extra_tensor_info()->device_type();
    ET_CHECK_OR_RETURN_ERROR(
        raw_device_type == executorch_flatbuffer::DeviceType::CPU ||
            raw_device_type == executorch_flatbuffer::DeviceType::CUDA,
        InvalidProgram,
        "Invalid DeviceType %" PRId8,
        static_cast<int8_t>(raw_device_type));
    device_type = raw_device_type == executorch_flatbuffer::DeviceType::CUDA
        ? c10::DeviceType::CUDA
        : c10::DeviceType::CPU;
    device_index = static_cast<c10::DeviceIndex>(
        s_tensor->extra_tensor_info()->device_index());
    // Reject a negative accelerator index from the untrusted PTE; -1
    // (any/current device) is not a valid serialized placement and would later
    // confuse device matching.
    ET_CHECK_OR_RETURN_ERROR(
        device_type == c10::DeviceType::CPU || device_index >= 0,
        InvalidProgram,
        "Invalid device_index %" PRId8,
        static_cast<int8_t>(device_index));
  }
  // CPU stays unindexed: an explicit cpu:0 would mismatch the graph's default
  // cpu tensors and trip ATen's same-device check. Only accelerators carry an
  // index.
  const c10::Device device = device_type == c10::DeviceType::CPU
      ? c10::Device(device_type)
      : c10::Device(device_type, device_index);
  // Sized with null data to compute nbytes; real device is applied below.
  auto options = at::CPU(type).options();

  ET_CHECK_OR_RETURN_ERROR(
      s_tensor->sizes() != nullptr, InvalidProgram, "Missing sizes field");
  size_t ndim = s_tensor->sizes()->size();

  ET_CHECK_OR_RETURN_ERROR(
      ndim <= kTensorDimensionLimit,
      InvalidProgram,
      "Tensor rank too large %" ET_PRIsize_t " > %zu",
      ndim,
      kTensorDimensionLimit)

  ET_CHECK_OR_RETURN_ERROR(
      s_tensor->dim_order() != nullptr,
      InvalidProgram,
      "Missing dim_order field");
  ET_CHECK_OR_RETURN_ERROR(
      s_tensor->dim_order()->size() == ndim,
      InvalidProgram,
      "dim_order size %" PRIu32 " != ndim %zu",
      s_tensor->dim_order()->size(),
      ndim);

  // convert int32 in serialization to int64 for aten
  std::vector<int64_t> sizes(
      s_tensor->sizes()->begin(), s_tensor->sizes()->end());
  std::vector<int64_t> strides(ndim);
  auto status = dim_order_to_stride(
      s_tensor->sizes()->data(),
      s_tensor->dim_order()->data(),
      ndim,
      strides.data());
  ET_CHECK_OR_RETURN_ERROR(
      status == Error::Ok,
      Internal,
      "dim_order_to_stride returned invalid status");

  // Create a tensor without data first so we can find its expected size before
  // getting its memory.
  at::Tensor tensor = at::from_blob(
      /*data=*/nullptr,
      sizes,
      strides,
      /*storage_offset=*/0,
      deleteNothing,
      options);

  if (s_tensor->shape_dynamism() ==
      executorch_flatbuffer::TensorShapeDynamism::DYNAMIC_UNBOUND) {
    // Fully dynamic tensors get an allocator so aten kernels can resize them.
    // Device-delegate planned buffers are statically bounded, so a device
    // tensor never reaches this CPU-tagged path.
    auto impl = tensor.unsafeGetTensorImpl();
    at::StorageImpl* storage = impl->unsafe_storage().unsafeGetStorageImpl();
    storage->set_allocator(at::getCPUAllocator());
    storage->set_resizable(true);
    storage->set_nbytes(0);
    impl->set_sizes_contiguous(0);
    // Leave the data as nullptr since it will be reallocated.
  } else {
    // Now that we know how big the tensor is, find and assign its memory.
    Result<void*> data_ptr = getTensorDataPtr(
        s_tensor,
        program,
        tensor.nbytes(),
        memory_manager->planned_memory(),
        named_data_map,
        external_constants);
    if (!data_ptr.ok()) {
      ET_LOG(
          Error,
          "getTensorDataPtr() failed: 0x%" PRIx32,
          static_cast<uint32_t>(data_ptr.error()));
      return data_ptr.error();
    }
    // Rebuild so storage DataPtr, TensorImpl device, and dispatch key agree.
    // target_device makes from_blob skip getDeviceFromPtr, so the same path
    // works for a real pointer and for a null runtime-bound one.
    tensor = at::from_blob(
        data_ptr.get(),
        sizes,
        strides,
        /*storage_offset=*/0,
        deleteNothing,
        at::TensorOptions().dtype(type).device(device),
        /*target_device=*/device);
  }

  return tensor;
}

} // namespace deserialization
} // namespace aten
} // namespace runtime
} // namespace executorch
