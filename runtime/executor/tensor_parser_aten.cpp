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
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/schema/program_generated.h>

#include <ATen/ATen.h> // @donotremove @manual=//caffe2/aten:ATen-core

namespace executorch {
namespace runtime {
namespace deserialization {

namespace {

void deleteNothing(void*);
void deleteNothing(void*) {}

} // namespace

Result<at::Tensor> parseTensor(
    const Program* program,
    MemoryManager* memory_manager,
    const executorch_flatbuffer::Tensor* s_tensor) {
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
  auto options = at::CPU(type).options();

  ET_CHECK_OR_RETURN_ERROR(
      s_tensor->sizes() != nullptr, InvalidProgram, "Missing sizes field");
  size_t ndim = s_tensor->sizes()->size();

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
    // Provide fully dynamic tensors with an allocator so they can be resized
    // within aten kernels.
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
        s_tensor, program, tensor.nbytes(), memory_manager->planned_memory());
    if (!data_ptr.ok()) {
      ET_LOG(
          Error,
          "getTensorDataPtr() failed: 0x%" PRIx32,
          static_cast<uint32_t>(data_ptr.error()));
      return data_ptr.error();
    }
    tensor.unsafeGetTensorImpl()->unsafe_storage().set_data_ptr(
        at::DataPtr(data_ptr.get(), c10::DeviceType::CPU));
  }

  return tensor;
}

} // namespace deserialization
} // namespace runtime
} // namespace executorch
