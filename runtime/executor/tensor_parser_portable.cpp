/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/tensor_parser.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/schema/program_generated.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {
namespace deserialization {

using executorch::runtime::Span;
using ::torch::executor::ScalarType;
using ::torch::executor::Tensor;
using ::torch::executor::TensorImpl;

Result<Tensor> parseTensor(
    const Program* program,
    MemoryManager* memory_manager,
    const executorch_flatbuffer::Tensor* s_tensor,
    const NamedDataMap* named_data_map,
    Span<NamedData> external_constants) {
  EXECUTORCH_SCOPE_PROF("TensorParser::parseTensor");
  auto method_allocator = memory_manager->method_allocator();

  ET_CHECK_OR_RETURN_ERROR(
      s_tensor->storage_offset() == 0,
      NotSupported,
      "Non-zero storage offset %" PRId32 " not supported",
      s_tensor->storage_offset());

  ScalarType scalar_type = static_cast<ScalarType>(s_tensor->scalar_type());
  ET_CHECK_OR_RETURN_ERROR(
      isValid(scalar_type),
      InvalidProgram,
      "Invalid or unsupported ScalarType %" PRId8,
      static_cast<int8_t>(scalar_type));

  TensorShapeDynamism dynamism =
      static_cast<TensorShapeDynamism>(s_tensor->shape_dynamism());
  // TODO(T175194371): Remove this check once fully dynamic shapes are
  // supported.
  ET_CHECK_OR_RETURN_ERROR(
      dynamism != TensorShapeDynamism::DYNAMIC_UNBOUND,
      NotSupported,
      "Fully dynamic tensor shapes not yet supported: T175194371");

  ET_CHECK_OR_RETURN_ERROR(
      s_tensor->sizes() != nullptr, InvalidProgram, "Missing sizes field");
  const auto serialized_sizes = s_tensor->sizes()->data();
  const auto dim = s_tensor->sizes()->size();

  ET_CHECK_OR_RETURN_ERROR(
      dim <= kTensorDimensionLimit,
      InvalidProgram,
      "Tensor rank too large %" PRIu32 " > %zu",
      dim,
      kTensorDimensionLimit)

  ET_CHECK_OR_RETURN_ERROR(
      s_tensor->dim_order() != nullptr,
      InvalidProgram,
      "Missing dim_order field");
  ET_CHECK_OR_RETURN_ERROR(
      s_tensor->dim_order()->size() == dim,
      InvalidProgram,
      "dim_order size %" PRIu32 " != dim %" PRIu32,
      s_tensor->dim_order()->size(),
      dim);
  const auto serialized_dim_order = s_tensor->dim_order()->data();

  executorch::aten::SizesType* sizes = nullptr;
  executorch::aten::DimOrderType* dim_order = nullptr;
  // For dynamic shape tensors, allocate local buffers to allow mutable sizes
  // and strides
  if (dynamism != TensorShapeDynamism::STATIC) {
    // copy sizes and dim order out of flatbuffer
    // kimishpate: I think dim order can remain immutable and point to fb
    // memory, unless we plan to implement in-place permute
    executorch::aten::SizesType* sizes_buf =
        method_allocator->allocateList<executorch::aten::SizesType>(dim);
    if (sizes_buf == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    executorch::aten::DimOrderType* dim_order_buf =
        method_allocator->allocateList<executorch::aten::DimOrderType>(dim);
    if (dim_order_buf == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    std::memcpy(
        sizes_buf, serialized_sizes, sizeof(executorch::aten::SizesType) * dim);
    std::memcpy(
        dim_order_buf,
        serialized_dim_order,
        sizeof(executorch::aten::DimOrderType) * dim);

    sizes = sizes_buf;
    dim_order = dim_order_buf;
  } else {
    // Const cast safe here as these tensors can't be resized, so these fields
    // will not be modified.
    sizes = const_cast<executorch::aten::SizesType*>(serialized_sizes);
    dim_order =
        const_cast<executorch::aten::DimOrderType*>(serialized_dim_order);
  }
  // Validate sizes before using them in case the PTE data is bad. We can't
  // detect bad positive values, but we can reject negative values, which would
  // otherwise panic in the TensorImpl ctor. dim_order_to_stride() will validate
  // dim_order.
  for (flatbuffers::uoffset_t i = 0; i < dim; i++) {
    ET_CHECK_OR_RETURN_ERROR(
        sizes[i] >= 0,
        InvalidProgram,
        "Negative size[%zu] %" PRId32,
        static_cast<size_t>(i),
        sizes[i]);
  }

  // We will remove strides from schema.
  // Allocating strides buffer here and populating it.
  // In subsequent diffs we can remove strides accessor, however this
  // will introduce incompatible APIs between ATen Tensor and ETensor.
  executorch::aten::StridesType* strides =
      method_allocator->allocateList<executorch::aten::StridesType>(dim);
  if (strides == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  auto status = dim_order_to_stride(sizes, dim_order, dim, strides);
  ET_CHECK_OR_RETURN_ERROR(
      status == Error::Ok,
      Internal,
      "dim_order_to_stride returned invalid status");

  auto* tensor_impl = method_allocator->allocateInstance<TensorImpl>();
  if (tensor_impl == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  // Placement new on the allocated memory space. Note that we create this first
  // with null data so we can find its expected size before getting its memory.
  new (tensor_impl) TensorImpl(
      scalar_type,
      dim,
      sizes,
      /*data=*/nullptr,
      dim_order,
      strides,
      dynamism);

  // Now that we know how big the tensor is, find and assign its memory.
  Result<void*> data_ptr = getTensorDataPtr(
      s_tensor,
      program,
      tensor_impl->nbytes(),
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
  tensor_impl->set_data(data_ptr.get());

  return Tensor(tensor_impl);
}

} // namespace deserialization
} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
