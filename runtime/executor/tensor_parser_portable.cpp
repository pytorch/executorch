/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <executorch/runtime/executor/tensor_parser.h>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/schema/program_generated.h>

namespace torch {
namespace executor {
namespace deserialization {

Result<torch::executor::Tensor> parseTensor(
    const Program* program,
    MemoryManager* memory_manager,
    const executorch_flatbuffer::Tensor* s_tensor) {
  EXECUTORCH_SCOPE_PROF("TensorParser::parseTensor");
  auto method_allocator = memory_manager->method_allocator();

  ET_CHECK_OR_RETURN_ERROR(
      s_tensor->storage_offset() == 0,
      NotSupported,
      "Non-zero storage offset %" PRId32 " not supported",
      s_tensor->storage_offset());

  TensorShapeDynamism dynamism =
      static_cast<TensorShapeDynamism>(s_tensor->shape_dynamism());
  // TODO(T133200526): Remove this check once fully dynamic shapes are
  // supported.
  ET_CHECK_OR_RETURN_ERROR(
      dynamism != TensorShapeDynamism::DYNAMIC_UNBOUND,
      NotSupported,
      "Fully dynamic tensor shapes not yet supported: T133200526");

  exec_aten::SizesType* sizes = nullptr;
  exec_aten::DimOrderType* dim_order = nullptr;
  const auto dim = s_tensor->sizes()->size();
  const auto serialized_sizes = s_tensor->sizes()->data();
  const auto serialized_dim_order = s_tensor->dim_order()->data();
  // For dynamic shape tensors, allocate local buffers to allow mutable sizes
  // and strides
  if (dynamism != TensorShapeDynamism::STATIC) {
    // copy sizes and dim order out of flatbuffer
    // kimishpate: I think dim order can remain immutable and point to fb
    // memory, unless we plan to implement in-place permute
    exec_aten::SizesType* sizes_buf = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
        method_allocator, exec_aten::SizesType, dim);
    exec_aten::DimOrderType* dim_order_buf = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
        method_allocator, exec_aten::DimOrderType, dim);
    std::memcpy(
        sizes_buf, serialized_sizes, sizeof(exec_aten::SizesType) * dim);
    std::memcpy(
        dim_order_buf,
        serialized_dim_order,
        sizeof(exec_aten::DimOrderType) * dim);

    sizes = sizes_buf;
    dim_order = dim_order_buf;
  } else {
    // Const cast safe here as these tensors can't be resized, so these fields
    // will not be modified.
    sizes = const_cast<exec_aten::SizesType*>(serialized_sizes);
    dim_order = const_cast<exec_aten::DimOrderType*>(serialized_dim_order);
  }
  // We will remove strides from schema.
  // Allocating strides buffer here and populating it.
  // In subsequent diffs we can remove strides accessor, however this
  // will introduce incompatible APIs between ATen Tensor and ETensor.
  exec_aten::StridesType* strides = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
      method_allocator, exec_aten::StridesType, dim);
  auto status =
      torch::executor::dim_order_to_stride(sizes, dim_order, dim, strides);
  ET_CHECK_OR_RETURN_ERROR(
      status == Error::Ok,
      Internal,
      "dim_order_to_stride returned invalid status");

  auto* tensor_impl = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
      method_allocator, torch::executor::TensorImpl);
  // Placement new on the allocated memory space. Note that we create this first
  // with null data so we can find its expected size before getting its memory.
  new (tensor_impl) torch::executor::TensorImpl(
      static_cast<ScalarType>(s_tensor->scalar_type()),
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
      memory_manager->planned_memory());
  if (!data_ptr.ok()) {
    ET_LOG(
        Error,
        "getTensorDataPtr() failed: 0x%" PRIx32,
        static_cast<uint32_t>(data_ptr.error()));
    return data_ptr.error();
  }
  tensor_impl->set_data(data_ptr.get());

  std::cout << "tensor_parser_portable.cpp: tensor values" << std::endl;
  for (int i = 0; i < tensor_impl->numel(); ++i) {
    std::cout << tensor_impl->data<float>()[i] << ", ";
  }
  std::cout << std::endl;
  return torch::executor::Tensor(tensor_impl);
}

} // namespace deserialization
} // namespace executor
} // namespace torch
