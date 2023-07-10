#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/util/DimOrderUtils.h>
#include <executorch/core/values/Evalue.h>
#include <executorch/executor/Program.h>
#include <executorch/executor/tensor_parser.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/profiler.h>

namespace torch {
namespace executor {
namespace deserialization {

Result<torch::executor::Tensor> parseTensor(
    const Program* program,
    MemoryManager* memory_manager,
    const executorch::Tensor* s_tensor) {
  EXECUTORCH_SCOPE_PROF("TensorParser::parseTensor");
  size_t dim = s_tensor->sizes()->size();
  auto runtime_allocator = memory_manager->get_runtime_allocator();
  auto* tensor_impl = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
      runtime_allocator, torch::executor::TensorImpl);

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
  auto serialized_sizes = s_tensor->sizes()->data();
  auto serialized_dim_order = s_tensor->dim_order()->data();
  // For dynamic shape tensors, allocate local buffers to allow mutable sizes
  // and strides
  if (dynamism != TensorShapeDynamism::STATIC) {
    // copy sizes and dim order out of flatbuffer
    // kimishpate: I think dim order can remain immutable and point to fb
    // memory, unless we plan to implement in-place permute
    exec_aten::SizesType* sizes_buf = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
        runtime_allocator, exec_aten::SizesType, dim);
    exec_aten::DimOrderType* dim_order_buf = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
        runtime_allocator, exec_aten::DimOrderType, dim);
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
      runtime_allocator, exec_aten::StridesType, dim);
  auto status =
      torch::executor::dim_order_to_stride(sizes, dim_order, dim, strides);
  ET_CHECK_OR_RETURN_ERROR(
      status == Error::Ok,
      Internal,
      "dim_order_to_stride returned invalid status");

  // Placement new on the allocated memory space. Note that we create this first
  // with null data so we can find its expected size before getting its memory.
  new (tensor_impl) torch::executor::TensorImpl(
      static_cast<ScalarType>(s_tensor->scalar_type()),
      dim,
      sizes,
      /*data=*/nullptr,
      dim_order,
      strides,
      s_tensor->storage_offset(),
      dynamism);

  // Now that we know how big the tensor is, find and assign its memory.
  Result<void*> data_ptr = getTensorDataPtr(
      s_tensor,
      program,
      tensor_impl->nbytes(),
      memory_manager->get_non_constant_allocator());
  if (!data_ptr.ok()) {
    ET_LOG(Error, "getTensorDataPtr() failed: 0x%" PRIx32, data_ptr.error());
    return data_ptr.error();
  }
  tensor_impl->set_data(data_ptr.get());

  return torch::executor::Tensor(tensor_impl);
}

} // namespace deserialization
} // namespace executor
} // namespace torch
