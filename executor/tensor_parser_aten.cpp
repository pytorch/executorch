#include <ATen/ATen.h> // @manual=//caffe2/aten:ATen-core
#include <executorch/core/Error.h>
#include <executorch/core/Result.h>
#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/util/DimOrderUtils.h>
#include <executorch/core/values/Evalue.h>
#include <executorch/executor/Program.h>
#include <executorch/executor/tensor_parser.h>
#include <executorch/runtime/platform/profiler.h>

namespace torch {
namespace executor {
namespace deserialization {

using torch::executor::Error;

namespace {

void deleteNothing(void*);
void deleteNothing(void*) {}

} // namespace

Result<at::Tensor> parseTensor(
    const Program* program,
    MemoryManager* memory_manager,
    const executorch::Tensor* s_tensor) {
  EXECUTORCH_SCOPE_PROF("TensorParser::parseTensor");
  // get metadata
  at::ScalarType type = static_cast<at::ScalarType>(s_tensor->scalar_type());
  auto options = at::CPU(type).options();

  // convert int32 in serialization to int64 for aten
  size_t ndim = s_tensor->sizes()->size();
  std::vector<int64_t> sizes(
      s_tensor->sizes()->begin(), s_tensor->sizes()->end());
  std::vector<int64_t> strides(ndim);
  auto status = torch::executor::dim_order_to_stride(
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
      s_tensor->storage_offset(),
      deleteNothing,
      options);

  if (s_tensor->shape_dynamism() ==
      executorch::TensorShapeDynamism::DYNAMIC_UNBOUND) {
    // Provide fully dynamic tensors with an allocator so they can be resized
    // within aten kernels.
    auto impl = tensor.unsafeGetTensorImpl();
    at::StorageImpl* storage = impl->unsafe_storage().unsafeGetStorageImpl();
    storage->set_allocator(getCPUAllocator());
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
        memory_manager->get_non_constant_allocator());
    if (!data_ptr.ok()) {
      ET_LOG(Error, "getTensorDataPtr() failed: 0x%" PRIx32, data_ptr.error());
      return data_ptr.error();
    }
    ET_CHECK_OR_RETURN_ERROR(
        data_ptr.get() != nullptr,
        Internal,
        "Expected non-null data for tensor with shape dynamism %d",
        int(s_tensor->shape_dynamism()));
    tensor.unsafeGetTensorImpl()->unsafe_storage().set_data_ptr(
        at::DataPtr(data_ptr.get(), DeviceType::CPU));
  }

  return tensor;
}

} // namespace deserialization
} // namespace executor
} // namespace torch
