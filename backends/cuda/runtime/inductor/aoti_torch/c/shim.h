#pragma once

// This header reimplements C shim functions in libtorch

#include <standalone/c10/core/Layout.h>
#include <standalone/slim/core/Empty.h>
#include <standalone/slim/core/FromBlob.h>
#include <standalone/slim/core/FromScalar.h>
#include <standalone/torch/csrc/inductor/aoti_torch/c/macros.h>
#ifdef USE_CUDA
#include <standalone/torch/csrc/inductor/aoti_torch/c/shim_cuda.h>
#endif

using AtenTensorOpaque = standalone::slim::SlimTensor;
using AtenTensorHandle = standalone::slim::SlimTensor *;

// AOTIProxyExecutorHandle isn't supported in standalone mode.
// Just defining it to void* to make the code compile
using AOTIProxyExecutorHandle = void *;

#ifdef __cplusplus
extern "C" {
#endif

// DeviceType
#define AOTI_TORCH_DEVICE_TYPE_IMPL(device_str, device_type)                   \
  AOTI_TORCH_EXPORT int32_t aoti_torch_device_type_##device_str() {            \
    return (int32_t)c10::DeviceType::device_type;                              \
  }

AOTI_TORCH_DEVICE_TYPE_IMPL(cpu, CPU)
AOTI_TORCH_DEVICE_TYPE_IMPL(cuda, CUDA)
AOTI_TORCH_DEVICE_TYPE_IMPL(mps, MPS)
AOTI_TORCH_DEVICE_TYPE_IMPL(xpu, XPU)
#undef AOTI_TORCH_DEVICE_TYPE_IMPL

// SclarType
#define AOTI_TORCH_DTYPE_IMPL(dtype, stype)                                    \
  AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_##dtype() {                       \
    return (int32_t)c10::ScalarType::stype;                                    \
  }

AOTI_TORCH_DTYPE_IMPL(float8_e5m2, Float8_e5m2)
AOTI_TORCH_DTYPE_IMPL(float8_e4m3fn, Float8_e4m3fn)
AOTI_TORCH_DTYPE_IMPL(float8_e5m2fnuz, Float8_e5m2fnuz)
AOTI_TORCH_DTYPE_IMPL(float8_e4m3fnuz, Float8_e4m3fnuz)
AOTI_TORCH_DTYPE_IMPL(bfloat16, BFloat16)
AOTI_TORCH_DTYPE_IMPL(float16, Half)
AOTI_TORCH_DTYPE_IMPL(float32, Float)
AOTI_TORCH_DTYPE_IMPL(float64, Double)
AOTI_TORCH_DTYPE_IMPL(uint8, Byte)
AOTI_TORCH_DTYPE_IMPL(uint16, UInt16)
AOTI_TORCH_DTYPE_IMPL(uint32, UInt32)
AOTI_TORCH_DTYPE_IMPL(uint64, UInt64)
AOTI_TORCH_DTYPE_IMPL(int8, Char)
AOTI_TORCH_DTYPE_IMPL(int16, Short)
AOTI_TORCH_DTYPE_IMPL(int32, Int)
AOTI_TORCH_DTYPE_IMPL(int64, Long)
AOTI_TORCH_DTYPE_IMPL(bool, Bool)
AOTI_TORCH_DTYPE_IMPL(complex32, ComplexHalf)
AOTI_TORCH_DTYPE_IMPL(complex64, ComplexFloat)
AOTI_TORCH_DTYPE_IMPL(complex128, ComplexDouble)
#undef AOTI_TORCH_DTYPE_IMPL

#define AOTI_TORCH_LAYOUT_IMPL(name, enum)                                     \
  AOTI_TORCH_EXPORT int32_t aoti_torch_layout_##name() {                       \
    return (int32_t)c10::Layout::enum;                                         \
  }

AOTI_TORCH_LAYOUT_IMPL(strided, Strided)
AOTI_TORCH_LAYOUT_IMPL(sparse_coo, Sparse)
AOTI_TORCH_LAYOUT_IMPL(sparse_csr, SparseCsr)
AOTI_TORCH_LAYOUT_IMPL(sparse_csc, SparseCsc)
AOTI_TORCH_LAYOUT_IMPL(sparse_bsr, SparseBsr)
AOTI_TORCH_LAYOUT_IMPL(sparse_bsc, SparseBsc)
AOTI_TORCH_LAYOUT_IMPL(_mkldnn, Mkldnn)
AOTI_TORCH_LAYOUT_IMPL(jagged, Jagged)
#undef AOTI_TORCH_LAYOUT_IMPL

#define AOTI_TORCH_MEMORY_FORMAT_IMPL(name, enum)                              \
  AOTI_TORCH_EXPORT int32_t aoti_torch_memory_format_##name() {                \
    return (int32_t)c10::MemoryFormat::enum;                                   \
  }

AOTI_TORCH_MEMORY_FORMAT_IMPL(contiguous_format, Contiguous)
AOTI_TORCH_MEMORY_FORMAT_IMPL(channels_last, ChannelsLast)
AOTI_TORCH_MEMORY_FORMAT_IMPL(channels_last_3d, ChannelsLast3d)
AOTI_TORCH_MEMORY_FORMAT_IMPL(preserve_format, Preserve)
#undef AOTI_TORCH_MEMORY_FORMAT_IMPL

#define AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(dtype, ctype)                         \
  AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_##dtype(        \
      ctype value, AtenTensorHandle *ret_new_tensor) {                         \
    *ret_new_tensor = new standalone::slim::SlimTensor(                        \
        standalone::slim::scalar_to_tensor(value));                            \
    return AOTI_TORCH_SUCCESS;                                                 \
  }

AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(float32, float)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(float64, double)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint8, uint8_t)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint16, uint16_t)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint32, uint32_t)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint64, uint64_t)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int8, int8_t)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int16, int16_t)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int32, int32_t)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int64, int64_t)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(bool, bool)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(complex64, c10::complex<float>)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(complex128, c10::complex<double>)
#undef AOTI_TORCH_SCALAR_TO_TENSOR_IMPL

AOTI_TORCH_EXPORT bool aoti_torch_grad_mode_is_enabled() { return false; }

AOTI_TORCH_EXPORT void aoti_torch_grad_mode_set_enabled(bool enabled) {
  // do nothing
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_tensor_object(AtenTensorHandle tensor) {
  delete tensor;
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_data_ptr(AtenTensorHandle tensor, void **ret_data_ptr) {
  *ret_data_ptr = tensor->data_ptr();
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_dtype(AtenTensorHandle tensor,
                                                      int32_t *ret_dtype) {
  *ret_dtype = static_cast<int32_t>(tensor->dtype());
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_device_type(AtenTensorHandle tensor, int32_t *ret_device_type) {
  *ret_device_type = static_cast<int32_t>(tensor->device_type());
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_device_index(
    AtenTensorHandle tensor, int32_t *ret_device_index) {
  *ret_device_index = static_cast<uint8_t>(tensor->device_index());
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_sizes(AtenTensorHandle tensor,
                                                      int64_t **ret_sizes) {
  *ret_sizes = (int64_t *)tensor->sizes().data();
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_size(AtenTensorHandle tensor,
                                                     int64_t d,
                                                     int64_t *ret_size) {
  *ret_size = tensor->size(d);
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_strides(AtenTensorHandle tensor,
                                                        int64_t **ret_strides) {
  *ret_strides = (int64_t *)tensor->strides().data();
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_stride(AtenTensorHandle tensor,
                                                       int64_t d,
                                                       int64_t *ret_stride) {
  *ret_stride = tensor->stride(d);
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_storage_size(AtenTensorHandle tensor, int64_t *ret_size) {
  *ret_size = static_cast<int64_t>(tensor->nbytes());
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_storage_offset(
    AtenTensorHandle tensor, int64_t *ret_storage_offset) {
  *ret_storage_offset = tensor->storage_offset();
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_new_tensor_handle(
    AtenTensorHandle orig_handle, AtenTensorHandle *new_handle) {
  *new_handle = new standalone::slim::SlimTensor(*orig_handle);
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_tensor_from_blob(
    void *data, int64_t ndim, const int64_t *sizes_ptr,
    const int64_t *strides_ptr, int64_t storage_offset, int32_t dtype,
    int32_t device_type, int32_t device_index,
    AtenTensorHandle *ret_new_tensor) {
  c10::IntArrayRef sizes(sizes_ptr, ndim);
  c10::IntArrayRef strides(strides_ptr, ndim);
  *ret_new_tensor =
      new standalone::slim::SlimTensor(standalone::slim::from_blob(
          data, sizes, strides, static_cast<c10::ScalarType>(dtype),
          {static_cast<c10::DeviceType>(device_type),
           static_cast<c10::DeviceIndex>(device_index)},
          storage_offset));
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void *data, int64_t ndim, const int64_t *sizes_ptr,
    const int64_t *strides_ptr, int64_t storage_offset, int32_t dtype,
    int32_t device_type, int32_t device_index, AtenTensorHandle *ret_new_tensor,
    int32_t layout, const uint8_t *opaque_metadata,
    int64_t opaque_metadata_size) {
  c10::IntArrayRef sizes(sizes_ptr, ndim);
  c10::IntArrayRef strides(strides_ptr, ndim);
  *ret_new_tensor =
      new standalone::slim::SlimTensor(standalone::slim::from_blob(
          data, sizes, strides, static_cast<c10::ScalarType>(dtype),
          {static_cast<c10::DeviceType>(device_type),
           static_cast<c10::DeviceIndex>(device_index)},
          storage_offset));
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_empty_strided(
    int64_t ndim, const int64_t *sizes_ptr, const int64_t *strides_ptr,
    int32_t dtype, int32_t device_type, int32_t device_index,
    AtenTensorHandle *ret_new_tensor) {
  c10::IntArrayRef sizes(sizes_ptr, ndim);
  c10::IntArrayRef strides(strides_ptr, ndim);
  *ret_new_tensor =
      new standalone::slim::SlimTensor(standalone::slim::empty_strided(
          sizes, strides, static_cast<c10::ScalarType>(dtype),
          {static_cast<c10::DeviceType>(device_type),
           static_cast<c10::DeviceIndex>(device_index)}));
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch__reinterpret_tensor(
    AtenTensorHandle self, int64_t ndim, const int64_t *sizes_ptr,
    const int64_t *strides_ptr, int64_t offset_increment,
    AtenTensorHandle *ret_new_tensor) {
  c10::IntArrayRef sizes(sizes_ptr, ndim);
  c10::IntArrayRef strides(strides_ptr, ndim);
  *ret_new_tensor = new standalone::slim::SlimTensor(
      self->storage(), sizes, strides, self->dtype(),
      self->storage_offset() + offset_increment);
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_as_strided(AtenTensorHandle self, const int64_t *sizes_ptr,
                      const int64_t *strides_ptr, AtenTensorHandle *ret) {
  c10::IntArrayRef sizes(sizes_ptr, self->dim());
  c10::IntArrayRef strides(strides_ptr, self->dim());
  *ret = new standalone::slim::SlimTensor(
      self->storage(), sizes, strides, self->dtype(), self->storage_offset());
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_clone(AtenTensorHandle self,
                                                  AtenTensorHandle *ret) {
  standalone::slim::SlimTensor tmp_tensor = standalone::slim::empty_strided(
      self->sizes(), self->strides(), self->dtype(),
      {self->device_type(), self->device_index()});
  tmp_tensor.copy_(*self);
  *ret = new standalone::slim::SlimTensor(tmp_tensor);
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_clone_preserve_strides(
    AtenTensorHandle self, AtenTensorHandle *ret) {
  int64_t needed_size = 1;
  for (size_t i = 0; i < self->dim(); i++) {
    if (self->size(i) == 0) {
      needed_size = 0;
      break;
    }
    needed_size += (self->size(i) - 1) * self->stride(i);
  }
  standalone::slim::SlimTensor tmp_tensor = *self;
  tmp_tensor.as_strided_({needed_size}, {1}, 0);
  aoti_torch_clone(&tmp_tensor, ret);
  (*ret)->as_strided_(self->sizes(), self->strides(), self->storage_offset());
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT void aoti_torch_warn(const char *func, const char *file,
                                       uint32_t line, const char *msg) {
  std::cerr << "[AOTInductor Warning] " << (msg ? msg : "(no message)")
            << " (in " << (func ? func : "(unknown function)") << " at "
            << (file ? file : "(unknown file)") << ":" << line << ")"
            << std::endl;
}

#ifdef __cplusplus
} // extern "C"
#endif
