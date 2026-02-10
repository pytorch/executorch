/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/aoti/slim/util/array_ref_util.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

namespace c10 = executorch::backends::aoti::slim::c10;
using c10::Device;
using c10::DeviceIndex;
using c10::DeviceType;
using c10::ScalarType;
using executorch::backends::aoti::slim::from_blob;
using executorch::backends::aoti::slim::IntArrayRef;
using executorch::runtime::Error;

using SlimTensor = executorch::backends::aoti::slim::SlimTensor;
using AOTITorchError = Error;

extern "C" {

AOTI_SHIM_EXPORT void aoti_torch_warn(
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg) {
  ET_LOG(Error, "[%s:%u] %s: %s", file, line, func, msg);
}

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_clone_preserve_strides(SlimTensor* self, SlimTensor** ret_new_tensor) {
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr,
      InvalidArgument,
      "aoti_torch_clone_preserve_strides: self is null");
  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr,
      InvalidArgument,
      "aoti_torch_clone_preserve_strides: ret_new_tensor is null");

  *ret_new_tensor = new SlimTensor(self->clone());
  return Error::Ok;
}

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_clone(SlimTensor* self, SlimTensor** ret_new_tensor) {
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr,
      InvalidArgument,
      "aoti_torch_clone: self is null");
  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr,
      InvalidArgument,
      "aoti_torch_clone: ret_new_tensor is null");

  *ret_new_tensor = new SlimTensor(self->clone());
  return Error::Ok;
}

AOTI_SHIM_EXPORT AOTITorchError aoti_torch_create_tensor_from_blob(
    void* data_ptr,
    int64_t ndim,
    const int64_t* sizes,
    const int64_t* strides,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    SlimTensor** ret_new_tensor) {
  ET_CHECK_OR_RETURN_ERROR(
      data_ptr != nullptr,
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob: data_ptr is null");
  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr,
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob: ret_new_tensor is null");
  ET_CHECK_OR_RETURN_ERROR(
      !(sizes == nullptr && ndim > 0),
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob: sizes is null but ndim > 0");

  IntArrayRef sizes_ref(sizes, static_cast<size_t>(ndim));
  IntArrayRef strides_ref(strides, static_cast<size_t>(ndim));

  *ret_new_tensor = new SlimTensor(from_blob(
      data_ptr,
      sizes_ref,
      strides_ref,
      static_cast<ScalarType>(dtype),
      Device(
          static_cast<DeviceType>(device_type),
          static_cast<DeviceIndex>(device_index)),
      storage_offset));

  return Error::Ok;
}

} // extern "C"

} // namespace executorch::backends::cuda
