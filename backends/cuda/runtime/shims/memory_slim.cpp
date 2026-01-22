/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/shims/memory_slim.h>

#include <executorch/backends/aoti/slim/factory/Empty.h>
#include <executorch/backends/aoti/slim/factory/FromBlob.h>
#include <executorch/backends/aoti/slim/util/ArrayRefUtil.h>
#include <executorch/runtime/platform/assert.h>

namespace executorch::backends::cuda {

namespace c10 = executorch::backends::aoti::slim::c10;
using c10::Device;
using c10::DeviceIndex;
using c10::DeviceType;
using c10::ScalarType;
using executorch::backends::aoti::slim::empty_strided;
using executorch::backends::aoti::slim::makeArrayRef;
using executorch::backends::aoti::slim::from_blob;
using executorch::backends::aoti::slim::IntArrayRef;

extern "C" {

AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    Tensor** ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  // Unused parameters
  (void)layout;
  (void)opaque_metadata;
  (void)opaque_metadata_size;

  ET_CHECK_OR_RETURN_ERROR(
      data != nullptr,
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2: data is null");

  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr,
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2: ret_new_tensor is null");

  ET_CHECK_OR_RETURN_ERROR(
      !(sizes_ptr == nullptr && ndim > 0),
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2: sizes_ptr is null but ndim > 0");

  IntArrayRef sizes(sizes_ptr, static_cast<size_t>(ndim));
  IntArrayRef strides(strides_ptr, static_cast<size_t>(ndim));

  // Create the SlimTensor using from_blob (non-owning)
  *ret_new_tensor = new Tensor(from_blob(
      data,
      sizes,
      strides,
      static_cast<ScalarType>(dtype),
      Device(
          static_cast<DeviceType>(device_type),
          static_cast<DeviceIndex>(device_index)),
      storage_offset));

  return Error::Ok;
}

AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    Tensor** ret_new_tensor) {
  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr,
      InvalidArgument,
      "aoti_torch_empty_strided: ret_new_tensor is null");

  ET_CHECK_OR_RETURN_ERROR(
      !(sizes_ptr == nullptr && ndim > 0),
      InvalidArgument,
      "aoti_torch_empty_strided: sizes_ptr is null but ndim > 0");

  IntArrayRef sizes(sizes_ptr, static_cast<size_t>(ndim));

  // Handle nullptr strides by computing contiguous strides
  if (strides_ptr == nullptr) {
    std::vector<int64_t> contig_strides =
        executorch::backends::aoti::slim::compute_contiguous_strides(sizes);
    *ret_new_tensor = new Tensor(empty_strided(
        sizes,
        makeArrayRef(contig_strides),
        static_cast<ScalarType>(dtype),
        Device(
            static_cast<DeviceType>(device_type),
            static_cast<DeviceIndex>(device_index))));
  } else {
    IntArrayRef strides(strides_ptr, static_cast<size_t>(ndim));
    *ret_new_tensor = new Tensor(empty_strided(
        sizes,
        strides,
        static_cast<ScalarType>(dtype),
        Device(
            static_cast<DeviceType>(device_type),
            static_cast<DeviceIndex>(device_index))));
  }

  return Error::Ok;
}

AOTITorchError aoti_torch_delete_tensor_object(Tensor* tensor) {
  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr,
      InvalidArgument,
      "aoti_torch_delete_tensor_object: tensor is null");

  // SlimTensor uses SharedPtr for storage, so simply deleting the tensor
  // will automatically handle reference counting and free the underlying
  // storage when no more references exist.
  delete tensor;

  return Error::Ok;
}

AOTITorchError aoti_torch_new_tensor_handle(
    Tensor* orig_handle,
    Tensor** new_handle) {
  ET_CHECK_OR_RETURN_ERROR(
      orig_handle != nullptr,
      InvalidArgument,
      "aoti_torch_new_tensor_handle: orig_handle is null");

  ET_CHECK_OR_RETURN_ERROR(
      new_handle != nullptr,
      InvalidArgument,
      "aoti_torch_new_tensor_handle: new_handle is null");

  // Create a new SlimTensor that shares the same underlying storage.
  // SlimTensor's copy constructor shares the SharedPtr<Storage>, so both
  // tensors will reference the same memory. When the last tensor is deleted,
  // the storage will be freed.
  *new_handle = new Tensor(*orig_handle);

  return Error::Ok;
}

} // extern "C"

} // namespace executorch::backends::cuda
