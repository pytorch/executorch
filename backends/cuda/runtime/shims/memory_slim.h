/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/SlimTensor.h>
#include <executorch/backends/aoti/slim/core/SlimTensorView-incl.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {

using executorch::runtime::Error;
using AOTITorchError = Error;
using Tensor = executorch::backends::aoti::slim::SlimTensor;

extern "C" {

/**
 * Creates a tensor object from an existing memory blob without copying the
 * data. The tensor will wrap the provided memory and will not take ownership of
 * it. When the tensor is deleted, the original memory will remain valid and
 * must be freed by the caller.
 *
 * @param data Pointer to the memory blob to wrap (must not be null)
 * @param ndim Number of dimensions in the tensor
 * @param sizes_ptr Pointer to array of dimension sizes
 * @param strides_ptr Pointer to array of strides for each dimension
 * @param storage_offset Storage offset in number of elements
 * @param dtype Data type identifier (matches PyTorch scalar types)
 * @param device_type Device type (CPU=0, CUDA=1)
 * @param device_index Device index
 * @param ret_new_tensor Output parameter for the created tensor
 * @param layout Tensor layout identifier (0=strided)
 * @param opaque_metadata Optional metadata pointer (can be null)
 * @param opaque_metadata_size Size of opaque metadata in bytes
 * @return AOTITorchError error code (Error::Ok on success)
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_create_tensor_from_blob_v2(
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
    int64_t opaque_metadata_size);

/**
 * Creates an uninitialized tensor with specified dimensions, strides, and
 * dtype on either CPU or CUDA device.
 *
 * @param ndim Number of dimensions in the tensor
 * @param sizes_ptr Pointer to array of dimension sizes
 * @param strides_ptr Pointer to array of strides for each dimension
 * @param dtype Data type identifier (matches PyTorch scalar types)
 * @param device_type Device type (0=CPU, 1=CUDA)
 * @param device_index Device index
 * @param ret_new_tensor Output parameter for the created tensor
 * @return AOTITorchError error code (Error::Ok on success)
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    Tensor** ret_new_tensor);

/**
 * Deletes a tensor object and frees associated resources.
 *
 * For SlimTensor, the underlying storage uses SharedPtr-based reference
 * counting. When the last tensor referencing the storage is deleted,
 * the memory is automatically freed.
 *
 * @param tensor Pointer to the tensor to delete (must not be null)
 * @return AOTITorchError error code (Error::Ok on success)
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_delete_tensor_object(Tensor* tensor);

/**
 * Creates a new tensor handle that shares storage with the original tensor.
 *
 * The new handle is a copy of the original tensor's metadata (sizes, strides,
 * dtype, device) and shares the same underlying storage via SharedPtr.
 * Both tensors will reference the same memory, and the memory will only be
 * freed when all references are deleted.
 *
 * @param orig_handle Pointer to the original tensor (must not be null)
 * @param new_handle Output parameter for the new tensor handle
 * @return AOTITorchError error code (Error::Ok on success)
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_new_tensor_handle(Tensor* orig_handle, Tensor** new_handle);

/**
 * Creates a reinterpreted view of a tensor with new sizes, strides, and offset.
 *
 * This is equivalent to torch.as_strided() - it creates a new tensor that
 * shares the same underlying storage but with different view parameters.
 *
 * @param self Original tensor to reinterpret (must not be null)
 * @param ndim Number of dimensions for the new view
 * @param sizes_ptr Pointer to array of dimension sizes
 * @param strides_ptr Pointer to array of strides for each dimension
 * @param storage_offset Storage offset in number of elements
 * @param ret_new_tensor Output parameter for the reinterpreted tensor view
 * @return AOTITorchError error code (Error::Ok on success)
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch__reinterpret_tensor(
    Tensor* self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    Tensor** ret_new_tensor);

/**
 * Copies data from source tensor to destination tensor.
 *
 * Handles all device combinations (CPU-CPU, CPU-CUDA, CUDA-CPU, CUDA-CUDA)
 * and supports tensors with different strides. The destination tensor must
 * already be allocated with sufficient storage.
 *
 * @param self Destination tensor (must not be null)
 * @param src Source tensor to copy from (must not be null)
 * @param non_blocking If true, the copy may be asynchronous (currently ignored)
 * @return AOTITorchError error code (Error::Ok on success)
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_copy_(Tensor* self, Tensor* src, int32_t non_blocking);

/**
 * Moves a tensor into a new handle and assigns it to the output parameter.
 *
 * Unlike aoti_torch_new_tensor_handle which copies, this function moves the
 * source tensor into the destination. After this operation, the source tensor
 * is left in an undefined/reset state and should not be used.
 *
 * @param src Source tensor to move from (must not be null, will be reset)
 * @param ret_dst Output parameter for the new tensor handle
 * @return AOTITorchError error code (Error::Ok on success)
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_assign_tensors_out(Tensor* src, Tensor** ret_dst);

} // extern "C"

} // namespace executorch::backends::cuda
