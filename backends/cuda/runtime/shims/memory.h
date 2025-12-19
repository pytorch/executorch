/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/export.h>
#include <cstdint>

namespace executorch::backends::cuda {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

extern "C" {

/**
 * Creates a tensor object from an existing memory blob without copying the
 * data. The tensor will wrap the provided memory and will not take ownership of
 * it. When the tensor is deleted, the original memory will remain valid and
 * must be freed by the caller.
 *
 * @param data Pointer to the memory blob to wrap (must not be null)
 * @param ndim Number of dimensions in the tensor
 * @param sizes_ptr Pointer to array of dimension sizes (using SizesType)
 * @param strides_ptr Pointer to array of strides for each dimension (using
 * StridesType, can be null for contiguous)
 * @param storage_offset Storage offset (must be 0 for current implementation)
 * @param dtype Data type identifier (supports FLOAT32 and BFLOAT16 from
 * SupportedDTypes)
 * @param device_type Device type (CPU=0, CUDA=1 from SupportedDevices)
 * @param device_index Device index (must be 0 for current implementation)
 * @param ret_new_tensor Output parameter for the created tensor (must not be
 * null)
 * @param layout Tensor layout identifier (0=strided)
 * @param opaque_metadata Optional metadata pointer (can be null)
 * @param opaque_metadata_size Size of opaque metadata in bytes
 * @return AOTITorchError error code (Error::Ok on success, or an error code on
 * failure)
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
 * dtyper on either CPU or CUDA device.
 *
 * @param ndim Number of dimensions in the tensor
 * @param sizes_ptr Pointer to array of dimension sizes
 * @param strides_ptr Pointer to array of strides for each dimension
 * @param dtype Data type identifier (matches PyTorch scalar types)
 * @param device_type Device type (0=CPU, 1=CUDA)
 * @param device_index Device index (must be 0 for current implementation)
 * @param ret_new_tensor Output parameter for the created tensor
 * @return AOTITorchError error code (Error::Ok on success, or an error code on
 * failure)
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
 * Deletes a tensor object and frees its associated memory.
 *
 * @param tensor Pointer to the tensor object to be deleted
 * @return AOTITorchError error code (Error::Ok on success, or an error code on
 * failure)
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_delete_tensor_object(Tensor* tensor);

/**
 * Creates a tensor view that reinterprets the same underlying memory with
 * different shape and strides without copying data.
 *
 * Note that the new tensor will not have the ownership of the underlying
 * memory.
 *
 * @param self Input tensor whose memory will be reinterpreted
 * @param ndim Number of dimensions for the new tensor view
 * @param sizes_ptr Array of sizes for each dimension
 * @param strides_ptr Array of strides for each dimension (or nullptr for
 * contiguous)
 * @param storage_offset Storage offset (must be 0)
 * @param ret_new_tensor Output pointer to store the new tensor view
 *
 * @return Error::Ok on success, appropriate error code on failure
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
 * This function implements copy function for tensors living in CUDA AOTI
 * backend. It supports copying between tensors with different shapes (as long
 * as they have the same total number of elements) and different memory
 * layouts/strides.
 *
 * Note that currently this function does not support copying between tensors
 * with different dtypes.
 *
 * @param self Destination tensor (data will be overwritten)
 * @param src Source tensor (data will be copied from this tensor)
 * @param non_blocking Whether the copy should be non-blocking (currently
 * ignored)
 *
 * @return Error::Ok on success, appropriate error code on failure:
 *         - Error::InvalidArgument: null pointers, dtype mismatch, numel
 * mismatch
 *         - Error::MemoryAllocationFailed: failed to allocate temporary memory
 *         - Error::Internal: CUDA operation failures
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_copy_(Tensor* self, Tensor* src, int32_t non_blocking);

/**
 * Creates a new tensor handle from an existing one.
 *
 * This function creates a new tensor object that shares the same underlying
 * memory as the original tensor. Similar to PyTorch's Tensor copy constructor,
 * it creates a new handle/reference to the same data without performing a deep
 * copy.
 *
 * The new tensor will:
 * - Share the same memory/storage as the original tensor
 * - Have the same shape, strides, and dtype as the original
 * - Increment the reference count for the underlying memory (if owned)
 *
 * @param orig_handle Original tensor to create a new handle from (must not be
 * null)
 * @param new_handle Output pointer to store the new tensor handle (must not be
 * null)
 *
 * @return Error::Ok on success, appropriate error code on failure:
 *         - Error::InvalidArgument: null pointers or invalid parameters
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_new_tensor_handle(Tensor* orig_handle, Tensor** new_handle);

/**
 * Retrieves a boolean value from a 0D boolean tensor.
 *
 * This function extracts the scalar boolean value from a tensor that contains
 * a single boolean element. The tensor can be on either CPU or CUDA device.
 * For CUDA tensors, the value is copied from device to host memory.
 *
 * @param tensor Pointer to a 0D boolean tensor (must not be null)
 * @param ret_value Output pointer to store the boolean value (must not be null)
 *
 * @return Error::Ok on success, appropriate error code on failure:
 *         - Error::InvalidArgument: null pointers or tensor dtype is not bool
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_item_bool(Tensor* tensor, bool* ret_value);

/**
 * Creates a new tensor that shares the same underlying data as the source
 * tensor.
 *
 * This function creates a new tensor view with the same shape, strides, and
 * dtype as the source tensor, sharing the same underlying memory. The new
 * tensor handle will be stored in ret_dst.
 *
 * @param src The source tensor providing the data and metadata.
 * @param ret_dst On output, this will point to the new tensor view.
 *
 * @return Error::Ok on success, appropriate error code on failure:
 *         - Error::InvalidArgument: null pointers or memory not tracked
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_assign_tensors_out(Tensor* src, Tensor** ret_dst);

// Function to clear all tensors from internal storage
AOTI_SHIM_EXPORT void clear_all_tensors();
} // extern "C"

} // namespace executorch::backends::cuda
