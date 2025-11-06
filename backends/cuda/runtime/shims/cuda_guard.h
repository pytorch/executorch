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
#include <executorch/backends/cuda/runtime/guard.h>
#include <cstdint>

namespace executorch::backends::cuda {

using executorch::backends::aoti::AOTITorchError;

extern "C" {

// Handle types for CUDA guards
using CUDAGuardHandle = CUDAGuard*;
using CUDAStreamGuardHandle = CUDAStreamGuard*;

/**
 * Creates a CUDA device guard that sets the current device and restores it
 * upon destruction.
 *
 * @param device_index The device index to set as current
 * @param ret_guard Output parameter for the created guard handle (must not be
 * null)
 * @return AOTITorchError error code (Error::Ok on success, or an error code on
 * failure)
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_create_cuda_guard(int32_t device_index, CUDAGuardHandle* ret_guard);

/**
 * Deletes a CUDA device guard and frees its associated resources.
 *
 * @param guard Handle to the guard to be deleted
 * @return AOTITorchError error code (Error::Ok on success, or an error code on
 * failure)
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_delete_cuda_guard(CUDAGuardHandle guard);

/**
 * Sets the CUDA device to a new index for an existing guard.
 *
 * @param guard Handle to the guard
 * @param device_index The device index to set as current
 * @return AOTITorchError error code (Error::Ok on success, or an error code on
 * failure)
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_cuda_guard_set_index(CUDAGuardHandle guard, int32_t device_index);

/**
 * Creates a CUDA stream guard that sets the current device and stream,
 * restoring both upon destruction.
 *
 * @param stream The CUDA stream to set as current
 * @param device_index The device index for the stream
 * @param ret_guard Output parameter for the created guard handle (must not be
 * null)
 * @return AOTITorchError error code (Error::Ok on success, or an error code on
 * failure)
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard);

/**
 * Deletes a CUDA stream guard and frees its associated resources.
 *
 * @param guard Handle to the stream guard to be deleted
 * @return AOTITorchError error code (Error::Ok on success, or an error code on
 * failure)
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_delete_cuda_stream_guard(CUDAStreamGuardHandle guard);

/**
 * Gets the current CUDA stream for a specified device.
 *
 * @param device_index The device index (-1 to use current device)
 * @param ret_stream Output parameter for the current stream (must not be null)
 * @return AOTITorchError error code (Error::Ok on success, or an error code on
 * failure)
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream);

} // extern "C"

} // namespace executorch::backends::cuda
