/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>

#include <executorch/extension/cuda/export.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#if defined(EXECUTORCH_USE_HIP)
#include <executorch/extension/cuda/runtime_api.h>
#else
// Declared rather than included, as in caller_stream.h.
typedef struct CUstream_st* cudaStream_t;
#endif

// The current stream of each device, per thread, which the CUDA delegate and
// its shims run on. It lives in this shared library, beside the caller stream,
// so that a process has one copy of it whichever libraries the delegate and its
// shims are linked into. It is the CUDA delegate's own selection, so it is
// internal: another backend should follow getCallerStream instead.
namespace executorch::extension::cuda::internal {

/**
 * Set the current CUDA stream for the specified device.
 *
 * @param stream The CUDA stream to set as current
 * @param device_index The device index (-1 to use current device)
 * @return Error code indicating success or failure
 */
EXECUTORCH_EXTENSION_CUDA_API ::executorch::runtime::Error setCurrentCUDAStream(
    cudaStream_t stream,
    int8_t device_index = -1);

/**
 * Get the current CUDA stream for the specified device.
 * If no stream has been set, returns cudaStreamPerThread, the stream the
 * delegate runs on when the caller selects none. That handle names a stream of
 * the device that is current when it is used, not of device_index.
 *
 * @param device_index The device index (-1 to use current device)
 * @return Result containing the current stream on success, or an error code on
 * failure
 */
EXECUTORCH_EXTENSION_CUDA_API ::executorch::runtime::Result<cudaStream_t>
getCurrentCUDAStream(int8_t device_index = -1);

/**
 * The CUDA stream registered for the specified device, or std::nullopt if none
 * is set. Unlike getCurrentCUDAStream, it does not fall back to
 * cudaStreamPerThread, so it can snapshot the selection exactly. Also returns
 * std::nullopt if the current device cannot be queried (device_index -1), so
 * nullopt does not distinguish "no stream set" from "device query failed".
 *
 * @param device_index The device index (-1 to use current device)
 */
EXECUTORCH_EXTENSION_CUDA_API std::optional<cudaStream_t> peekCurrentCUDAStream(
    int8_t device_index = -1);

/**
 * Clears any CUDA stream registered for the specified device, restoring the
 * "no stream selected" state. Best-effort: if device_index is -1 and the
 * current device cannot be queried, it silently does nothing.
 *
 * @param device_index The device index (-1 to use current device)
 */
EXECUTORCH_EXTENSION_CUDA_API void clearCurrentCUDAStream(
    int8_t device_index = -1);

} // namespace executorch::extension::cuda::internal
