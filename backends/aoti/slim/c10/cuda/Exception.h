/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

/// Checks a CUDA expression and aborts on error.
/// @param EXPR The CUDA expression to check.
#ifndef ET_CUDA_CHECK
#define ET_CUDA_CHECK(EXPR)                                           \
  do {                                                                \
    const cudaError_t __err = EXPR;                                   \
    if (__err == cudaSuccess) {                                       \
      break;                                                          \
    }                                                                 \
    ET_LOG(                                                           \
        Error,                                                        \
        "%s:%d CUDA error: %s",                                       \
        __FILE__,                                                     \
        __LINE__,                                                     \
        cudaGetErrorString(__err));                                   \
    ET_CHECK_MSG(false, "CUDA error: %s", cudaGetErrorString(__err)); \
  } while (0)
#endif

/// Checks a CUDA expression and returns Error::Internal on failure.
/// @param EXPR The CUDA expression to check.
#ifndef ET_CUDA_CHECK_OR_RETURN_ERROR
#define ET_CUDA_CHECK_OR_RETURN_ERROR(EXPR)        \
  do {                                             \
    const cudaError_t __err = EXPR;                \
    if (__err == cudaSuccess) {                    \
      break;                                       \
    }                                              \
    ET_LOG(                                        \
        Error,                                     \
        "%s:%d CUDA error: %s",                    \
        __FILE__,                                  \
        __LINE__,                                  \
        cudaGetErrorString(__err));                \
    return ::executorch::runtime::Error::Internal; \
  } while (0)
#endif

/// Checks a CUDA expression and logs a warning on error (non-fatal).
/// @param EXPR The CUDA expression to check.
#ifndef ET_CUDA_LOG_WARN
#define ET_CUDA_LOG_WARN(EXPR)                                      \
  do {                                                              \
    const cudaError_t __err = EXPR;                                 \
    if (SLIMTENSOR_UNLIKELY(__err != cudaSuccess)) {                \
      [[maybe_unused]] auto error_unused = cudaGetLastError();      \
      ET_LOG(Error, "CUDA warning: %s", cudaGetErrorString(__err)); \
    }                                                               \
  } while (0)
#endif

/// Kernel launch check macro (with return) - checks cudaGetLastError after
/// kernel launch.
#ifndef ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR
#define ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR() \
  ET_CUDA_CHECK_OR_RETURN_ERROR(cudaGetLastError())
#endif

/// Kernel launch check macro (without return) - checks cudaGetLastError after
/// kernel launch.
#ifndef ET_CUDA_KERNEL_LAUNCH_CHECK
#define ET_CUDA_KERNEL_LAUNCH_CHECK() ET_CUDA_CHECK(cudaGetLastError())
#endif
