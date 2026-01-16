/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef CUDA_AVAILABLE

#include <cuda.h>
#include <cuda_runtime.h>

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

/// Checks a CUDA expression and aborts on error.
/// @param EXPR The CUDA expression to check.
#define ET_CUDA_CHECK(EXPR)                                                 \
  do {                                                                      \
    const cudaError_t __err = EXPR;                                         \
    ET_CHECK_MSG(                                                           \
        __err == cudaSuccess, "CUDA error: %s", cudaGetErrorString(__err)); \
  } while (0)

/// Checks a CUDA expression and logs a warning on error (non-fatal).
/// @param EXPR The CUDA expression to check.
#define ET_CUDA_LOG_WARN(EXPR)                                      \
  do {                                                              \
    const cudaError_t __err = EXPR;                                 \
    if (SLIMTENSOR_UNLIKELY(__err != cudaSuccess)) {                \
      [[maybe_unused]] auto error_unused = cudaGetLastError();      \
      ET_LOG(Error, "CUDA warning: %s", cudaGetErrorString(__err)); \
    }                                                               \
  } while (0)

#endif // CUDA_AVAILABLE
