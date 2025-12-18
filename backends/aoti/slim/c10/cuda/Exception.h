#pragma once
#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

#define STANDALONE_CUDA_CHECK(EXPR)                                         \
  do {                                                                      \
    const cudaError_t __err = EXPR;                                         \
    ET_CHECK_MSG(                                                           \
        __err == cudaSuccess, "CUDA error: %s", cudaGetErrorString(__err)); \
  } while (0)

#define STANDALONE_CUDA_CHECK_WARN(EXPR)                            \
  do {                                                              \
    const cudaError_t __err = EXPR;                                 \
    if (STANDALONE_UNLIKELY(__err != cudaSuccess)) {                \
      [[maybe_unused]] auto error_unused = cudaGetLastError();      \
      ET_LOG(Error, "CUDA warning: %s", cudaGetErrorString(__err)); \
    }                                                               \
  } while (0)

#endif
