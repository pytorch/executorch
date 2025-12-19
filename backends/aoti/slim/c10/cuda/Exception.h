#pragma once
#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <executorch/backends/aoti/slim/c10/util/Exception.h>

#define STANDALONE_CUDA_CHECK(EXPR)                                    \
  do {                                                                 \
    const cudaError_t __err = EXPR;                                    \
    STANDALONE_CHECK(__err == cudaSuccess, cudaGetErrorString(__err)); \
  } while (0)

#define STANDALONE_CUDA_CHECK_WARN(EXPR)                            \
  do {                                                              \
    const cudaError_t __err = EXPR;                                 \
    if (STANDALONE_UNLIKELY(__err != cudaSuccess)) {                \
      [[maybe_unused]] auto error_unused = cudaGetLastError();      \
      STANDALONE_WARN("CUDA warning: ", cudaGetErrorString(__err)); \
    }                                                               \
  } while (0)

#endif
