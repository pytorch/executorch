/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace executorch {
namespace extension {
namespace llm {
namespace cuda {

struct ArgMaxPair {
  float v;
  int i;
};

// tie-break: smaller index wins on equal values
__device__ __forceinline__ ArgMaxPair better(ArgMaxPair a, ArgMaxPair b) {
  if (b.v > a.v)
    return b;
  if (b.v < a.v)
    return a;
  return (b.i < a.i) ? b : a;
}

__device__ __forceinline__ ArgMaxPair
warp_argmax_xor(ArgMaxPair x, unsigned mask = 0xffffffffu) {
  for (int d = 16; d > 0; d >>= 1) {
    ArgMaxPair y;
    y.v = __shfl_xor_sync(mask, x.v, d);
    y.i = __shfl_xor_sync(mask, x.i, d);
    x = better(x, y);
  }
  return x;
}

// ---- dtype -> float load helpers ----
template <typename T>
__device__ __forceinline__ float load_as_float(const T* p);

template <>
__device__ __forceinline__ float load_as_float<float>(const float* p) {
  return *p;
}

template <>
__device__ __forceinline__ float load_as_float<half>(const half* p) {
  return __half2float(*p);
}

template <>
__device__ __forceinline__ float
load_as_float<nv_bfloat16>(const nv_bfloat16* p) {
  return __bfloat162float(*p);
}

// logits: [rows, vocab] row-major contiguous
// out_token: [rows]
// out_maxlogit: [rows] (optional; pass nullptr if not needed)
template <typename T>
__global__ void argmax_vocab_rows_kernel(
    const T* __restrict__ logits,
    int rows,
    int vocab,
    int* __restrict__ out_token,
    float* __restrict__ out_maxlogit) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warps_per_block = (blockDim.x + 31) >> 5;

  const T* row_ptr = logits + (size_t)row * (size_t)vocab;

  // local scan
  ArgMaxPair best;
  best.v = -FLT_MAX;
  best.i = -1;

  for (int j = tid; j < vocab; j += blockDim.x) {
    float v = load_as_float<T>(row_ptr + j);
    best = better(best, ArgMaxPair{v, j});
  }

  // warp reduce
  best = warp_argmax_xor(best);

  // shared collect warp winners (dynamic size based on warps_per_block)
  extern __shared__ char smem[];
  float* s_val = reinterpret_cast<float*>(smem);
  int* s_idx = reinterpret_cast<int*>(s_val + warps_per_block);

  if (lane == 0) {
    s_val[warp] = best.v;
    s_idx[warp] = best.i;
  }
  __syncthreads();

  // first warp reduces warp winners
  if (warp == 0) {
    ArgMaxPair wbest;
    if (lane < warps_per_block) {
      wbest.v = s_val[lane];
      wbest.i = s_idx[lane];
    } else {
      wbest.v = -FLT_MAX;
      wbest.i = -1;
    }

    wbest = warp_argmax_xor(wbest);

    if (lane == 0) {
      out_token[row] = wbest.i;
      if (out_maxlogit)
        out_maxlogit[row] = wbest.v;
    }
  }
}

inline void launch_argmax_vocab_rows(
    const void* logits,
    ::executorch::aten::ScalarType scalar_type,
    int rows,
    int vocab,
    int* out_token,
    float* out_maxlogit,
    cudaStream_t stream) {
  // 256 threads (8 warps) is the standard choice for reduction kernels:
  // good occupancy, low shared memory, works well across all architectures.
  constexpr int kThreadsPerBlock = 256;
  constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

  dim3 block(kThreadsPerBlock);
  dim3 grid(rows);

  // Calculate shared memory size: one float and one int per warp
  constexpr size_t smem_size = kWarpsPerBlock * (sizeof(float) + sizeof(int));

  switch (scalar_type) {
    case ::executorch::aten::ScalarType::Float:
      argmax_vocab_rows_kernel<float><<<grid, block, smem_size, stream>>>(
          (const float*)logits, rows, vocab, out_token, out_maxlogit);
      break;
    case ::executorch::aten::ScalarType::Half:
      argmax_vocab_rows_kernel<half><<<grid, block, smem_size, stream>>>(
          (const half*)logits, rows, vocab, out_token, out_maxlogit);
      break;
    case ::executorch::aten::ScalarType::BFloat16:
      argmax_vocab_rows_kernel<nv_bfloat16><<<grid, block, smem_size, stream>>>(
          (const nv_bfloat16*)logits, rows, vocab, out_token, out_maxlogit);
      break;
    default:
      // Unsupported type, fall back to float
      argmax_vocab_rows_kernel<float><<<grid, block, smem_size, stream>>>(
          (const float*)logits, rows, vocab, out_token, out_maxlogit);
      break;
  }
}

// Wrapper function that performs argmax on GPU logits tensor (single row).
// Returns the token index with the highest logit value.
// logits_ptr: pointer to GPU memory containing logits
// vocab_size: vocabulary size
// scalar_type: data type of the logits tensor
// cuda_stream: CUDA stream for async execution (nullptr for default stream)
// out_token_gpu: pre-allocated GPU memory for output token (int*)
int32_t argmax_cuda(
    const void* logits_ptr,
    int vocab_size,
    ::executorch::aten::ScalarType scalar_type,
    cudaStream_t cuda_stream,
    int* out_token_gpu);

} // namespace cuda
} // namespace llm
} // namespace extension
} // namespace executorch

