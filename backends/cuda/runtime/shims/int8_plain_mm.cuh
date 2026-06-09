/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// W8A8 dp4a matvec for INT8 decode (M <= 4).
//
// Reads plain (unpacked) [N, K] int8 weights (IntxUnpackedToInt8Tensor format).
// Scale layout: [N, K//gs] bf16, zero layout: [N, K//gs] int8 (row-major).
//
// Dynamically quantizes bf16 activations to INT8 (per-32-element blocks,
// natural order), then uses dp4a for fused int8×int8 dot products with 16-byte
// vectorized weight loads and warp-cooperative quantization.
//
// Symbol names are suffixed _i8 / distinct from int4_plain_mm.cuh so both
// translation units can be linked together without ODR conflicts.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/utils.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::Tensor;
namespace c10 = executorch::backends::aoti::slim::c10;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr int32_t MV8_NWARPS = 8;
constexpr int32_t MV8_WARP_SIZE = 32;
constexpr int32_t MV8_THREADS = MV8_NWARPS * MV8_WARP_SIZE;
constexpr int32_t Q8_NAT_BLOCK_SIZE = 32;

__host__ __forceinline__ int32_t log2_pow2_i8(int32_t v) {
  int32_t r = 0;
  while (v > 1) {
    v >>= 1;
    r++;
  }
  return r;
}

// ---------------------------------------------------------------------------
// Activation quantization: bf16 → int8 (warp-cooperative, per-32-element
// blocks, NATURAL order — qs[k] holds the quantized value for element k).
// ---------------------------------------------------------------------------

struct Q8BlockNat {
  int8_t qs[Q8_NAT_BLOCK_SIZE];
  float d; // scale
};

__global__ void quantize_activations_q8_natural_kernel(
    const __nv_bfloat16* __restrict__ A,
    Q8BlockNat* __restrict__ q8,
    int32_t K) {
  const int32_t m = blockIdx.y;
  const int32_t block_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t n_blocks = K / Q8_NAT_BLOCK_SIZE;
  if (block_id >= n_blocks)
    return;

  const int32_t lane = threadIdx.x;
  const __nv_bfloat16* src =
      A + static_cast<int64_t>(m) * K + block_id * Q8_NAT_BLOCK_SIZE;
  Q8BlockNat* dst = q8 + static_cast<int64_t>(m) * n_blocks + block_id;

  float val = __bfloat162float(src[lane]);

  float amax = fabsf(val);
  for (int offset = 16; offset > 0; offset >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));

  float d = amax / 127.0f;
  float id = (d > 0.0f) ? 1.0f / d : 0.0f;
  int32_t q = __float2int_rn(val * id);
  q = max(-128, min(127, q));

  dst->qs[lane] = static_cast<int8_t>(q);
  if (lane == 0)
    dst->d = d;
}

// ---------------------------------------------------------------------------
// W8A8 dp4a matvec kernel
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(MV8_THREADS) int8_w8a8_matvec_kernel(
    const int8_t* __restrict__ qdata, // [N, K]
    const __nv_bfloat16* __restrict__ w_scale, // [N, K//gs]
    const int8_t* __restrict__ w_zero, // [N, K//gs]
    const Q8BlockNat* __restrict__ q8,
    __nv_bfloat16* __restrict__ out,
    int32_t N,
    int32_t K,
    int32_t n_groups,
    int32_t gs_shift) {
  const int32_t n = blockIdx.x * MV8_NWARPS + threadIdx.y;
  const int32_t m = blockIdx.y;
  if (n >= N)
    return;

  const int32_t lane_id = threadIdx.x;
  const int32_t n_q8_blocks = K / Q8_NAT_BLOCK_SIZE;

  const int8_t* qrow = qdata + static_cast<int64_t>(n) * K;
  const __nv_bfloat16* scale_row = w_scale + static_cast<int64_t>(n) * n_groups;
  const int8_t* zero_row = w_zero + static_cast<int64_t>(n) * n_groups;
  const Q8BlockNat* q8_row = q8 + static_cast<int64_t>(m) * n_q8_blocks;

  // Vectorized 16-byte loads: 16 int8 weights (4 int32 words) per uint4.
  const uint4* qrow16 = reinterpret_cast<const uint4*>(qrow);
  const int32_t K_16 = K / 16;

  float sum = 0.0f;

  int32_t prev_g = -1;
  float ws = 0.0f, wz = 0.0f;

  for (int32_t i = lane_id; i < K_16; i += MV8_WARP_SIZE) {
    uint4 packed16 = __ldg(&qrow16[i]);
    int32_t k_base = i * 16;
    uint32_t words[4] = {packed16.x, packed16.y, packed16.z, packed16.w};

#pragma unroll
    for (int32_t w = 0; w < 4; w++) {
      int32_t k_word = k_base + w * 4; // 4 int8 weights start here
      int32_t g = k_word >> gs_shift;

      if (g != prev_g) {
        ws = __bfloat162float(__ldg(&scale_row[g]));
        wz = static_cast<float>(__ldg(&zero_row[g]));
        prev_g = g;
      }

      int32_t w_word = static_cast<int32_t>(words[w]);

      int32_t q8_block_idx = k_word / Q8_NAT_BLOCK_SIZE;
      int32_t q8_offset = k_word % Q8_NAT_BLOCK_SIZE;
      const Q8BlockNat* qb = &q8_row[q8_block_idx];
      int32_t a_word = *reinterpret_cast<const int32_t*>(qb->qs + q8_offset);

      int32_t dp = __dp4a(w_word, a_word, 0);
      int32_t a_sum = __dp4a(0x01010101, a_word, 0);
      float a_scale = qb->d;

      sum += ws * a_scale *
          (static_cast<float>(dp) - wz * static_cast<float>(a_sum));
    }
  }

  for (int offset = MV8_WARP_SIZE / 2; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);

  if (lane_id == 0)
    out[static_cast<int64_t>(m) * N + n] = __float2bfloat16(sum);
}

// ---------------------------------------------------------------------------
// Persistent Q8 buffer (lazy init, not thread-safe — single-stream only).
// Freed at process exit via a static guard so leak detectors stay quiet; the
// CUDA runtime would otherwise reclaim it on teardown anyway.
// ---------------------------------------------------------------------------

static Q8BlockNat* g_q8_buf_i8 = nullptr;
static size_t g_q8_buf_i8_size = 0;

namespace {
struct Q8BufferGuardI8 {
  ~Q8BufferGuardI8() {
    if (g_q8_buf_i8) {
      // Ignore errors: during process teardown the CUDA context may already be
      // gone (cudaErrorCudartUnloading), which is harmless here.
      cudaFree(g_q8_buf_i8);
      g_q8_buf_i8 = nullptr;
      g_q8_buf_i8_size = 0;
    }
  }
};
Q8BufferGuardI8 g_q8_buf_i8_guard;
} // namespace

static Q8BlockNat* get_q8_buffer_i8(size_t needed) {
  if (g_q8_buf_i8_size < needed) {
    if (g_q8_buf_i8)
      cudaFree(g_q8_buf_i8);
    cudaError_t err = cudaMalloc(&g_q8_buf_i8, needed);
    ET_CHECK_MSG(
        err == cudaSuccess,
        "cudaMalloc failed for Q8 buffer (int8): %s",
        cudaGetErrorString(err));
    g_q8_buf_i8_size = needed;
  }
  return g_q8_buf_i8;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

inline void _int8_plain_mm_cuda(
    const Tensor& A, // [M, K] bf16
    const Tensor& qdata, // [N, K] int8
    const Tensor& scale, // [N, K//gs] bf16
    const Tensor& zero, // [N, K//gs] int8
    int64_t group_size,
    Tensor* output) { // [M, N] bf16, pre-allocated
  int32_t M = A.size(0);
  int32_t K = A.size(1);
  int32_t N = qdata.size(0);

  ET_CHECK(A.dtype() == c10::ScalarType::BFloat16);
  ET_CHECK(qdata.dtype() == c10::ScalarType::Char);
  ET_CHECK(scale.dtype() == c10::ScalarType::BFloat16);
  ET_CHECK(zero.dtype() == c10::ScalarType::Char);
  ET_CHECK(A.dim() == 2);
  ET_CHECK(qdata.dim() == 2);
  ET_CHECK(qdata.size(1) == K);
  ET_CHECK(scale.dim() == 2);
  ET_CHECK(scale.size(0) == N);
  ET_CHECK(zero.dim() == 2);
  ET_CHECK(zero.size(0) == N);

  int32_t gs = static_cast<int32_t>(group_size);
  ET_CHECK_MSG(
      gs > 0 && (gs & (gs - 1)) == 0, "group_size=%d must be a power of 2", gs);
  ET_CHECK_MSG(
      gs % Q8_NAT_BLOCK_SIZE == 0,
      "group_size=%d must be a multiple of %d",
      gs,
      Q8_NAT_BLOCK_SIZE);
  ET_CHECK_MSG(
      K >= Q8_NAT_BLOCK_SIZE && K % Q8_NAT_BLOCK_SIZE == 0,
      "K=%d must be a positive multiple of %d for dp4a int8 kernel",
      K,
      Q8_NAT_BLOCK_SIZE);

  int32_t n_groups = K / gs;

  auto stream_result = getCurrentCUDAStream(0);
  ET_CHECK_MSG(stream_result.ok(), "Failed to get CUDA stream");
  cudaStream_t stream = stream_result.get();

  int32_t gs_shift = log2_pow2_i8(gs);

  // Quantize activations to INT8 (natural order)
  int32_t n_q8_blocks = K / Q8_NAT_BLOCK_SIZE;
  size_t q8_bytes = static_cast<size_t>(M) * n_q8_blocks * sizeof(Q8BlockNat);
  Q8BlockNat* q8_buf = get_q8_buffer_i8(q8_bytes);

  constexpr int32_t Q8_WARPS = 8;
  int32_t blocks_per_m = (n_q8_blocks + Q8_WARPS - 1) / Q8_WARPS;
  dim3 q8_grid(blocks_per_m, M);
  dim3 q8_block(MV8_WARP_SIZE, Q8_WARPS);
  quantize_activations_q8_natural_kernel<<<q8_grid, q8_block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()), q8_buf, K);

  // dp4a matvec
  dim3 grid((N + MV8_NWARPS - 1) / MV8_NWARPS, M);
  dim3 block(MV8_WARP_SIZE, MV8_NWARPS);
  int8_w8a8_matvec_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const int8_t*>(qdata.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr()),
      reinterpret_cast<const int8_t*>(zero.data_ptr()),
      q8_buf,
      reinterpret_cast<__nv_bfloat16*>(output->data_ptr()),
      N,
      K,
      n_groups,
      gs_shift);
}

} // namespace executorch::backends::cuda
