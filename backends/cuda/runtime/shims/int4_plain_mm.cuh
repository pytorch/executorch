/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// W4A8 dp4a matvec for INT4 decode (M <= 4).
//
// Reads plain nibble-packed [N, K//2] weights (Int4Tensor format).
// Scale/zero layout: [K//gs, N] (Int4Tensor's native layout).
//
// Dynamically quantizes bf16 activations to INT8 (per-32-element blocks),
// then uses dp4a for fused int4×int8 dot products with 16-byte vectorized
// loads and warp-cooperative quantization.

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

constexpr int32_t MV_NWARPS = 8;
constexpr int32_t MV_WARP_SIZE = 32;
constexpr int32_t MV_THREADS = MV_NWARPS * MV_WARP_SIZE;
constexpr int32_t Q8_BLOCK_SIZE = 32;

__host__ __forceinline__ int32_t log2_pow2(int32_t v) {
  int32_t r = 0;
  while (v > 1) {
    v >>= 1;
    r++;
  }
  return r;
}

// ---------------------------------------------------------------------------
// Activation quantization: bf16 → int8 (warp-cooperative, per-32-element blocks)
// ---------------------------------------------------------------------------

struct Q8Block {
  int8_t qs_even[Q8_BLOCK_SIZE / 2];
  int8_t qs_odd[Q8_BLOCK_SIZE / 2];
  float d; // scale
};

__global__ void quantize_activations_q8_kernel(
    const __nv_bfloat16* __restrict__ A,
    Q8Block* __restrict__ q8,
    int32_t K) {
  const int32_t m = blockIdx.y;
  const int32_t block_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t n_blocks = K / Q8_BLOCK_SIZE;
  if (block_id >= n_blocks)
    return;

  const int32_t lane = threadIdx.x;
  const __nv_bfloat16* src =
      A + static_cast<int64_t>(m) * K + block_id * Q8_BLOCK_SIZE;
  Q8Block* dst = q8 + static_cast<int64_t>(m) * n_blocks + block_id;

  float val = __bfloat162float(src[lane]);

  float amax = fabsf(val);
  for (int offset = 16; offset > 0; offset >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));

  float d = amax / 127.0f;
  float id = (d > 0.0f) ? 1.0f / d : 0.0f;
  int32_t q = __float2int_rn(val * id);
  q = max(-128, min(127, q));

  if (lane % 2 == 0)
    dst->qs_even[lane / 2] = static_cast<int8_t>(q);
  else
    dst->qs_odd[lane / 2] = static_cast<int8_t>(q);

  if (lane == 0) {
    dst->d = d;
  }
}

// ---------------------------------------------------------------------------
// W4A8 dp4a matvec kernel
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(MV_THREADS)
    int4_w4a8_matvec_kernel(
        const uint8_t* __restrict__ qdata,
        const __nv_bfloat16* __restrict__ w_scale,
        const __nv_bfloat16* __restrict__ w_zero,
        const Q8Block* __restrict__ q8,
        __nv_bfloat16* __restrict__ out,
        int32_t N,
        int32_t K,
        int32_t gs_shift) {
  const int32_t n = blockIdx.x * MV_NWARPS + threadIdx.y;
  const int32_t m = blockIdx.y;
  if (n >= N)
    return;

  const int32_t K_half = K / 2;
  const int32_t lane_id = threadIdx.x;
  const int32_t n_q8_blocks = K / Q8_BLOCK_SIZE;

  const uint8_t* qrow = qdata + static_cast<int64_t>(n) * K_half;
  const __nv_bfloat16* scale_base = w_scale + n;
  const __nv_bfloat16* zero_base = w_zero + n;
  const int32_t scale_stride = N;
  const Q8Block* q8_row = q8 + static_cast<int64_t>(m) * n_q8_blocks;

  const uint4* qrow16 = reinterpret_cast<const uint4*>(qrow);
  const int32_t K_half_16 = K_half / 16;

  float sum = 0.0f;

  int32_t prev_g = -1;
  float ws = 0.0f, wz = 0.0f;

  for (int32_t i = lane_id; i < K_half_16; i += MV_WARP_SIZE) {
    uint4 packed16 = __ldg(&qrow16[i]);
    int32_t k_base = i * 32;
    uint32_t words[4] = {packed16.x, packed16.y, packed16.z, packed16.w};

#pragma unroll
    for (int32_t w = 0; w < 4; w++) {
      uint32_t packed = words[w];
      int32_t k_word = k_base + w * 8;
      int32_t g = k_word >> gs_shift;

      if (g != prev_g) {
        ws = __bfloat162float(__ldg(&scale_base[g * scale_stride]));
        wz = __bfloat162float(__ldg(&zero_base[g * scale_stride]));
        prev_g = g;
      }

      int32_t vi_lo = packed & 0x0F0F0F0F;
      int32_t vi_hi = (packed >> 4) & 0x0F0F0F0F;

      int32_t q8_block_idx = k_word / Q8_BLOCK_SIZE;
      int32_t q8_half_offset = (k_word % Q8_BLOCK_SIZE) / 2;
      const Q8Block* qb = &q8_row[q8_block_idx];

      int32_t a_even = *reinterpret_cast<const int32_t*>(
          qb->qs_even + q8_half_offset);
      int32_t a_odd = *reinterpret_cast<const int32_t*>(
          qb->qs_odd + q8_half_offset);

      int32_t dp = __dp4a(vi_lo, a_even, 0);
      dp = __dp4a(vi_hi, a_odd, dp);

      float a_scale = qb->d;

      int32_t a_sum8 = __dp4a(0x01010101, a_even, 0);
      a_sum8 = __dp4a(0x01010101, a_odd, a_sum8);

      sum += ws * a_scale *
          (static_cast<float>(dp) - wz * static_cast<float>(a_sum8));
    }
  }

  for (int offset = MV_WARP_SIZE / 2; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);

  if (lane_id == 0)
    out[static_cast<int64_t>(m) * N + n] = __float2bfloat16(sum);
}

// ---------------------------------------------------------------------------
// Persistent Q8 buffer (lazy init, not thread-safe — single-stream only)
// ---------------------------------------------------------------------------

static Q8Block* g_q8_buf = nullptr;
static size_t g_q8_buf_size = 0;

static Q8Block* get_q8_buffer(size_t needed) {
  if (g_q8_buf_size < needed) {
    if (g_q8_buf)
      cudaFree(g_q8_buf);
    cudaError_t err = cudaMalloc(&g_q8_buf, needed);
    ET_CHECK_MSG(
        err == cudaSuccess,
        "cudaMalloc failed for Q8 buffer: %s",
        cudaGetErrorString(err));
    g_q8_buf_size = needed;
  }
  return g_q8_buf;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

void _int4_plain_mm_cuda(
    const Tensor& A, // [M, K] bf16
    const Tensor& qdata, // [N, K//2] uint8
    const Tensor& scale, // [K//gs, N] bf16
    const Tensor& zero, // [K//gs, N] bf16
    int64_t group_size,
    Tensor* output) { // [M, N] bf16, pre-allocated
  int32_t M = A.size(0);
  int32_t K = A.size(1);
  int32_t N = qdata.size(0);

  ET_CHECK(A.dtype() == c10::ScalarType::BFloat16);
  ET_CHECK(
      qdata.dtype() == c10::ScalarType::Byte ||
      qdata.dtype() == c10::ScalarType::Char);
  ET_CHECK(scale.dtype() == c10::ScalarType::BFloat16);
  ET_CHECK(zero.dtype() == c10::ScalarType::BFloat16);
  ET_CHECK(A.dim() == 2);
  ET_CHECK(qdata.dim() == 2);
  ET_CHECK(qdata.size(1) == K / 2);
  ET_CHECK(scale.dim() == 2);
  ET_CHECK(scale.size(1) == N);
  ET_CHECK(zero.dim() == 2);
  ET_CHECK(zero.size(1) == N);

  int32_t gs = static_cast<int32_t>(group_size);
  ET_CHECK_MSG(
      gs > 0 && (gs & (gs - 1)) == 0,
      "group_size=%d must be a power of 2",
      gs);
  ET_CHECK_MSG(
      K >= Q8_BLOCK_SIZE && K % Q8_BLOCK_SIZE == 0,
      "K=%d must be a positive multiple of %d for dp4a kernel",
      K,
      Q8_BLOCK_SIZE);

  auto stream_result = getCurrentCUDAStream(0);
  ET_CHECK_MSG(stream_result.ok(), "Failed to get CUDA stream");
  cudaStream_t stream = stream_result.get();

  int32_t gs_shift = log2_pow2(gs);

  // Quantize activations to INT8
  int32_t n_q8_blocks = K / Q8_BLOCK_SIZE;
  size_t q8_bytes = static_cast<size_t>(M) * n_q8_blocks * sizeof(Q8Block);
  Q8Block* q8_buf = get_q8_buffer(q8_bytes);

  constexpr int32_t Q8_WARPS = 8;
  int32_t blocks_per_m = (n_q8_blocks + Q8_WARPS - 1) / Q8_WARPS;
  dim3 q8_grid(blocks_per_m, M);
  dim3 q8_block(MV_WARP_SIZE, Q8_WARPS);
  quantize_activations_q8_kernel<<<q8_grid, q8_block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      q8_buf,
      K);

  // dp4a matvec
  dim3 grid((N + MV_NWARPS - 1) / MV_NWARPS, M);
  dim3 block(MV_WARP_SIZE, MV_NWARPS);
  int4_w4a8_matvec_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(qdata.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(zero.data_ptr()),
      q8_buf,
      reinterpret_cast<__nv_bfloat16*>(output->data_ptr()),
      N, K, gs_shift);
}

} // namespace executorch::backends::cuda
