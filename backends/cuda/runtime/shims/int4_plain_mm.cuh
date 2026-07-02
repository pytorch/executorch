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
// Scale/zero layout: [N, K//gs] uint8 codes (transposed AOT for coalesced
// loads), with a per-row [N, 2] bf16 super-scale (scale_step, zero_step):
// scale = code * scale_step, zero = code * zero_step. This halves the
// per-group metadata vs bf16 scale+zero (5.0 -> 4.5 bpw) at ~baseline
// accuracy, since Q4_K group scales fit an 8-bit per-row-normalized code.
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
// Activation quantization: bf16 → int8 (warp-cooperative, per-32-element
// blocks)
// ---------------------------------------------------------------------------

// alignas(16) pads sizeof(Q8Block) to 48 so each block (and its qs_even/qs_odd
// 16-byte halves) is 16-byte aligned. This lets the matvec load a whole block's
// int8 activations with two vectorized uint4 loads instead of eight scalar
// int32 loads, cutting activation load instructions ~4x.
struct alignas(16) Q8Block {
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
// Coalesced-scale W4A8 dp4a matvec
//
// Reads scale/zero in the transposed [N, n_groups] layout (transposed AOT at
// export time) as uint8 codes, decoded with a per-row bf16 super-scale
// (scale = code*scale_step, zero = code*zero_step). With group_size >= 32, one
// uint4 (32 weights) maps to exactly one activation block and one weight
// group, so within a warp the 32 lanes touch 32 consecutive groups. In
// [N, n_groups] layout those 32 group codes are contiguous => a single
// coalesced load, vs 32 stride-N cache lines in the native layout. For the
// gemma group_size=32 weights this is the dominant decode-matvec cost.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(MV_THREADS)
    int4_w4a8_matvec_coalesced_kernel(
        const uint8_t* __restrict__ qdata,
        const uint8_t* __restrict__ w_scale_t, // [N, n_groups] uint8 codes
        const uint8_t* __restrict__ w_zero_t, // [N, n_groups] uint8 codes
        const __nv_bfloat16* __restrict__ w_steps, // [N, 2] (scale_step, zero_step)
        const Q8Block* __restrict__ q8,
        __nv_bfloat16* __restrict__ out,
        int32_t N,
        int32_t K,
        int32_t gs_shift,
        int32_t n_groups) {
  const int32_t n = blockIdx.x * MV_NWARPS + threadIdx.y;
  const int32_t m = blockIdx.y;
  if (n >= N)
    return;

  const int32_t K_half = K / 2;
  const int32_t lane_id = threadIdx.x;
  const int32_t n_q8_blocks = K / Q8_BLOCK_SIZE;

  const uint8_t* qrow = qdata + static_cast<int64_t>(n) * K_half;
  const uint8_t* scale_row = w_scale_t + static_cast<int64_t>(n) * n_groups;
  const uint8_t* zero_row = w_zero_t + static_cast<int64_t>(n) * n_groups;
  // Per-row super-scales: the int8 group codes are decoded as
  // scale = code * scale_step, zero = code * zero_step. scale_step is a
  // per-row constant (it factors out of the dp4a sum), so the dp4a dot
  // products below are bit-identical to the bf16-metadata kernel.
  const float scale_step =
      __bfloat162float(__ldg(&w_steps[static_cast<int64_t>(n) * 2 + 0]));
  const float zero_step =
      __bfloat162float(__ldg(&w_steps[static_cast<int64_t>(n) * 2 + 1]));
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

    // One uint4 (32 weights) maps to exactly one Q8 activation block (32
    // activations), i.e. q8_block_idx == i. Load the whole block with two
    // vectorized uint4 loads (+ one scale load) instead of eight scalar int32
    // loads. ae.{x,y,z,w} == qs_even[0:4],[4:8],[8:12],[12:16] == a_even for
    // w=0..3 (same for ao/qs_odd) -> bit-identical to the scalar path.
    const Q8Block* qb = &q8_row[i];
    uint4 ae = *reinterpret_cast<const uint4*>(qb->qs_even);
    uint4 ao = *reinterpret_cast<const uint4*>(qb->qs_odd);
    float a_scale = qb->d;
    const uint32_t a_even[4] = {ae.x, ae.y, ae.z, ae.w};
    const uint32_t a_odd[4] = {ao.x, ao.y, ao.z, ao.w};

#pragma unroll
    for (int32_t w = 0; w < 4; w++) {
      uint32_t packed = words[w];
      int32_t k_word = k_base + w * 8;
      int32_t g = k_word >> gs_shift;

      if (g != prev_g) {
        ws = static_cast<float>(__ldg(&scale_row[g])) * scale_step;
        wz = static_cast<float>(__ldg(&zero_row[g])) * zero_step;
        prev_g = g;
      }

      int32_t vi_lo = packed & 0x0F0F0F0F;
      int32_t vi_hi = (packed >> 4) & 0x0F0F0F0F;

      int32_t dp = __dp4a(vi_lo, static_cast<int32_t>(a_even[w]), 0);
      dp = __dp4a(vi_hi, static_cast<int32_t>(a_odd[w]), dp);

      int32_t a_sum8 = __dp4a(0x01010101, static_cast<int32_t>(a_even[w]), 0);
      a_sum8 = __dp4a(0x01010101, static_cast<int32_t>(a_odd[w]), a_sum8);

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
// Persistent Q8 buffer (lazy init, not thread-safe — single-stream only).
// Freed at process exit via a static guard so leak detectors stay quiet; the
// CUDA runtime would otherwise reclaim it on teardown anyway.
// ---------------------------------------------------------------------------

static Q8Block* g_q8_buf = nullptr;
static size_t g_q8_buf_size = 0;

namespace {
struct Q8BufferGuard {
  ~Q8BufferGuard() {
    if (g_q8_buf) {
      // Ignore errors: during process teardown the CUDA context may already be
      // gone (cudaErrorCudartUnloading), which is harmless here.
      cudaFree(g_q8_buf);
      g_q8_buf = nullptr;
      g_q8_buf_size = 0;
    }
  }
};
Q8BufferGuard g_q8_buf_guard;
} // namespace

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
    const Tensor& scale, // [N, K//gs] uint8 codes
    const Tensor& zero, // [N, K//gs] uint8 codes
    const Tensor& steps, // [N, 2] bf16 (scale_step, zero_step)
    int64_t group_size,
    Tensor* output) { // [M, N] bf16, pre-allocated
  int32_t M = A.size(0);
  int32_t K = A.size(1);
  int32_t N = qdata.size(0);

  ET_CHECK(A.dtype() == c10::ScalarType::BFloat16);
  ET_CHECK(
      qdata.dtype() == c10::ScalarType::Byte ||
      qdata.dtype() == c10::ScalarType::Char);
  ET_CHECK(
      scale.dtype() == c10::ScalarType::Byte ||
      scale.dtype() == c10::ScalarType::Char);
  ET_CHECK(
      zero.dtype() == c10::ScalarType::Byte ||
      zero.dtype() == c10::ScalarType::Char);
  ET_CHECK(steps.dtype() == c10::ScalarType::BFloat16);
  ET_CHECK(A.dim() == 2);
  ET_CHECK(qdata.dim() == 2);
  ET_CHECK(qdata.size(1) == K / 2);
  ET_CHECK(scale.dim() == 2);
  ET_CHECK(scale.size(0) == N);
  ET_CHECK(zero.dim() == 2);
  ET_CHECK(zero.size(0) == N);
  ET_CHECK(steps.dim() == 2);
  ET_CHECK(steps.size(0) == N);
  ET_CHECK(steps.size(1) == 2);

  int32_t gs = static_cast<int32_t>(group_size);
  ET_CHECK_MSG(
      gs > 0 && (gs & (gs - 1)) == 0, "group_size=%d must be a power of 2", gs);
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
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()), q8_buf, K);

  // dp4a matvec
  dim3 grid((N + MV_NWARPS - 1) / MV_NWARPS, M);
  dim3 block(MV_WARP_SIZE, MV_NWARPS);

  int32_t n_groups = static_cast<int32_t>(scale.size(1));
  int4_w4a8_matvec_coalesced_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(qdata.data_ptr()),
      reinterpret_cast<const uint8_t*>(scale.data_ptr()),
      reinterpret_cast<const uint8_t*>(zero.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(steps.data_ptr()),
      q8_buf,
      reinterpret_cast<__nv_bfloat16*>(output->data_ptr()),
      N, K, gs_shift, n_groups);
}

} // namespace executorch::backends::cuda
