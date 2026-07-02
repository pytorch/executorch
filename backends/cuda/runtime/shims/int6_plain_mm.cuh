/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// W6A8 dp4a matvec for packed INT6 decode (M <= 4), used for GGUF Q6_K weights.
//
// Reads a genuine 6-bit packed weight (CudaDp4aPlanarInt6Tensor format), split
// into two planes:
//   ql    : [N, K/2] uint8 — low-nibble plane, nibble-packed even/odd exactly
//           like the INT4 path (ql[:,j] = lo[:,2j] | (lo[:,2j+1] << 4)).
//   qh    : [N, K/4] uint8 — high-2-bit plane, 4 values/byte, arranged per
//           32-weight chunk as hi_even_packed[4] then hi_odd_packed[4] (each
//           byte holds the four 2-bit highs of one dp4a word in even/odd
//           order).
//   scale : [N, K/gs] bf16 — per-group scales, row-major (coalesced; no zero).
// The stored 6-bit value is u = q + 32 in [0, 63] (q in [-32, 31]); the
// constant -32 offset is applied in the kernel, so Q6_K's symmetry means NO
// zero tensor.
//
// Dynamically quantizes bf16 activations to INT8 (per-32-element blocks,
// even/odd order, identical to the INT4 path), reconstructs full 6-bit weight
// bytes per dp4a word (vfull = vi_lo | (spread2(hi_byte) << 4)), and uses dp4a
// for fused int6xint8 dot products with vectorized weight loads and
// warp-cooperative quantization.
//
// Symbol names are suffixed _i6 / distinct from int4_plain_mm.cuh and
// int8_plain_mm.cuh so all three translation units can be linked together
// without ODR conflicts.

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

constexpr int32_t MV6_NWARPS = 8;
constexpr int32_t MV6_WARP_SIZE = 32;
constexpr int32_t MV6_THREADS = MV6_NWARPS * MV6_WARP_SIZE;
constexpr int32_t Q8_BLOCK_SIZE_I6 = 32;

__host__ __forceinline__ int32_t log2_pow2_i6(int32_t v) {
  int32_t r = 0;
  while (v > 1) {
    v >>= 1;
    r++;
  }
  return r;
}

// Expand a byte's four 2-bit fields into four byte lanes (each in bits 0-1):
//   in  : b = [.. b7 b6 | b5 b4 | b3 b2 | b1 b0]
//   out : lane0=[b1 b0], lane1=[b3 b2], lane2=[b5 b4], lane3=[b7 b6]
// ~6 ALU ops; verified by truth-table. Used to place the high 2 bits of each
// weight into bits 4-5 of the corresponding dp4a byte lane.
__device__ __forceinline__ uint32_t spread2_i6(uint32_t b) {
  uint32_t t = (b | (b << 12)) & 0x000F000F;
  uint32_t r = (t | (t << 6)) & 0x03030303;
  return r;
}

// ---------------------------------------------------------------------------
// Activation quantization: bf16 -> int8 (warp-cooperative, per-32-element
// blocks, EVEN/ODD order — identical to the INT4 path's Q8Block).
// ---------------------------------------------------------------------------

// alignas(16) pads sizeof(Q8Block_i6) to 48 so each block (and its
// qs_even/qs_odd 16-byte halves) is 16-byte aligned, allowing two vectorized
// uint4 loads of a block's int8 activations instead of eight scalar int32
// loads.
struct alignas(16) Q8Block_i6 {
  int8_t qs_even[Q8_BLOCK_SIZE_I6 / 2];
  int8_t qs_odd[Q8_BLOCK_SIZE_I6 / 2];
  float d; // scale
};

__global__ void quantize_activations_q8_i6_kernel(
    const __nv_bfloat16* __restrict__ A,
    Q8Block_i6* __restrict__ q8,
    int32_t K) {
  const int32_t m = blockIdx.y;
  const int32_t block_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t n_blocks = K / Q8_BLOCK_SIZE_I6;
  if (block_id >= n_blocks)
    return;

  const int32_t lane = threadIdx.x;
  const __nv_bfloat16* src =
      A + static_cast<int64_t>(m) * K + block_id * Q8_BLOCK_SIZE_I6;
  Q8Block_i6* dst = q8 + static_cast<int64_t>(m) * n_blocks + block_id;

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

  if (lane == 0)
    dst->d = d;
}

// ---------------------------------------------------------------------------
// W6A8 dp4a matvec kernel
//
// dp4a is linear, so reconstructing v = lo + (hi<<4) and dotting once is
// equivalent to two separate dp4a passes. We reconstruct the full 6-bit byte
// (vfull = vi_lo | (spread2(hi_byte) << 4)) so a single dp4a per even/odd half
// covers the whole weight. The per-group zero is the constant 32 (in u-space),
// applied as out += scale * a_scale * (dp - 32 * a_sum) — no zero load.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(MV6_THREADS) int6_w6a8_matvec_kernel(
    const uint8_t* __restrict__ ql, // [N, K/2]
    const uint8_t* __restrict__ qh, // [N, K/4]
    const int8_t* __restrict__ w_scale, // [N, n_groups] int8 codes
    const __nv_bfloat16* __restrict__ w_steps, // [N, 1] per-row scale_step
    const Q8Block_i6* __restrict__ q8,
    __nv_bfloat16* __restrict__ out,
    int32_t N,
    int32_t K,
    int32_t gs_shift,
    int32_t n_groups) {
  const int32_t n = blockIdx.x * MV6_NWARPS + threadIdx.y;
  const int32_t m = blockIdx.y;
  if (n >= N)
    return;

  const int32_t K_half = K / 2;
  const int32_t K_quarter = K / 4;
  const int32_t lane_id = threadIdx.x;
  const int32_t n_q8_blocks = K / Q8_BLOCK_SIZE_I6;

  const uint8_t* qlrow = ql + static_cast<int64_t>(n) * K_half;
  const uint8_t* qhrow = qh + static_cast<int64_t>(n) * K_quarter;
  const int8_t* scale_row = w_scale + static_cast<int64_t>(n) * n_groups;
  // Per-row super-scale: int8 group codes decode as scale = code * scale_step.
  // scale_step is a per-row constant (it factors out of the dp4a sum), so the
  // dp4a dot products below are bit-identical to the bf16-metadata kernel.
  const float scale_step = __bfloat162float(__ldg(&w_steps[n]));
  const Q8Block_i6* q8_row = q8 + static_cast<int64_t>(m) * n_q8_blocks;

  // Vectorized loads: one uint4 of ql (32 weights) + one uint2 of qh (the
  // 8 high-bit bytes for the same 32-weight chunk) per iteration.
  const uint4* qlrow16 = reinterpret_cast<const uint4*>(qlrow);
  const uint2* qhrow8 = reinterpret_cast<const uint2*>(qhrow);
  const int32_t K_half_16 = K_half / 16;

  float sum = 0.0f;

  int32_t prev_g = -1;
  float ws = 0.0f;

  for (int32_t i = lane_id; i < K_half_16; i += MV6_WARP_SIZE) {
    uint4 packed16 = __ldg(&qlrow16[i]);
    uint2 qh_chunk = __ldg(&qhrow8[i]);
    int32_t k_base = i * 32;
    uint32_t words[4] = {packed16.x, packed16.y, packed16.z, packed16.w};
    // qh_chunk.x bytes = hi_even_packed[0..3], qh_chunk.y =
    // hi_odd_packed[0..3].
    uint32_t hi_even_word = qh_chunk.x;
    uint32_t hi_odd_word = qh_chunk.y;

    // One uint4 (32 weights) maps to exactly one Q8 activation block (32
    // activations), i.e. q8_block_idx == i. Load the whole block with two
    // vectorized uint4 loads (+ one scale load).
    const Q8Block_i6* qb = &q8_row[i];
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
        prev_g = g;
      }

      int32_t vi_lo = static_cast<int32_t>(packed & 0x0F0F0F0F);
      int32_t vi_hi = static_cast<int32_t>((packed >> 4) & 0x0F0F0F0F);

      uint32_t hi_even_byte = (hi_even_word >> (w * 8)) & 0xFF;
      uint32_t hi_odd_byte = (hi_odd_word >> (w * 8)) & 0xFF;

      // Reconstruct full 6-bit weight bytes (u in [0, 63]).
      int32_t vfull_even =
          vi_lo | static_cast<int32_t>(spread2_i6(hi_even_byte) << 4);
      int32_t vfull_odd =
          vi_hi | static_cast<int32_t>(spread2_i6(hi_odd_byte) << 4);

      int32_t dp = __dp4a(vfull_even, static_cast<int32_t>(a_even[w]), 0);
      dp = __dp4a(vfull_odd, static_cast<int32_t>(a_odd[w]), dp);

      int32_t a_sum = __dp4a(0x01010101, static_cast<int32_t>(a_even[w]), 0);
      a_sum = __dp4a(0x01010101, static_cast<int32_t>(a_odd[w]), a_sum);

      // q = u - 32, so the -32 offset replaces the per-group zero point.
      sum += ws * a_scale *
          (static_cast<float>(dp) - 32.0f * static_cast<float>(a_sum));
    }
  }

  for (int offset = MV6_WARP_SIZE / 2; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);

  if (lane_id == 0)
    out[static_cast<int64_t>(m) * N + n] = __float2bfloat16(sum);
}

// ---------------------------------------------------------------------------
// Persistent Q8 buffer (lazy init, not thread-safe — single-stream only).
// Freed at process exit via a static guard so leak detectors stay quiet; the
// CUDA runtime would otherwise reclaim it on teardown anyway.
// ---------------------------------------------------------------------------

static Q8Block_i6* g_q8_buf_i6 = nullptr;
static size_t g_q8_buf_i6_size = 0;

namespace {
struct Q8BufferGuardI6 {
  ~Q8BufferGuardI6() {
    if (g_q8_buf_i6) {
      // Ignore errors: during process teardown the CUDA context may already be
      // gone (cudaErrorCudartUnloading), which is harmless here.
      cudaFree(g_q8_buf_i6);
      g_q8_buf_i6 = nullptr;
      g_q8_buf_i6_size = 0;
    }
  }
};
Q8BufferGuardI6 g_q8_buf_i6_guard;
} // namespace

static Q8Block_i6* get_q8_buffer_i6(size_t needed) {
  if (g_q8_buf_i6_size < needed) {
    if (g_q8_buf_i6)
      cudaFree(g_q8_buf_i6);
    cudaError_t err = cudaMalloc(&g_q8_buf_i6, needed);
    ET_CHECK_MSG(
        err == cudaSuccess,
        "cudaMalloc failed for Q8 buffer (int6): %s",
        cudaGetErrorString(err));
    g_q8_buf_i6_size = needed;
  }
  return g_q8_buf_i6;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

inline void _int6_plain_mm_cuda(
    const Tensor& A, // [M, K] bf16
    const Tensor& ql, // [N, K/2] uint8
    const Tensor& qh, // [N, K/4] uint8
    const Tensor& scale, // [N, K/gs] int8 codes
    const Tensor& steps, // [N, 1] bf16 per-row scale_step
    int64_t group_size,
    Tensor* output) { // [M, N] bf16, pre-allocated
  int32_t M = A.size(0);
  int32_t K = A.size(1);
  int32_t N = ql.size(0);

  ET_CHECK(A.dtype() == c10::ScalarType::BFloat16);
  ET_CHECK(
      ql.dtype() == c10::ScalarType::Byte ||
      ql.dtype() == c10::ScalarType::Char);
  ET_CHECK(
      qh.dtype() == c10::ScalarType::Byte ||
      qh.dtype() == c10::ScalarType::Char);
  ET_CHECK(
      scale.dtype() == c10::ScalarType::Byte ||
      scale.dtype() == c10::ScalarType::Char);
  ET_CHECK(steps.dtype() == c10::ScalarType::BFloat16);
  ET_CHECK(A.dim() == 2);
  ET_CHECK(ql.dim() == 2);
  ET_CHECK(ql.size(1) == K / 2);
  ET_CHECK(qh.dim() == 2);
  ET_CHECK(qh.size(1) == K / 4);
  ET_CHECK(scale.dim() == 2);
  ET_CHECK(scale.size(0) == N);
  ET_CHECK(steps.dim() == 2);
  ET_CHECK(steps.size(0) == N);
  ET_CHECK(steps.size(1) == 1);

  int32_t gs = static_cast<int32_t>(group_size);
  ET_CHECK_MSG(
      gs > 0 && (gs & (gs - 1)) == 0, "group_size=%d must be a power of 2", gs);
  // group_size must be a multiple of 8 (the dp4a word stride) so a word never
  // straddles a group boundary; gs=16 covers GGUF Q6_K.
  ET_CHECK_MSG(
      gs % 8 == 0,
      "group_size=%d must be a multiple of 8 (e.g. 16 for GGUF Q6_K)",
      gs);
  ET_CHECK_MSG(
      K >= Q8_BLOCK_SIZE_I6 && K % Q8_BLOCK_SIZE_I6 == 0,
      "K=%d must be a positive multiple of %d for dp4a int6 kernel",
      K,
      Q8_BLOCK_SIZE_I6);

  auto stream_result = getCurrentCUDAStream(0);
  ET_CHECK_MSG(stream_result.ok(), "Failed to get CUDA stream");
  cudaStream_t stream = stream_result.get();

  int32_t gs_shift = log2_pow2_i6(gs);

  // Quantize activations to INT8 (even/odd order)
  int32_t n_q8_blocks = K / Q8_BLOCK_SIZE_I6;
  size_t q8_bytes = static_cast<size_t>(M) * n_q8_blocks * sizeof(Q8Block_i6);
  Q8Block_i6* q8_buf = get_q8_buffer_i6(q8_bytes);

  constexpr int32_t Q8_WARPS = 8;
  int32_t blocks_per_m = (n_q8_blocks + Q8_WARPS - 1) / Q8_WARPS;
  dim3 q8_grid(blocks_per_m, M);
  dim3 q8_block(MV6_WARP_SIZE, Q8_WARPS);
  quantize_activations_q8_i6_kernel<<<q8_grid, q8_block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()), q8_buf, K);

  // dp4a matvec
  dim3 grid((N + MV6_NWARPS - 1) / MV6_NWARPS, M);
  dim3 block(MV6_WARP_SIZE, MV6_NWARPS);

  int32_t n_groups = static_cast<int32_t>(scale.size(1));
  int6_w6a8_matvec_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(ql.data_ptr()),
      reinterpret_cast<const uint8_t*>(qh.data_ptr()),
      reinterpret_cast<const int8_t*>(scale.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(steps.data_ptr()),
      q8_buf,
      reinterpret_cast<__nv_bfloat16*>(output->data_ptr()),
      N,
      K,
      gs_shift,
      n_groups);
}

} // namespace executorch::backends::cuda
