/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// W5A8 dp4a matvec for packed INT5 decode (M <= 4), used for GGUF Q5_K weights.
//
// Reads a genuine 5-bit packed weight (CudaDp4aPlanarInt5Tensor format), split
// into two planes:
//   ql    : [N, K/2] uint8 — low-nibble plane, nibble-packed even/odd exactly
//           like the INT4 path (ql[:,j] = lo[:,2j] | (lo[:,2j+1] << 4)).
//   qh    : [N, K/8] uint8 — high-1-bit plane, 8 values/byte, arranged per
//           32-weight chunk as 4 bytes (one per dp4a word); each byte holds the
//           four 1-bit highs of that word's even weights in the low nibble and
//           its odd weights in the high nibble
//           (hi_even_nibble | (hi_odd_nibble << 4), hi = (u >> 4) & 1).
//   scale : [N, K/gs] uint8 codes — per-group scales, row-major (coalesced).
//   zero  : [N, K/gs] uint8 codes — per-group zero points, row-major.
//   steps : [N, 2] bf16 — per-row super-scale (scale_step, zero_step); the group
//           scale/zero are decoded as code*scale_step / code*zero_step.
// The stored 5-bit value is the raw unsigned u = q in [0, 31]; Q5_K is
// asymmetric so a per-group zero point is subtracted in the kernel (like INT4).
//
// Dynamically quantizes bf16 activations to INT8 (per-32-element blocks,
// even/odd order, identical to the INT4/INT6 path), reconstructs full 5-bit
// weight bytes per dp4a word (vfull = vi_lo | (spread1(hi_nibble) << 4)), and
// uses dp4a for fused int5×int8 dot products with vectorized weight loads and
// warp-cooperative quantization.
//
// Symbol names are suffixed _i5 / distinct from int4/int6/int8_plain_mm.cuh so
// all translation units can be linked together without ODR conflicts.

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

constexpr int32_t MV5_NWARPS = 8;
constexpr int32_t MV5_WARP_SIZE = 32;
constexpr int32_t MV5_THREADS = MV5_NWARPS * MV5_WARP_SIZE;
constexpr int32_t Q8_BLOCK_SIZE_I5 = 32;

__host__ __forceinline__ int32_t log2_pow2_i5(int32_t v) {
  int32_t r = 0;
  while (v > 1) {
    v >>= 1;
    r++;
  }
  return r;
}

// Expand a nibble's four 1-bit fields into four byte lanes (each in bit 0):
//   in  : nibble n = [.. .. .. .. | b3 b2 b1 b0]
//   out : lane0=[b0], lane1=[b1], lane2=[b2], lane3=[b3]
// ~4 ALU ops; verified by truth-table. Used to place the high bit of each
// weight into bit 4 of the corresponding dp4a byte lane (after << 4).
__device__ __forceinline__ uint32_t spread1_i5(uint32_t n) {
  return (n & 0x1u) | ((n & 0x2u) << 7) | ((n & 0x4u) << 14) |
      ((n & 0x8u) << 21);
}

// ---------------------------------------------------------------------------
// Activation quantization: bf16 -> int8 (warp-cooperative, per-32-element
// blocks, EVEN/ODD order — identical to the INT4/INT6 path's Q8Block).
// ---------------------------------------------------------------------------

// alignas(16) pads sizeof(Q8Block_i5) to 48 so each block (and its
// qs_even/qs_odd 16-byte halves) is 16-byte aligned, allowing two vectorized
// uint4 loads of a block's int8 activations instead of eight scalar int32 loads.
struct alignas(16) Q8Block_i5 {
  int8_t qs_even[Q8_BLOCK_SIZE_I5 / 2];
  int8_t qs_odd[Q8_BLOCK_SIZE_I5 / 2];
  float d; // scale
};

__global__ void quantize_activations_q8_i5_kernel(
    const __nv_bfloat16* __restrict__ A,
    Q8Block_i5* __restrict__ q8,
    int32_t K) {
  const int32_t m = blockIdx.y;
  const int32_t block_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t n_blocks = K / Q8_BLOCK_SIZE_I5;
  if (block_id >= n_blocks)
    return;

  const int32_t lane = threadIdx.x;
  const __nv_bfloat16* src =
      A + static_cast<int64_t>(m) * K + block_id * Q8_BLOCK_SIZE_I5;
  Q8Block_i5* dst = q8 + static_cast<int64_t>(m) * n_blocks + block_id;

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
// W5A8 dp4a matvec kernel
//
// dp4a is linear, so reconstructing v = lo + (hi<<4) and dotting once is
// equivalent to two separate dp4a passes. We reconstruct the full 5-bit byte
// (vfull = vi_lo | (spread1(hi_nibble) << 4)) so a single dp4a per even/odd half
// covers the whole weight. Q5_K is asymmetric, so the per-group zero point is
// subtracted as out += scale * a_scale * (dp - zero * a_sum) (like INT4).
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(MV5_THREADS) int5_w5a8_matvec_kernel(
    const uint8_t* __restrict__ ql, // [N, K/2]
    const uint8_t* __restrict__ qh, // [N, K/8]
    const uint8_t* __restrict__ w_scale, // [N, n_groups] uint8 codes
    const uint8_t* __restrict__ w_zero, // [N, n_groups] uint8 codes
    const __nv_bfloat16* __restrict__ w_steps, // [N, 2] (scale_step, zero_step)
    const Q8Block_i5* __restrict__ q8,
    __nv_bfloat16* __restrict__ out,
    int32_t N,
    int32_t K,
    int32_t gs_shift,
    int32_t n_groups) {
  const int32_t n = blockIdx.x * MV5_NWARPS + threadIdx.y;
  const int32_t m = blockIdx.y;
  if (n >= N)
    return;

  const int32_t K_half = K / 2;
  const int32_t K_eighth = K / 8;
  const int32_t lane_id = threadIdx.x;
  const int32_t n_q8_blocks = K / Q8_BLOCK_SIZE_I5;

  const uint8_t* qlrow = ql + static_cast<int64_t>(n) * K_half;
  const uint8_t* qhrow = qh + static_cast<int64_t>(n) * K_eighth;
  const uint8_t* scale_row = w_scale + static_cast<int64_t>(n) * n_groups;
  const uint8_t* zero_row = w_zero + static_cast<int64_t>(n) * n_groups;
  // Per-row super-scales: the uint8 group codes decode as scale = code *
  // scale_step, zero = code * zero_step. scale_step is a per-row constant (it
  // factors out of the dp4a sum), so the dp4a dot products below are
  // bit-identical to the bf16-metadata kernel.
  const float scale_step =
      __bfloat162float(__ldg(&w_steps[static_cast<int64_t>(n) * 2 + 0]));
  const float zero_step =
      __bfloat162float(__ldg(&w_steps[static_cast<int64_t>(n) * 2 + 1]));
  const Q8Block_i5* q8_row = q8 + static_cast<int64_t>(m) * n_q8_blocks;

  // Vectorized loads: one uint4 of ql (32 weights) + one uint (the 4 high-bit
  // bytes for the same 32-weight chunk) per iteration.
  const uint4* qlrow16 = reinterpret_cast<const uint4*>(qlrow);
  const uint32_t* qhrow4 = reinterpret_cast<const uint32_t*>(qhrow);
  const int32_t K_half_16 = K_half / 16;

  float sum = 0.0f;

  int32_t prev_g = -1;
  float ws = 0.0f, wz = 0.0f;

  for (int32_t i = lane_id; i < K_half_16; i += MV5_WARP_SIZE) {
    uint4 packed16 = __ldg(&qlrow16[i]);
    // qh_word bytes = [word0, word1, word2, word3] high nibbles for this chunk.
    uint32_t qh_word = __ldg(&qhrow4[i]);
    int32_t k_base = i * 32;
    uint32_t words[4] = {packed16.x, packed16.y, packed16.z, packed16.w};

    // One uint4 (32 weights) maps to exactly one Q8 activation block (32
    // activations), i.e. q8_block_idx == i. Load the whole block with two
    // vectorized uint4 loads (+ one scale/zero load).
    const Q8Block_i5* qb = &q8_row[i];
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

      int32_t vi_lo = static_cast<int32_t>(packed & 0x0F0F0F0F);
      int32_t vi_hi = static_cast<int32_t>((packed >> 4) & 0x0F0F0F0F);

      // Byte w of qh_word holds this dp4a word's high bits:
      //   low nibble  = even weights' high bit (positions 0,2,4,6)
      //   high nibble = odd weights' high bit  (positions 1,3,5,7)
      uint32_t hi_byte = (qh_word >> (w * 8)) & 0xFF;
      uint32_t hi_even_nib = hi_byte & 0xF;
      uint32_t hi_odd_nib = (hi_byte >> 4) & 0xF;

      // Reconstruct full 5-bit weight bytes (u in [0, 31]).
      int32_t vfull_even =
          vi_lo | static_cast<int32_t>(spread1_i5(hi_even_nib) << 4);
      int32_t vfull_odd =
          vi_hi | static_cast<int32_t>(spread1_i5(hi_odd_nib) << 4);

      int32_t dp = __dp4a(vfull_even, static_cast<int32_t>(a_even[w]), 0);
      dp = __dp4a(vfull_odd, static_cast<int32_t>(a_odd[w]), dp);

      int32_t a_sum = __dp4a(0x01010101, static_cast<int32_t>(a_even[w]), 0);
      a_sum = __dp4a(0x01010101, static_cast<int32_t>(a_odd[w]), a_sum);

      // Asymmetric: w = scale * (u - zero), so the per-group zero point is
      // subtracted from the dp4a sum (weighted by the activation sum).
      sum += ws * a_scale *
          (static_cast<float>(dp) - wz * static_cast<float>(a_sum));
    }
  }

  for (int offset = MV5_WARP_SIZE / 2; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);

  if (lane_id == 0)
    out[static_cast<int64_t>(m) * N + n] = __float2bfloat16(sum);
}

// ---------------------------------------------------------------------------
// Persistent Q8 buffer (lazy init, not thread-safe — single-stream only).
// Freed at process exit via a static guard so leak detectors stay quiet; the
// CUDA runtime would otherwise reclaim it on teardown anyway.
// ---------------------------------------------------------------------------

static Q8Block_i5* g_q8_buf_i5 = nullptr;
static size_t g_q8_buf_i5_size = 0;

namespace {
struct Q8BufferGuardI5 {
  ~Q8BufferGuardI5() {
    if (g_q8_buf_i5) {
      // Ignore errors: during process teardown the CUDA context may already be
      // gone (cudaErrorCudartUnloading), which is harmless here.
      cudaFree(g_q8_buf_i5);
      g_q8_buf_i5 = nullptr;
      g_q8_buf_i5_size = 0;
    }
  }
};
Q8BufferGuardI5 g_q8_buf_i5_guard;
} // namespace

static Q8Block_i5* get_q8_buffer_i5(size_t needed) {
  if (g_q8_buf_i5_size < needed) {
    if (g_q8_buf_i5)
      cudaFree(g_q8_buf_i5);
    cudaError_t err = cudaMalloc(&g_q8_buf_i5, needed);
    ET_CHECK_MSG(
        err == cudaSuccess,
        "cudaMalloc failed for Q8 buffer (int5): %s",
        cudaGetErrorString(err));
    g_q8_buf_i5_size = needed;
  }
  return g_q8_buf_i5;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

inline void _int5_plain_mm_cuda(
    const Tensor& A, // [M, K] bf16
    const Tensor& ql, // [N, K/2] uint8
    const Tensor& qh, // [N, K/8] uint8
    const Tensor& scale, // [N, K/gs] uint8 codes
    const Tensor& zero, // [N, K/gs] uint8 codes
    const Tensor& steps, // [N, 2] bf16 (scale_step, zero_step)
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
  ET_CHECK(
      zero.dtype() == c10::ScalarType::Byte ||
      zero.dtype() == c10::ScalarType::Char);
  ET_CHECK(steps.dtype() == c10::ScalarType::BFloat16);
  ET_CHECK(A.dim() == 2);
  ET_CHECK(ql.dim() == 2);
  ET_CHECK(ql.size(1) == K / 2);
  ET_CHECK(qh.dim() == 2);
  ET_CHECK(qh.size(1) == K / 8);
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
  // group_size must be a multiple of 32 (one uint4 ql chunk == one group) so a
  // chunk never straddles a group boundary; gs=32 covers GGUF Q5_K.
  ET_CHECK_MSG(
      gs % 32 == 0,
      "group_size=%d must be a multiple of 32 (e.g. 32 for GGUF Q5_K)",
      gs);
  ET_CHECK_MSG(
      K >= Q8_BLOCK_SIZE_I5 && K % Q8_BLOCK_SIZE_I5 == 0,
      "K=%d must be a positive multiple of %d for dp4a int5 kernel",
      K,
      Q8_BLOCK_SIZE_I5);

  auto stream_result = getCurrentCUDAStream(0);
  ET_CHECK_MSG(stream_result.ok(), "Failed to get CUDA stream");
  cudaStream_t stream = stream_result.get();

  int32_t gs_shift = log2_pow2_i5(gs);

  // Quantize activations to INT8 (even/odd order)
  int32_t n_q8_blocks = K / Q8_BLOCK_SIZE_I5;
  size_t q8_bytes = static_cast<size_t>(M) * n_q8_blocks * sizeof(Q8Block_i5);
  Q8Block_i5* q8_buf = get_q8_buffer_i5(q8_bytes);

  constexpr int32_t Q8_WARPS = 8;
  int32_t blocks_per_m = (n_q8_blocks + Q8_WARPS - 1) / Q8_WARPS;
  dim3 q8_grid(blocks_per_m, M);
  dim3 q8_block(MV5_WARP_SIZE, Q8_WARPS);
  quantize_activations_q8_i5_kernel<<<q8_grid, q8_block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()), q8_buf, K);

  // dp4a matvec
  dim3 grid((N + MV5_NWARPS - 1) / MV5_NWARPS, M);
  dim3 block(MV5_WARP_SIZE, MV5_NWARPS);

  int32_t n_groups = static_cast<int32_t>(scale.size(1));
  int5_w5a8_matvec_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(ql.data_ptr()),
      reinterpret_cast<const uint8_t*>(qh.data_ptr()),
      reinterpret_cast<const uint8_t*>(scale.data_ptr()),
      reinterpret_cast<const uint8_t*>(zero.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(steps.data_ptr()),
      q8_buf,
      reinterpret_cast<__nv_bfloat16*>(output->data_ptr()),
      N,
      K,
      gs_shift,
      n_groups);
}

} // namespace executorch::backends::cuda
