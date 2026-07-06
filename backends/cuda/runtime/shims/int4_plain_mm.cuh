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
// Metadata encoding (transposed AOT for coalesced loads):
//   scale : [N, K//gs] uint8 code + a per-256-super-block fp16 step
//           scale_step[N, K/256]. The group scale is scale_code * scale_step[b],
//           b = super-block index = k >> 8.
//   zero  : [N, K//gs] uint8 code + a per-256-super-block fp16 step
//           zero_step[N, K/256]. The group zero is zero_code * zero_step[b].
// The finer per-256 scale AND zero steps (vs per-row) lift whole-weight dequant
// SNR to ~45.89 dB (vs 45.15 for a per-row zero step) at ~4.625 bpw.
//
// T3 super-block-cooperative step reuse: the per-256 fp16 scale_step and
// zero_step live in separate [N, K/256] tensors, so a naive per-group load costs
// a distant global access every group. Instead, the 32 warp lanes form 8-lane
// subgroups that each cover ONE super-block per iteration; only the subgroup
// leader loads + PACKS both fp16 steps into one 32-bit word and __shfl-
// broadcasts that ONE word to its 7 followers (z_pack: 8x fewer step loads, no
// extra shuffle vs the scale-only baseline, register-only, no smem => no
// occupancy cliff). Mirrors llama.cpp's per-super-block metadata amortization.
//
// Dynamically quantizes bf16 activations to INT8 (per-32-element blocks),
// then uses dp4a for fused int4×int8 dot products with 16-byte vectorized
// loads and warp-cooperative quantization.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
// GGUF Q4_K super-block = 256 weights; the fp16 scale step is per-super-block.
constexpr int32_t SUPER_BLOCK = 256;
constexpr int32_t SUPER_BLOCK_SHIFT = 8; // log2(SUPER_BLOCK)

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
// Coalesced-metadata W4A8 dp4a matvec (T3 super-block-cooperative step reuse)
//
// Reads scale/zero in the transposed [N, n_groups] layout (transposed AOT at
// export time) as uint8 codes. Both the scale and the zero are decoded with a
// per-256-super-block fp16 step; the leader packs BOTH fp16 steps into the ONE
// 32-bit warp-shuffle word and broadcasts it across each 8-lane subgroup
// (z_pack, see file header). With group_size >= 32, one uint4 (32 weights) maps
// to exactly one activation block and one weight group, so within a warp the 32
// lanes touch 32 consecutive groups (4 super-blocks). In [N, n_groups] layout
// those 32 group codes are contiguous => a single coalesced load.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(MV_THREADS)
    int4_w4a8_matvec_coalesced_kernel(
        const uint8_t* __restrict__ qdata,
        const uint8_t* __restrict__ w_scale_t, // [N, n_groups] uint8 codes
        const __half* __restrict__ w_scale_step, // [N, n_super] fp16
        const uint8_t* __restrict__ w_zero_t, // [N, n_groups] uint8 codes
        const __half* __restrict__ w_zero_step, // [N, n_super] fp16
        const Q8Block* __restrict__ q8,
        __nv_bfloat16* __restrict__ out,
        int32_t N,
        int32_t K,
        int32_t gs_shift,
        int32_t n_groups,
        int32_t n_super) {
  const int32_t n = blockIdx.x * MV_NWARPS + threadIdx.y;
  const int32_t m = blockIdx.y;
  if (n >= N)
    return;

  const int32_t K_half = K / 2;
  const int32_t lane_id = threadIdx.x;
  const int32_t n_q8_blocks = K / Q8_BLOCK_SIZE;

  const uint8_t* qrow = qdata + static_cast<int64_t>(n) * K_half;
  const uint8_t* scale_row = w_scale_t + static_cast<int64_t>(n) * n_groups;
  const __half* scale_step_row =
      w_scale_step + static_cast<int64_t>(n) * n_super;
  const uint8_t* zero_row = w_zero_t + static_cast<int64_t>(n) * n_groups;
  // Per-256 fp16 zero step (z_pack): decoded via the SAME 8-lane leader
  // broadcast as the scale step (both packed into one 32-bit shuffle word
  // below), so the dp4a dot products stay bit-identical to the scale-only
  // kernel. zero = zero_code * zero_step[super-block].
  const __half* zero_step_row = w_zero_step + static_cast<int64_t>(n) * n_super;
  const Q8Block* q8_row = q8 + static_cast<int64_t>(m) * n_q8_blocks;

  const uint4* qrow16 = reinterpret_cast<const uint4*>(qrow);
  const int32_t K_half_16 = K_half / 16;

  float sum = 0.0f;

  // T3: within a warp iteration the 32 lanes cover groups i0..i0+31 = 4
  // consecutive super-blocks, split into 8-lane subgroups (lanes 8s..8s+7 share
  // super-block b = g >> sb_shift). Only each subgroup leader (lane_id % 8 == 0)
  // loads + converts the two fp16 steps, PACKS them into one 32-bit word, and
  // __shfl-broadcasts that single word to the 7 followers. 8x fewer step loads,
  // ONE shuffle (same count as the scale-only baseline), register-only (no smem
  // => no occupancy cliff).
  const int32_t sb_shift = SUPER_BLOCK_SHIFT - gs_shift; // group g -> super-block
  const int32_t leader = lane_id & ~7; // base lane of this 8-lane subgroup

  // Warp-aligned trip count so ALL 32 lanes execute the same number of
  // iterations and therefore all reach the __shfl_sync every iteration (a
  // full-mask shuffle deadlocks if some lanes exit the loop early — which
  // happens when K_half_16 < 32, e.g. tiny test shapes). Out-of-range lanes do a
  // safe dummy load (index 0) and contribute 0 to the accumulation.
  const int32_t n_iters =
      ((K_half_16 + MV_WARP_SIZE - 1) / MV_WARP_SIZE) * MV_WARP_SIZE;

  for (int32_t it = 0; it < n_iters; it += MV_WARP_SIZE) {
    int32_t i = it + lane_id;
    bool active = i < K_half_16;
    int32_t i_safe = active ? i : 0;

    uint4 packed16 = __ldg(&qrow16[i_safe]);
    int32_t k_base = i_safe * 32;
    uint32_t words[4] = {packed16.x, packed16.y, packed16.z, packed16.w};

    // Group index for this uint4 (constant across its 4 dp4a words at gs=32).
    int32_t g = k_base >> gs_shift;
    // Subgroup leader packs BOTH per-256 fp16 steps (scale low16, zero high16)
    // into one 32-bit word and broadcasts it once; followers unpack. All lanes
    // reach this shuffle (warp-aligned loop), so the full mask is safe.
    uint32_t steps_packed = 0;
    if (lane_id == leader) {
      int32_t sb = g >> sb_shift;
      unsigned short s_bits = __half_as_ushort(__ldg(&scale_step_row[sb]));
      unsigned short z_bits = __half_as_ushort(__ldg(&zero_step_row[sb]));
      steps_packed = static_cast<uint32_t>(s_bits) |
          (static_cast<uint32_t>(z_bits) << 16);
    }
    steps_packed = __shfl_sync(0xffffffff, steps_packed, leader);
    if (!active)
      continue;
    float scale_step = __half2float(
        __ushort_as_half(static_cast<unsigned short>(steps_packed & 0xFFFF)));
    float zero_step = __half2float(
        __ushort_as_half(static_cast<unsigned short>(steps_packed >> 16)));
    // Effective per-group scale/zero (one coalesced code byte each per group).
    float ws = static_cast<float>(__ldg(&scale_row[g])) * scale_step;
    float wz = static_cast<float>(__ldg(&zero_row[g])) * zero_step;

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
    const Tensor& scale_step, // [N, K//256] fp16
    const Tensor& zero, // [N, K//gs] uint8 codes
    const Tensor& zero_step, // [N, K//256] fp16
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
  ET_CHECK(scale_step.dtype() == c10::ScalarType::Half);
  ET_CHECK(
      zero.dtype() == c10::ScalarType::Byte ||
      zero.dtype() == c10::ScalarType::Char);
  ET_CHECK(zero_step.dtype() == c10::ScalarType::Half);
  ET_CHECK(A.dim() == 2);
  ET_CHECK(qdata.dim() == 2);
  ET_CHECK(qdata.size(1) == K / 2);
  ET_CHECK(scale.dim() == 2);
  ET_CHECK(scale.size(0) == N);
  ET_CHECK(scale_step.dim() == 2);
  ET_CHECK(scale_step.size(0) == N);
  ET_CHECK(zero.dim() == 2);
  ET_CHECK(zero.size(0) == N);
  ET_CHECK(zero_step.dim() == 2);
  ET_CHECK(zero_step.size(0) == N);
  ET_CHECK(zero_step.size(1) == scale_step.size(1));

  int32_t gs = static_cast<int32_t>(group_size);
  ET_CHECK_MSG(
      gs > 0 && (gs & (gs - 1)) == 0, "group_size=%d must be a power of 2", gs);
  ET_CHECK_MSG(
      K >= Q8_BLOCK_SIZE && K % Q8_BLOCK_SIZE == 0,
      "K=%d must be a positive multiple of %d for dp4a kernel",
      K,
      Q8_BLOCK_SIZE);
  ET_CHECK_MSG(
      K % SUPER_BLOCK == 0,
      "K=%d must be a multiple of %d (super-block) for the per-256 scale step",
      K,
      SUPER_BLOCK);

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
  int32_t n_super = static_cast<int32_t>(scale_step.size(1));
  int4_w4a8_matvec_coalesced_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(qdata.data_ptr()),
      reinterpret_cast<const uint8_t*>(scale.data_ptr()),
      reinterpret_cast<const __half*>(scale_step.data_ptr()),
      reinterpret_cast<const uint8_t*>(zero.data_ptr()),
      reinterpret_cast<const __half*>(zero_step.data_ptr()),
      q8_buf,
      reinterpret_cast<__nv_bfloat16*>(output->data_ptr()),
      N, K, gs_shift, n_groups, n_super);
}

} // namespace executorch::backends::cuda
