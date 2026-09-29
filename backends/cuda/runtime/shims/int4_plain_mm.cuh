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
//           zero_point_step[N, K/256]. The group zero is zero_code *
//           zero_point_step[b].
// The finer per-256 scale AND zero steps (vs per-row) lift whole-weight dequant
// SNR to ~45.89 dB (vs 45.15 for a per-row zero step) at ~4.625 bpw.
//
// T3 super-block-cooperative step reuse: the per-256 fp16 scale_step and
// zero_point_step live in separate [N, K/256] tensors, so a naive per-group
// load costs
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

__device__ __forceinline__ uint32_t int4_uint4_at(uint4 value, int32_t index) {
  switch (index) {
    case 0:
      return value.x;
    case 1:
      return value.y;
    case 2:
      return value.z;
    default:
      return value.w;
  }
}

template <int32_t ROWS>
__device__ __forceinline__ void int4_w4a8_matvec_coalesced_body(
    const uint8_t* __restrict__ qdata,
    const uint8_t* __restrict__ w_scale_t,
    const __half* __restrict__ w_scale_step,
    const uint8_t* __restrict__ w_zero_t,
    const __half* __restrict__ w_zero_point_step,
    const Q8Block* __restrict__ q8,
    __nv_bfloat16* __restrict__ out,
    int32_t n,
    int32_t N,
    int32_t K,
    int32_t gs_shift,
    int32_t n_groups,
    int32_t n_super) {
  const int32_t K_half = K / 2;
  const int32_t lane_id = threadIdx.x;
  const int32_t n_q8_blocks = K / Q8_BLOCK_SIZE;
  const uint8_t* qrow = qdata + static_cast<int64_t>(n) * K_half;
  const uint8_t* scale_row = w_scale_t + static_cast<int64_t>(n) * n_groups;
  const __half* scale_step_row =
      w_scale_step + static_cast<int64_t>(n) * n_super;
  const uint8_t* zero_row = w_zero_t + static_cast<int64_t>(n) * n_groups;
  const __half* zero_point_step_row =
      w_zero_point_step + static_cast<int64_t>(n) * n_super;

  const uint4* qrow16 = reinterpret_cast<const uint4*>(qrow);
  const int32_t K_half_16 = K_half / 16;
  const int32_t sb_shift = SUPER_BLOCK_SHIFT - gs_shift;
  const int32_t leader = lane_id & ~7;
  const int32_t n_iters =
      ((K_half_16 + MV_WARP_SIZE - 1) / MV_WARP_SIZE) * MV_WARP_SIZE;
  float sums[ROWS] = {};

  for (int32_t it = 0; it < n_iters; it += MV_WARP_SIZE) {
    const int32_t i = it + lane_id;
    const bool active = i < K_half_16;
    const int32_t i_safe = active ? i : 0;
    const uint4 packed16 = __ldg(&qrow16[i_safe]);
    const int32_t k_base = i_safe * 32;
    const uint32_t words[4] = {
        packed16.x, packed16.y, packed16.z, packed16.w};
    const int32_t g = k_base >> gs_shift;

    uint32_t steps_packed = 0;
    if (lane_id == leader) {
      const int32_t sb = g >> sb_shift;
      const unsigned short s_bits =
          __half_as_ushort(__ldg(&scale_step_row[sb]));
      const unsigned short z_bits =
          __half_as_ushort(__ldg(&zero_point_step_row[sb]));
      steps_packed = static_cast<uint32_t>(s_bits) |
          (static_cast<uint32_t>(z_bits) << 16);
    }
    steps_packed = __shfl_sync(0xffffffff, steps_packed, leader);
    if (!active) {
      continue;
    }

    const float scale_step = __half2float(
        __ushort_as_half(static_cast<unsigned short>(steps_packed & 0xFFFF)));
    const float zero_point_step = __half2float(
        __ushort_as_half(static_cast<unsigned short>(steps_packed >> 16)));
    const float ws = static_cast<float>(__ldg(&scale_row[g])) * scale_step;
    const float wz = static_cast<float>(__ldg(&zero_row[g])) * zero_point_step;

    uint4 activations_even[ROWS];
    uint4 activations_odd[ROWS];
    float activation_scales[ROWS];
#pragma unroll
    for (int32_t row = 0; row < ROWS; ++row) {
      const Q8Block* qb =
          q8 + static_cast<int64_t>(row) * n_q8_blocks + i_safe;
      activations_even[row] =
          *reinterpret_cast<const uint4*>(qb->qs_even);
      activations_odd[row] =
          *reinterpret_cast<const uint4*>(qb->qs_odd);
      activation_scales[row] = qb->d;
    }

#pragma unroll
    for (int32_t w = 0; w < 4; ++w) {
      const uint32_t packed = words[w];
      const int32_t vi_lo = packed & 0x0F0F0F0F;
      const int32_t vi_hi = (packed >> 4) & 0x0F0F0F0F;

#pragma unroll
      for (int32_t row = 0; row < ROWS; ++row) {
        const uint32_t a_even = int4_uint4_at(activations_even[row], w);
        const uint32_t a_odd = int4_uint4_at(activations_odd[row], w);
        int32_t dp = __dp4a(vi_lo, static_cast<int32_t>(a_even), 0);
        dp = __dp4a(vi_hi, static_cast<int32_t>(a_odd), dp);
        int32_t a_sum8 =
            __dp4a(0x01010101, static_cast<int32_t>(a_even), 0);
        a_sum8 =
            __dp4a(0x01010101, static_cast<int32_t>(a_odd), a_sum8);
        sums[row] += ws * activation_scales[row] *
            (static_cast<float>(dp) - wz * static_cast<float>(a_sum8));
      }
    }
  }

  for (int32_t offset = MV_WARP_SIZE / 2; offset > 0; offset >>= 1) {
#pragma unroll
    for (int32_t row = 0; row < ROWS; ++row) {
      sums[row] += __shfl_xor_sync(0xffffffff, sums[row], offset);
    }
  }

  if (lane_id == 0) {
#pragma unroll
    for (int32_t row = 0; row < ROWS; ++row) {
      out[static_cast<int64_t>(row) * N + n] =
          __float2bfloat16(sums[row]);
    }
  }
}

__global__ void __launch_bounds__(MV_THREADS)
    int4_w4a8_matvec_coalesced_kernel(
        const uint8_t* __restrict__ qdata,
        const uint8_t* __restrict__ w_scale_t,
        const __half* __restrict__ w_scale_step,
        const uint8_t* __restrict__ w_zero_t,
        const __half* __restrict__ w_zero_point_step,
        const Q8Block* __restrict__ q8,
        __nv_bfloat16* __restrict__ out,
        int32_t N,
        int32_t K,
        int32_t gs_shift,
        int32_t n_groups,
        int32_t n_super) {
  const int32_t n = blockIdx.x * MV_NWARPS + threadIdx.y;
  const int32_t m = blockIdx.y;
  if (n >= N) {
    return;
  }
  int4_w4a8_matvec_coalesced_body<1>(
      qdata,
      w_scale_t,
      w_scale_step,
      w_zero_t,
      w_zero_point_step,
      q8 + static_cast<int64_t>(m) * (K / Q8_BLOCK_SIZE),
      out + static_cast<int64_t>(m) * N,
      n,
      N,
      K,
      gs_shift,
      n_groups,
      n_super);
}

#define DEFINE_INT4_MULTIROW_KERNEL(ROWS)                                      \
  __global__ void __launch_bounds__(MV_THREADS)                               \
      int4_w4a8_matvec_m##ROWS##_coalesced_kernel(                            \
          const uint8_t* __restrict__ qdata,                                  \
          const uint8_t* __restrict__ w_scale_t,                              \
          const __half* __restrict__ w_scale_step,                            \
          const uint8_t* __restrict__ w_zero_t,                               \
          const __half* __restrict__ w_zero_point_step,                       \
          const Q8Block* __restrict__ q8,                                     \
          __nv_bfloat16* __restrict__ out,                                    \
          int32_t N,                                                          \
          int32_t K,                                                          \
          int32_t gs_shift,                                                   \
          int32_t n_groups,                                                   \
          int32_t n_super) {                                                  \
    const int32_t n = blockIdx.x * MV_NWARPS + threadIdx.y;                   \
    if (n >= N) {                                                             \
      return;                                                                 \
    }                                                                         \
    int4_w4a8_matvec_coalesced_body<ROWS>(                                    \
        qdata,                                                                \
        w_scale_t,                                                            \
        w_scale_step,                                                         \
        w_zero_t,                                                             \
        w_zero_point_step,                                                    \
        q8,                                                                   \
        out,                                                                  \
        n,                                                                    \
        N,                                                                    \
        K,                                                                    \
        gs_shift,                                                             \
        n_groups,                                                             \
        n_super);                                                             \
  }

DEFINE_INT4_MULTIROW_KERNEL(2)
DEFINE_INT4_MULTIROW_KERNEL(3)
DEFINE_INT4_MULTIROW_KERNEL(4)

#undef DEFINE_INT4_MULTIROW_KERNEL

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

inline void _int4_plain_mm_cuda(
    const Tensor& A, // [M, K] bf16
    const Tensor& qdata, // [N, K//2] uint8
    const Tensor& scale, // [N, K//gs] uint8 codes
    const Tensor& scale_step, // [N, K//256] fp16
    const Tensor& zero, // [N, K//gs] uint8 codes
    const Tensor& zero_point_step, // [N, K//256] fp16
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
  ET_CHECK(zero_point_step.dtype() == c10::ScalarType::Half);
  ET_CHECK(A.dim() == 2);
  ET_CHECK(qdata.dim() == 2);
  ET_CHECK(qdata.size(1) == K / 2);
  ET_CHECK(scale.dim() == 2);
  ET_CHECK(scale.size(0) == N);
  ET_CHECK(scale_step.dim() == 2);
  ET_CHECK(scale_step.size(0) == N);
  ET_CHECK(zero.dim() == 2);
  ET_CHECK(zero.size(0) == N);
  ET_CHECK(zero_point_step.dim() == 2);
  ET_CHECK(zero_point_step.size(0) == N);
  ET_CHECK(zero_point_step.size(1) == scale_step.size(1));

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
  if (M == 4) {
    dim3 m4_grid((N + MV_NWARPS - 1) / MV_NWARPS);
    int4_w4a8_matvec_m4_coalesced_kernel<<<m4_grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(qdata.data_ptr()),
        reinterpret_cast<const uint8_t*>(scale.data_ptr()),
        reinterpret_cast<const __half*>(scale_step.data_ptr()),
        reinterpret_cast<const uint8_t*>(zero.data_ptr()),
        reinterpret_cast<const __half*>(zero_point_step.data_ptr()),
        q8_buf,
        reinterpret_cast<__nv_bfloat16*>(output->data_ptr()),
        N,
        K,
        gs_shift,
        n_groups,
        n_super);
    return;
  }

  if (M == 3) {
    dim3 m3_grid((N + MV_NWARPS - 1) / MV_NWARPS);
    int4_w4a8_matvec_m3_coalesced_kernel<<<m3_grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(qdata.data_ptr()),
        reinterpret_cast<const uint8_t*>(scale.data_ptr()),
        reinterpret_cast<const __half*>(scale_step.data_ptr()),
        reinterpret_cast<const uint8_t*>(zero.data_ptr()),
        reinterpret_cast<const __half*>(zero_point_step.data_ptr()),
        q8_buf,
        reinterpret_cast<__nv_bfloat16*>(output->data_ptr()),
        N,
        K,
        gs_shift,
        n_groups,
        n_super);
    return;
  }

  if (M == 2) {
    dim3 m2_grid((N + MV_NWARPS - 1) / MV_NWARPS);
    int4_w4a8_matvec_m2_coalesced_kernel<<<m2_grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(qdata.data_ptr()),
        reinterpret_cast<const uint8_t*>(scale.data_ptr()),
        reinterpret_cast<const __half*>(scale_step.data_ptr()),
        reinterpret_cast<const uint8_t*>(zero.data_ptr()),
        reinterpret_cast<const __half*>(zero_point_step.data_ptr()),
        q8_buf,
        reinterpret_cast<__nv_bfloat16*>(output->data_ptr()),
        N,
        K,
        gs_shift,
        n_groups,
        n_super);
    return;
  }

  int4_w4a8_matvec_coalesced_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(qdata.data_ptr()),
      reinterpret_cast<const uint8_t*>(scale.data_ptr()),
      reinterpret_cast<const __half*>(scale_step.data_ptr()),
      reinterpret_cast<const uint8_t*>(zero.data_ptr()),
      reinterpret_cast<const __half*>(zero_point_step.data_ptr()),
      q8_buf,
      reinterpret_cast<__nv_bfloat16*>(output->data_ptr()),
      N,
      K,
      gs_shift,
      n_groups,
      n_super);
}

} // namespace executorch::backends::cuda
