#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
``mlx::int8_int4_gemm``: 8-bit dynamic activation x 4-bit per-channel weight
linear, lowered to Metal Performance Primitives (Metal 4) kernels that do
INT8 x INT4 -> INT32 accumulation over the full K dimension.

Quantization scheme (symmetric, no zero points):

    x_scale[m] = max(max_k |x[m, k]|, eps) / 127                   per token
    xq[m, k]   = clamp(round(x[m, k] * (1 / x_scale[m])), -128, 127)
    W[n, k]    = w_scale[n] * wq[n, k]                           per channel
    Y[m, n]    = x_scale[m] * w_scale[n] * sum_k xq[m, k] * wq[n, k]

``round`` is round-half-to-even and the scale math is fp32, so the kernels and
the eager reference quantize identically.

Weight packing (see :func:`pack_int4`): ``w_packed`` is ``uint8 [N, K // 2]``;
byte ``j`` of row ``n`` holds ``wq[n, 2j]`` in its low nibble and
``wq[n, 2j + 1]`` in its high nibble, both as two's-complement int4 (the
layout of Metal's ``int4b_format``).

INT32 accumulation cannot overflow while ``K * 128 * 8 < 2**31``
(``K < 2**21``).

Lowering variants (``full_k``):
    * ``False``: one kernel. Each threadgroup quantizes its activation rows and
      unpacks int4 weights into threadgroup memory one K tile at a time, and
      MPP multiplies the int8 tiles.
    * ``True``: two kernels. A per-token kernel writes int8 activations and
      scales; then MPP runs ``int8 x int4b_format`` over the full K straight
      from device memory, choosing its own K tiling and buffering.

``_SPLIT_K_SOURCE`` / ``_SPLIT_K_REDUCE_SOURCE`` (split-K full-K GEMM) and
``_MATVEC_SOURCE`` (small-M matvec) produce bit-identical results to the full-K
kernel. The handler does not emit them yet; ``test/bench_int8_int4_gemm.py``
times them.

Constraints for MLX lowering:
    * ``K`` must be statically known and a multiple of 64.
    * ``N`` must be statically known.
    * Requires an MPP-capable GPU (Metal 4).

Usage::

    import executorch.backends.mlx.custom_kernel_ops.int8_int4_gemm  # noqa: F401

    y = torch.ops.mlx.int8_int4_gemm(x, w_packed, w_scale, full_k=True)
    # x:        (..., K)    fp16 / bf16 / fp32
    # w_packed: (N, K // 2) uint8
    # w_scale:  (N,)        float
    # y:        (..., N)    x.dtype
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.fx.node import Node

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Packing / quantization helpers
# ---------------------------------------------------------------------------


def pack_int4(wq: Tensor) -> Tensor:
    """Pack int4 values in ``[-8, 7]`` of shape ``[N, K]`` into ``uint8 [N, K // 2]``."""
    if wq.shape[-1] % 2 != 0:
        raise ValueError(f"pack_int4: K must be even; got {wq.shape[-1]}")
    nibbles = wq.to(torch.int16) & 0xF
    return (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)


def unpack_int4(w_packed: Tensor) -> Tensor:
    """Inverse of :func:`pack_int4`; returns ``int8 [N, K]``."""
    b = w_packed.to(torch.int16)
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    wq = torch.stack([lo, hi], dim=-1).flatten(-2)
    return torch.where(wq >= 8, wq - 16, wq).to(torch.int8)


def quantize_per_token_int8(x: Tensor) -> tuple[Tensor, Tensor]:
    """Kernel-matching symmetric per-token quantization of ``x [M, K]``.

    Returns ``(xq int8 [M, K], x_scale fp32 [M, 1])``.
    """
    xf = x.float()
    x_scale = (xf.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=_EPS)
    xq = torch.round(xf * torch.reciprocal(x_scale)).clamp(-128, 127)
    return xq.to(torch.int8), x_scale


# ---------------------------------------------------------------------------
# Custom op + eager fallback
# ---------------------------------------------------------------------------


@torch.library.custom_op("mlx::int8_int4_gemm", mutates_args=())
def int8_int4_gemm(
    x: Tensor, w_packed: Tensor, w_scale: Tensor, full_k: bool = False
) -> Tensor:
    K = x.shape[-1]
    xq, x_scale = quantize_per_token_int8(x.reshape(-1, K))
    wq = unpack_int4(w_packed)
    acc = xq.to(torch.int64) @ wq.to(torch.int64).T
    y = (x_scale * w_scale.float()) * acc.float()
    return y.to(x.dtype).reshape(*x.shape[:-1], w_packed.shape[0])


@torch.library.register_fake("mlx::int8_int4_gemm")
def int8_int4_gemm_fake(
    x: Tensor, w_packed: Tensor, w_scale: Tensor, full_k: bool = False
) -> Tensor:
    return x.new_empty((*x.shape[:-1], w_packed.shape[0]))


# ---------------------------------------------------------------------------
# MLX handler
# ---------------------------------------------------------------------------

from executorch.backends.mlx.builder.op_helpers import (
    emit_ceil_div,
    emit_product,
    emit_shape,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    IntOrVid,
    MetalKernelNode,
)

_BM = 32
_BN = 32
_BK = 64
_THREADS = 128
_QUANT_THREADS = 256


_HEADER = f"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant float INT8_GEMM_EPS = {_EPS}f;

// 8 packed bytes -> 16 int8 values in K order (low nibble first).
inline uint4 unpack_int4x16(uint2 packed) {{
  const uchar4 b0 = as_type<uchar4>(packed.x);
  const uchar4 b1 = as_type<uchar4>(packed.y);
  const char4 lo0 = as_type<char4>(uchar4(b0 << 4)) >> 4;
  const char4 hi0 = as_type<char4>(b0) >> 4;
  const char4 lo1 = as_type<char4>(uchar4(b1 << 4)) >> 4;
  const char4 hi1 = as_type<char4>(b1) >> 4;
  return uint4(
      as_type<uint>(char4(lo0.x, hi0.x, lo0.y, hi0.y)),
      as_type<uint>(char4(lo0.z, hi0.z, lo0.w, hi0.w)),
      as_type<uint>(char4(lo1.x, hi1.x, lo1.y, hi1.y)),
      as_type<uint>(char4(lo1.z, hi1.z, lo1.w, hi1.w)));
}}

template <typename T>
inline uint quantize_int8x4(vec<T, 4> v, float inv_scale) {{
  const float4 q = clamp(rint(float4(v) * inv_scale), -128.0f, 127.0f);
  return as_type<uint>(char4(q));
}}

inline float int8_scale_from_amax(float amax) {{
  return max(precise::divide(amax, 127.0f), INT8_GEMM_EPS);
}}
"""


# full_k=False. Template args BM, BN, BK (BK a multiple of 32 dividing K). One
# threadgroup of 128 threads (4 SIMD groups) computes a BM x BN output tile.
#   1. Pre-pass: one SIMD group per tile row reduces |x| over K into the row's
#      scale and inverse scale.
#   2. Main loop per BK step: threads stride over 16-byte chunks, quantizing
#      activations and unpacking int4 weights into threadgroup memory (global
#      loads are issued before the barrier); MPP matmul2d accumulates INT32.
#   3. Epilogue: scale each INT32 sum by x_scale[m] * w_scale[n].
_SOURCE = """
    constexpr int THREADS = 128;
    constexpr int KP = K / 2;
    constexpr int CPR = BK / 16;
    constexpr int X_CHUNKS = BM * CPR;
    constexpr int W_CHUNKS = BN * CPR;
    constexpr int X_PER_THREAD = (X_CHUNKS + THREADS - 1) / THREADS;
    constexpr int W_PER_THREAD = (W_CHUNKS + THREADS - 1) / THREADS;

    int M = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
        M *= x_shape[d];
    }
    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;
    const int tid = int(thread_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int sg = int(simdgroup_index_in_threadgroup);

    threadgroup uint4 x_tile_v[X_CHUNKS];
    threadgroup uint4 w_tile_v[W_CHUNKS];
    threadgroup float row_scale[BM];
    threadgroup float row_inv_scale[BM];

    for (int r = sg; r < BM; r += THREADS / 32) {
        float amax = 0.0f;
        if (m0 + r < M) {
            device const vec<T, 4>* xr =
                (device const vec<T, 4>*)(x + size_t(m0 + r) * K);
            for (int i = lane; i < K / 4; i += 32) {
                const float4 v = abs(float4(xr[i]));
                amax = max(amax, max(max(v.x, v.y), max(v.z, v.w)));
            }
        }
        amax = simd_max(amax);
        if (lane == 0) {
            const float s = int8_scale_from_amax(amax);
            row_scale[r] = s;
            row_inv_scale[r] = precise::divide(1.0f, s);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tensor dimension 0 is contiguous: the X tile is [K, M] and the W tile is
    // [K, N], so W enters transposed.
    using TileX = tensor<threadgroup int8_t, extents<int, BK, BM>, tensor_inline>;
    using TileW = tensor<threadgroup int8_t, extents<int, BK, BN>, tensor_inline>;
    TileX a((threadgroup int8_t*)x_tile_v, extents<int, BK, BM>{});
    TileW b((threadgroup int8_t*)w_tile_v, extents<int, BK, BN>{});
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, BK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto acc = mm.template get_destination_cooperative_tensor<TileX, TileW, int>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        uint4 xv[X_PER_THREAD];
        uint2 wv[W_PER_THREAD];
        #pragma clang loop unroll(full)
        for (int j = 0; j < X_PER_THREAD; ++j) {
            const int c = tid + j * THREADS;
            const int row = c / CPR;
            xv[j] = uint4(0);
            if (c < X_CHUNKS && m0 + row < M) {
                device const vec<T, 4>* src = (device const vec<T, 4>*)(
                    x + size_t(m0 + row) * K + k0 + (c % CPR) * 16);
                const float inv = row_inv_scale[row];
                #pragma clang loop unroll(full)
                for (int q = 0; q < 4; ++q) {
                    xv[j][q] = quantize_int8x4(src[q], inv);
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int j = 0; j < W_PER_THREAD; ++j) {
            const int c = tid + j * THREADS;
            const int row = c / CPR;
            wv[j] = (c < W_CHUNKS && n0 + row < N)
                ? *(device const uint2*)(
                      w_packed + size_t(n0 + row) * KP + (k0 + (c % CPR) * 16) / 2)
                : uint2(0);
        }
        // Also orders the previous step's MMA reads before these writes.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int j = 0; j < X_PER_THREAD; ++j) {
            const int c = tid + j * THREADS;
            if (c < X_CHUNKS) {
                x_tile_v[c] = xv[j];
            }
        }
        #pragma clang loop unroll(full)
        for (int j = 0; j < W_PER_THREAD; ++j) {
            const int c = tid + j * THREADS;
            if (c < W_CHUNKS) {
                w_tile_v[c] = unpack_int4x16(wv[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm.run(a, b, acc);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            const auto coord = acc.get_multidimensional_index(i);
            const int n = n0 + int(coord[0]);
            const int m = m0 + int(coord[1]);
            if (m < M && n < N) {
                const float s = row_scale[coord[1]] * float(w_scale[n]);
                out[size_t(m) * N + n] = T(s * float(acc[i]));
            }
        }
    }
"""


# full_k=True, kernel 1 (template args K, TPR): one threadgroup of TPR threads
# per xq row. Rows below M (from x) write xq and x_scale; padding rows past M
# write zeros to xq so the GEMM sees whole BM tiles.
_QUANT_SOURCE = """
    constexpr int SG = TPR / 32;
    threadgroup float sg_amax[SG];

    int M = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
        M *= x_shape[d];
    }
    const int m = int(threadgroup_position_in_grid.y);
    const int tid = int(thread_index_in_threadgroup);
    device uint* xq_row = (device uint*)(xq + size_t(m) * K);
    if (m >= M) {
        for (int i = tid; i < K / 4; i += TPR) {
            xq_row[i] = 0;
        }
        return;
    }
    device const vec<T, 4>* x_row = (device const vec<T, 4>*)(x + size_t(m) * K);

    float amax = 0.0f;
    for (int i = tid; i < K / 4; i += TPR) {
        const float4 v = abs(float4(x_row[i]));
        amax = max(amax, max(max(v.x, v.y), max(v.z, v.w)));
    }
    amax = simd_max(amax);
    if (SG > 1) {
        if (thread_index_in_simdgroup == 0) {
            sg_amax[simdgroup_index_in_threadgroup] = amax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        amax = sg_amax[0];
        for (int s = 1; s < SG; ++s) {
            amax = max(amax, sg_amax[s]);
        }
    }
    const float scale = int8_scale_from_amax(amax);
    const float inv_scale = precise::divide(1.0f, scale);

    for (int i = tid; i < K / 4; i += TPR) {
        xq_row[i] = quantize_int8x4(x_row[i], inv_scale);
    }
    if (tid == 0) {
        x_scale[m] = scale;
    }
"""


# full_k=True, kernel 2 (template args BM, BN): MPP multiplies an int8 [K, rows]
# tensor by an int4b [K, N] tensor (W transposed) over the whole K for each
# BM x BN output tile. xq may have zero padding rows past M = len(x_scale).
_FULL_K_SOURCE = """
    const int rows = xq_shape[0];
    const int M = x_scale_shape[0];
    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;

    tensor<device int8_t, dextents<int, 2>, tensor_inline> X(
        (device int8_t*)xq, dextents<int, 2>(K, rows));
    tensor<device int4b_format, dextents<int, 2>, tensor_inline> W(
        (device uchar*)w_packed, dextents<int, 2>(K, N), array<int, 2>{1, K});
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto mA = X.slice(0, m0);
    auto mB = W.slice(0, n0);
    auto acc = mm.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), int>();
    mm.run(mA, mB, acc);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            const auto coord = acc.get_multidimensional_index(i);
            const int n = n0 + int(coord[0]);
            const int m = m0 + int(coord[1]);
            if (m < M && n < N) {
                const float s = x_scale[m] * float(w_scale[n]);
                out[size_t(m) * N + n] = T(s * float(acc[i]));
            }
        }
    }
"""


# Split-K full-K, kernel 2 (template args BM, BN, SPLITS; K % (64 * SPLITS)
# == 0): grid z picks one of SPLITS contiguous K ranges. Each threadgroup runs
# MPP over its range and writes exact INT32 partial sums to partial[s, m, n].
_SPLIT_K_SOURCE = """
    constexpr int KS = K / SPLITS;
    const int rows = xq_shape[0];
    const int M = x_scale_shape[0];
    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;
    const int s = int(threadgroup_position_in_grid.z);
    const int k0 = s * KS;

    tensor<device int8_t, dextents<int, 2>, tensor_inline> X(
        (device int8_t*)xq + k0, dextents<int, 2>(KS, rows), array<int, 2>{1, K});
    tensor<device int4b_format, dextents<int, 2>, tensor_inline> W(
        (device uchar*)w_packed + k0 / 2, dextents<int, 2>(KS, N),
        array<int, 2>{1, K});
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto mA = X.slice(0, m0);
    auto mB = W.slice(0, n0);
    auto acc = mm.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), int>();
    mm.run(mA, mB, acc);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            const auto coord = acc.get_multidimensional_index(i);
            const int n = n0 + int(coord[0]);
            const int m = m0 + int(coord[1]);
            if (m < M && n < N) {
                partial[(size_t(s) * M + m) * N + n] = acc[i];
            }
        }
    }
"""


# Split-K, kernel 3 (template args T, N, SPLITS): one thread per output sums
# the INT32 partials and applies the scales in the same order as the full-K
# epilogue, so results match the unsplit kernel exactly.
_SPLIT_K_REDUCE_SOURCE = """
    const int M = x_scale_shape[0];
    const size_t idx = thread_position_in_grid.x;
    if (idx >= size_t(M) * N) {
        return;
    }
    const int m = int(idx / N);
    const int n = int(idx % N);
    int sum = 0;
    for (int s = 0; s < SPLITS; ++s) {
        sum += partial[size_t(s) * M * N + idx];
    }
    const float scale = x_scale[m] * float(w_scale[n]);
    out[idx] = T(scale * float(sum));
"""


_MATVEC_SIMDGROUPS = 2
_MATVEC_ROWS_PER_SIMDGROUP = 4
_MATVEC_BLOCK = 512

# Matvec (template args T, N, K, MR; K % 512 == 0) for small M: reads int8
# activations from the quantize kernel (unpadded) and streams packed weights
# like MLX's qmv. Each SIMD group computes 4 output channels for up to MR
# activation rows; each lane covers 16 K values per 512-wide step. XOR 0x88
# makes nibbles unsigned (q = w + 8) so masked ushort words convert directly to
# float against activations pre-scaled by 1, 1/16, 1/256, 1/4096 (exact in
# fp32); subtracting 8 * sum(x) restores the signed dot product. Per-lane sums
# stay below 2^24, so they are exact integers and the INT32 simd_sum matches the
# MPP kernels bit for bit.
_MATVEC_SOURCE = """
    constexpr int SG = 2;
    constexpr int RPS = 4;
    constexpr int VPT = 16;
    constexpr int BLOCK = 32 * VPT;
    constexpr int KP = K / 2;

    const int M = x_scale_shape[0];
    const int lane = int(thread_index_in_simdgroup);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int n0 = (int(threadgroup_position_in_grid.x) * SG + sg) * RPS;
    const int m0 = int(threadgroup_position_in_grid.y) * MR;
    if (n0 >= N) {
        return;
    }
    const int mrows = min(MR, M - m0);

    device const uchar* w_rows[RPS];
    #pragma clang loop unroll(full)
    for (int r = 0; r < RPS; ++r) {
        w_rows[r] = w_packed + size_t(min(n0 + r, N - 1)) * KP + lane * (VPT / 2);
    }
    device const char* x_base = (device const char*)xq + size_t(m0) * K + lane * VPT;
    const float4 prescale =
        float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

    float acc[MR][RPS];
    float xsum[MR];
    #pragma clang loop unroll(full)
    for (int m = 0; m < MR; ++m) {
        xsum[m] = 0.0f;
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPS; ++r) {
            acc[m][r] = 0.0f;
        }
    }

    for (int k = 0; k < K; k += BLOCK) {
        // Converted once per block and reused for every activation row.
        float4 wf[RPS][4];
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPS; ++r) {
            const ushort4 q = as_type<ushort4>(
                *(device const uint2*)(w_rows[r] + k / 2) ^ uint2(0x88888888u));
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                wf[r][i] = float4(
                    q[i] & 0x000f, q[i] & 0x00f0, q[i] & 0x0f00, q[i] & 0xf000);
            }
        }
        #pragma clang loop unroll(full)
        for (int m = 0; m < MR; ++m) {
            if (m < mrows) {
                const uint4 xb = *(device const uint4*)(x_base + size_t(m) * K + k);
                float4 xs[4];
                #pragma clang loop unroll(full)
                for (int i = 0; i < 4; ++i) {
                    const float4 v = float4(as_type<char4>(xb[i]));
                    xsum[m] += (v.x + v.y) + (v.z + v.w);
                    xs[i] = v * prescale;
                }
                #pragma clang loop unroll(full)
                for (int r = 0; r < RPS; ++r) {
                    float a = 0.0f;
                    #pragma clang loop unroll(full)
                    for (int i = 0; i < 4; ++i) {
                        a += dot(xs[i], wf[r][i]);
                    }
                    acc[m][r] += a;
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (int m = 0; m < MR; ++m) {
        if (m < mrows) {
            #pragma clang loop unroll(full)
            for (int r = 0; r < RPS; ++r) {
                const int total = simd_sum(int(acc[m][r] - 8.0f * xsum[m]));
                const int n = n0 + r;
                if (lane == 0 && n < N) {
                    const float s = x_scale[m0 + m] * float(w_scale[n]);
                    out[size_t(m0 + m) * N + n] = T(s * float(total));
                }
            }
        }
    }
"""


def _lit(*values: int) -> list[IntOrVid]:
    return [IntOrVid.from_literal(v) for v in values]


@REGISTRY.register(target=[torch.ops.mlx.int8_int4_gemm.default])
def _int8_int4_gemm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    if len(args) not in (3, 4):
        raise ValueError(f"mlx::int8_int4_gemm: expected 3-4 args, got {len(args)}")
    x_slot, w_packed_slot, w_scale_slot = args[:3]
    full_k = bool(args[3]) if len(args) == 4 else bool(n.kwargs.get("full_k", False))
    x_node = n.args[0]
    x_meta = x_node.meta["val"]
    w_packed_meta = n.args[1].meta["val"]

    N, KP = w_packed_meta.shape
    if not isinstance(N, int) or not isinstance(KP, int):
        raise NotImplementedError(
            "mlx::int8_int4_gemm: w_packed shape must be statically known"
        )
    K = 2 * KP
    if K % _BK != 0:
        raise NotImplementedError(
            f"mlx::int8_int4_gemm: K must be a multiple of {_BK}; got K={K}"
        )
    if x_meta.shape[-1] != K:
        raise ValueError(
            f"mlx::int8_int4_gemm: x last dim {x_meta.shape[-1]} does not match "
            f"K={K} from w_packed"
        )

    dtype_int = torch_dtype_to_scalar_type(x_meta.dtype)
    out = P.make_or_get_slot(n)
    leading = emit_shape(P, x_node, x_slot, end_dim=-1)
    out_shape_flat = leading + [IntOrVid.from_literal(N)]
    m_iov = emit_product(P, leading)
    gemm_grid = [
        IntOrVid.from_literal((N + _BN - 1) // _BN * _THREADS),
        emit_ceil_div(P, m_iov, _BM),
        IntOrVid.from_literal(1),
    ]

    if not full_k:
        P.emit(
            MetalKernelNode(
                name="int8_int4_gemm",
                source=_SOURCE,
                header=_HEADER,
                inputs=[
                    P.slot_to_tid(s) for s in (x_slot, w_packed_slot, w_scale_slot)
                ],
                outputs=[P.slot_to_tid(out)],
                grid=gemm_grid,
                threadgroup=_lit(_THREADS, 1, 1),
                input_names=["x", "w_packed", "w_scale"],
                output_names=["out"],
                output_shapes_flat=out_shape_flat,
                output_shape_lengths=[len(out_shape_flat)],
                output_dtypes=[dtype_int],
                template_arg_names=["T", "N", "K", "BM", "BN", "BK"],
                template_arg_kinds=[2, 0, 0, 0, 0, 0],
                template_arg_values=[dtype_int, N, K, _BM, _BN, _BK],
            )
        )
        return out

    _, xq_slot = P.make_tmp_slot()
    _, x_scale_slot = P.make_tmp_slot()
    P.emit(
        MetalKernelNode(
            name="int8_int4_gemm_quantize",
            source=_QUANT_SOURCE,
            header=_HEADER,
            inputs=[P.slot_to_tid(x_slot)],
            outputs=[P.slot_to_tid(xq_slot), P.slot_to_tid(x_scale_slot)],
            grid=[
                IntOrVid.from_literal(_QUANT_THREADS),
                m_iov,
                IntOrVid.from_literal(1),
            ],
            threadgroup=_lit(_QUANT_THREADS, 1, 1),
            input_names=["x"],
            output_names=["xq", "x_scale"],
            output_shapes_flat=[m_iov, IntOrVid.from_literal(K), m_iov],
            output_shape_lengths=[2, 1],
            output_dtypes=[
                torch_dtype_to_scalar_type(torch.int8),
                torch_dtype_to_scalar_type(torch.float32),
            ],
            template_arg_names=["T", "K", "TPR"],
            template_arg_kinds=[2, 0, 0],
            template_arg_values=[dtype_int, K, _QUANT_THREADS],
        )
    )
    P.emit(
        MetalKernelNode(
            name="int8_int4_gemm_full_k",
            source=_FULL_K_SOURCE,
            header=_HEADER,
            inputs=[
                P.slot_to_tid(s)
                for s in (xq_slot, x_scale_slot, w_packed_slot, w_scale_slot)
            ],
            outputs=[P.slot_to_tid(out)],
            grid=gemm_grid,
            threadgroup=_lit(_THREADS, 1, 1),
            input_names=["xq", "x_scale", "w_packed", "w_scale"],
            output_names=["out"],
            output_shapes_flat=out_shape_flat,
            output_shape_lengths=[len(out_shape_flat)],
            output_dtypes=[dtype_int],
            template_arg_names=["T", "N", "K", "BM", "BN"],
            template_arg_kinds=[2, 0, 0, 0, 0],
            template_arg_values=[dtype_int, N, K, _BM, _BN],
        )
    )
    return out
