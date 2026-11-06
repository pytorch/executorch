#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
``mlx::int8_intx_gemm``: 8-bit dynamic activation x 4- or 2-bit per-channel
weight linear, lowered to Metal Performance Primitives (Metal 4) kernels that
do INT8 x INT{4,2} -> INT32 accumulation over the full K dimension.

Quantization scheme (symmetric, no zero points), with ``bits`` in {4, 2} and
``qmin, qmax = -2**(bits - 1), 2**(bits - 1) - 1``:

    x_scale[m] = max(max_k |x[m, k]|, eps) / 127                   per token
    xq[m, k]   = clamp(round(x[m, k] * (1 / x_scale[m])), -128, 127)
    W[n, k]    = w_scale[n] * wq[n, k],  wq in [qmin, qmax]      per channel
    Y[m, n]    = x_scale[m] * w_scale[n] * sum_k xq[m, k] * wq[n, k]

``round`` is round-half-to-even and the scale math is fp32, so the kernels and
the eager reference quantize identically.

Weight packing (see :func:`pack_intx`): ``w_packed`` is
``uint8 [N, K * bits // 8]``; element ``k`` of row ``n`` sits in byte
``k * bits // 8`` at bit offset ``(k * bits) % 8`` (low bits first), as
two's-complement ``bits``-bit integers (the layout of Metal's ``int4b_format``
and ``int2b_format``). ``bits`` is inferred from ``w_packed``'s width.

INT32 accumulation cannot overflow while ``K * 128 * 8 < 2**31``.

Lowering: a per-token kernel writes int8 activations and scales; then the GEMM
is picked at runtime from M (see :func:`full_k_plans`): a matvec for M <= 2,
otherwise MPP ``int8 x int{4,2}b_format`` over the full K straight from device
memory, on activations zero-padded to whole BM tiles, split over K into INT32
partials plus a reduce kernel when that adds enough threadgroups. All paths
produce bit-identical results.

Constraints for MLX lowering:
    * ``K`` must be statically known and a multiple of 64.
    * ``N`` must be statically known.
    * Requires an MPP-capable GPU (Metal 4).

Usage::

    import executorch.backends.mlx.custom_kernel_ops.int8_intx_gemm  # noqa: F401

    y = torch.ops.mlx.int8_intx_gemm(x, w_packed, w_scale)
    # x:        (..., K)              fp16 / bf16 / fp32
    # w_packed: (N, K * bits // 8)    uint8, bits in {4, 2}
    # w_scale:  (N,)                  float
    # y:        (..., N)              x.dtype
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import torch
from torch import nn, Tensor
from torch.fx.node import Node

_EPS = 1e-10
BITS = (4, 2)


# ---------------------------------------------------------------------------
# Packing / quantization helpers
# ---------------------------------------------------------------------------


def _check_bits(bits: int) -> None:
    if bits not in BITS:
        raise ValueError(f"bits must be one of {BITS}; got {bits}")


def pack_intx(wq: Tensor, bits: int) -> Tensor:
    """Pack ``bits``-bit values of shape ``[N, K]`` into
    ``uint8 [N, K * bits // 8]``, low bits first."""
    _check_bits(bits)
    per_byte = 8 // bits
    if wq.shape[-1] % per_byte != 0:
        raise ValueError(
            f"pack_intx: K must be a multiple of {per_byte}; got {wq.shape[-1]}"
        )
    fields = wq.to(torch.int16) & ((1 << bits) - 1)
    out = fields[..., 0::per_byte]
    for i in range(1, per_byte):
        out = out | (fields[..., i::per_byte] << (i * bits))
    return out.to(torch.uint8)


def unpack_intx(w_packed: Tensor, bits: int) -> Tensor:
    """Inverse of :func:`pack_intx`; returns ``int8 [N, K]``."""
    _check_bits(bits)
    b = w_packed.to(torch.int16)
    mask = (1 << bits) - 1
    wq = torch.stack([(b >> s) & mask for s in range(0, 8, bits)], dim=-1).flatten(-2)
    return torch.where(wq >= 1 << (bits - 1), wq - (1 << bits), wq).to(torch.int8)


def pack_int4(wq: Tensor) -> Tensor:
    return pack_intx(wq, 4)


def unpack_int4(w_packed: Tensor) -> Tensor:
    return unpack_intx(w_packed, 4)


def infer_bits(w_packed: Tensor, K: int) -> int:
    """Weight bit width from ``w_packed [N, K * bits // 8]`` and ``K``."""
    bits = 8 * w_packed.shape[-1] // K
    if bits not in BITS or K * bits != 8 * w_packed.shape[-1]:
        raise ValueError(
            f"w_packed width {w_packed.shape[-1]} is not K * bits // 8 for "
            f"K={K} and bits in {BITS}"
        )
    return bits


def quantize_per_token_int8(x: Tensor) -> tuple[Tensor, Tensor]:
    """Kernel-matching symmetric per-token quantization of ``x [M, K]``.

    Returns ``(xq int8 [M, K], x_scale fp32 [M, 1])``.
    """
    xf = x.float()
    x_scale = (xf.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=_EPS)
    xq = torch.round(xf * torch.reciprocal(x_scale)).clamp(-128, 127)
    return xq.to(torch.int8), x_scale


def quantize_intx_per_channel(
    rows: Callable[[int, int], Tensor], N: int, bits: int, chunk_rows: int = 4096
) -> tuple[Tensor, Tensor]:
    """Symmetric per-output-channel ``bits``-bit quantization of an ``[N, K]``
    weight. The scale maps max |w| to ``2**(bits - 1) - 0.5`` (7.5 for int4,
    1.5 for int2) so the range is used symmetrically before clamping.

    ``rows(start, end)`` returns rows ``[start, end)`` as a float tensor, so a
    large weight is only ever materialized ``chunk_rows`` rows at a time.
    Returns ``(w_packed uint8 [N, K * bits // 8], w_scale fp32 [N])``.
    """
    _check_bits(bits)
    qmax = (1 << (bits - 1)) - 1
    packed, scales = [], []
    for start in range(0, N, chunk_rows):
        w = rows(start, min(start + chunk_rows, N)).float()
        scale = (w.abs().amax(dim=-1) / (qmax + 0.5)).clamp(min=_EPS)
        wq = torch.round(w / scale[:, None]).clamp(-qmax - 1, qmax)
        packed.append(pack_intx(wq, bits))
        scales.append(scale)
        del w, wq
    return torch.cat(packed), torch.cat(scales)


# ---------------------------------------------------------------------------
# Custom op + eager fallback
# ---------------------------------------------------------------------------


class Int8IntxLinear(nn.Module):
    """Bias-free linear computed by ``mlx::int8_intx_gemm``."""

    def __init__(self, w_packed: Tensor, w_scale: Tensor, bits: int):
        super().__init__()
        _check_bits(bits)
        self.bits = bits
        self.in_features = w_packed.shape[1] * 8 // bits
        self.out_features = w_packed.shape[0]
        self.register_buffer("w_packed", w_packed)
        self.register_buffer("w_scale", w_scale.float())

    def _apply(self, fn, recurse=True):
        # Keep fp32 scales through model.to(activation_dtype); move device only.
        w_scale = self.w_scale
        result = super()._apply(fn, recurse)
        self.w_scale = w_scale.to(self.w_packed.device)
        return result

    def forward(self, x: Tensor) -> Tensor:
        return torch.ops.mlx.int8_intx_gemm(x, self.w_packed, self.w_scale)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}"
        )


@torch.library.custom_op("mlx::int8_intx_gemm", mutates_args=())
def int8_intx_gemm(x: Tensor, w_packed: Tensor, w_scale: Tensor) -> Tensor:
    K = x.shape[-1]
    xq, x_scale = quantize_per_token_int8(x.reshape(-1, K))
    wq = unpack_intx(w_packed, infer_bits(w_packed, K))
    acc = xq.to(torch.int64) @ wq.to(torch.int64).T
    y = (x_scale * w_scale.float()) * acc.float()
    return y.to(x.dtype).reshape(*x.shape[:-1], w_packed.shape[0])


@torch.library.register_fake("mlx::int8_intx_gemm")
def int8_intx_gemm_fake(x: Tensor, w_packed: Tensor, w_scale: Tensor) -> Tensor:
    return x.new_empty((*x.shape[:-1], w_packed.shape[0]))


# ---------------------------------------------------------------------------
# MLX handler
# ---------------------------------------------------------------------------

from executorch.backends.mlx.builder.op_helpers import (
    emit_ceil_div,
    emit_floordiv,
    emit_if_else,
    emit_product,
    emit_shape,
    emit_sub_int,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    IntOrVid,
    MetalKernelNode,
)

_K_ALIGN = 64
_THREADS = 128
_QUANT_THREADS = 256
_MATVEC_SIMDGROUPS = 2
_MATVEC_ROWS_PER_SIMDGROUP = 4
_MATVEC_BLOCK = 512
_MAX_M = 2**31 - 1


class Int8IntxPlan(NamedTuple):
    """Kernel choice for ``M <= max_m``. ``bn == 0`` selects the matvec kernel
    with ``bm`` activation rows per threadgroup; otherwise a padded full-K MPP
    GEMM with a ``bm x bn`` tile, split over K when ``splits > 1``."""

    max_m: int
    bm: int
    bn: int
    splits: int = 1


# Muse Glimmer (N, K) -> (bm, bn, splits) for M <= 8, M <= 16, and larger M,
# from the int4 M4 Max sweep (test/bench_muse_glimmer_gemm.py --m 1 2 4 8 16 256,
# fp16). lm_head uses full-K past M = 16: split-K was within noise
# there and would need a [2, M, 202048] int32 partial buffer. int2 has not been
# swept and uses the int4 tiles.
_MG_TILES: dict[tuple[int, int], list[tuple[int, int, int]]] = {
    (8704, 6656): [(8, 128, 8), (16, 128, 8), (64, 64, 4)],  # qkv
    (6656, 4096): [(8, 64, 8), (16, 64, 8), (128, 32, 2)],  # o
    (39936, 6656): [(8, 128, 4), (16, 128, 8), (32, 128, 1)],  # gate_up
    (6656, 19968): [(8, 64, 8), (16, 64, 8), (32, 128, 4)],  # down
    (202048, 6656): [(8, 128, 2), (16, 128, 2), (64, 64, 1)],  # lm_head
}


def full_k_plans(N: int, K: int) -> list[Int8IntxPlan]:
    """Kernel plans, smallest ``max_m`` first; the last covers every larger M.
    Matvec for M <= 2, then padded (split-K) MPP tiles. Other shapes use the
    Muse Glimmer pattern: more splits when N gives few output tiles."""
    plans = []
    if K % _MATVEC_BLOCK == 0:
        plans += [Int8IntxPlan(1, 1, 0), Int8IntxPlan(2, 2, 0)]
    tiles = _MG_TILES.get((N, K))
    if tiles is None:
        bn, splits = (64, 8) if N <= 16384 else (128, 2)
        tiles = [(8, bn, splits), (16, bn, splits), (32, 128, 1)]
    for max_m, (bm, bn, splits) in zip((8, 16, _MAX_M), tiles):
        while K % (_K_ALIGN * splits):
            splits //= 2
        plans.append(Int8IntxPlan(max_m, bm, bn, splits))
    return plans


_HEADER = f"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant float INT8_GEMM_EPS = {_EPS}f;

template <typename T>
inline uint quantize_int8x4(vec<T, 4> v, float inv_scale) {{
  const float4 q = clamp(rint(float4(v) * inv_scale), -128.0f, 127.0f);
  return as_type<uint>(char4(q));
}}

inline float int8_scale_from_amax(float amax) {{
  return max(precise::divide(amax, 127.0f), INT8_GEMM_EPS);
}}
"""

_W_FORMAT = {4: "int4b_format", 2: "int2b_format"}


# Template args T, K, TPR: one threadgroup of TPR threads per xq row. Rows
# below M (from x) write xq and x_scale; padding rows past M write zeros to xq
# so the GEMM sees whole BM tiles.
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


# Template args T, N, K, BM, BN; {W_FORMAT} is substituted per bit width. MPP
# multiplies an int8 [K, rows] tensor by a packed [K, N] weight tensor (W
# transposed) over the whole K for each BM x BN output tile. xq may have zero
# padding rows past M = len(x_scale).
def _full_k_source(bits: int) -> str:
    return """
    const int rows = xq_shape[0];
    const int M = x_scale_shape[0];
    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;

    tensor<device int8_t, dextents<int, 2>, tensor_inline> X(
        (device int8_t*)xq, dextents<int, 2>(K, rows));
    tensor<device W_FORMAT, dextents<int, 2>, tensor_inline> W(
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
""".replace(
        "W_FORMAT", _W_FORMAT[bits]
    )


# Template args T, N, K, BM, BN, SPLITS (K % (64 * SPLITS) == 0): grid z picks
# one of SPLITS contiguous K ranges. Each threadgroup runs MPP over its range
# and writes exact INT32 partial sums to partial[s, m, n].
def _split_k_source(bits: int) -> str:
    return """
    constexpr int KS = K / SPLITS;
    const int rows = xq_shape[0];
    const int M = x_scale_shape[0];
    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;
    const int s = int(threadgroup_position_in_grid.z);
    const int k0 = s * KS;

    tensor<device int8_t, dextents<int, 2>, tensor_inline> X(
        (device int8_t*)xq + k0, dextents<int, 2>(KS, rows), array<int, 2>{1, K});
    tensor<device W_FORMAT, dextents<int, 2>, tensor_inline> W(
        (device uchar*)w_packed + k0 / (8 / BITS), dextents<int, 2>(KS, N),
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
""".replace(
        "W_FORMAT", _W_FORMAT[bits]
    ).replace(
        "BITS", str(bits)
    )


# Template args T, N, SPLITS: one thread per output sums the INT32 partials and
# applies the scales in the same order as the full-K epilogue, so results match
# the unsplit kernel exactly.
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


# Matvec (template args T, N, K, MR; K % 512 == 0) for small M: reads int8
# activations from the quantize kernel (unpadded) and streams packed weights
# like MLX's qmv. Each SIMD group computes 4 output channels for up to MR
# activation rows; each lane covers 16 K values per 512-wide step. XORing every
# field's sign bit makes fields unsigned (q = w + 2**(bits - 1)) so masked
# ushort words convert directly to float against activations pre-scaled by the
# matching power of two (exact in fp32); subtracting 2**(bits - 1) * sum(x)
# restores the signed dot product. Per-lane sums stay below 2^24, so they are
# exact integers and the INT32 simd_sum matches the MPP kernels bit for bit.
_MATVEC_PROLOGUE = """
    constexpr int SG = 2;
    constexpr int RPS = 4;
    constexpr int VPT = 16;
    constexpr int BLOCK = 32 * VPT;
    constexpr int KP = K * BITS / 8;

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
        w_rows[r] = w_packed + size_t(min(n0 + r, N - 1)) * KP + lane * (VPT * BITS / 8);
    }
    device const char* x_base = (device const char*)xq + size_t(m0) * K + lane * VPT;

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
"""

_MATVEC_LOAD = {
    4: """
            const ushort4 q = as_type<ushort4>(
                *(device const uint2*)(w_rows[r] + k / 2) ^ uint2(0x88888888u));
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                wf[r][i] = float4(
                    q[i] & 0x000f, q[i] & 0x00f0, q[i] & 0x0f00, q[i] & 0xf000);
            }
""",
    2: """
            const ushort2 q = as_type<ushort2>(
                *(device const uint*)(w_rows[r] + k / 4) ^ 0xAAAAAAAAu);
            #pragma clang loop unroll(full)
            for (int h = 0; h < 2; ++h) {
                wf[r][2 * h] = float4(
                    q[h] & 0x0003, q[h] & 0x000c, q[h] & 0x0030, q[h] & 0x00c0);
                wf[r][2 * h + 1] = float4(
                    q[h] & 0x0300, q[h] & 0x0c00, q[h] & 0x3000, q[h] & 0xc000);
            }
""",
}

_MATVEC_BODY = """
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
                    xs[i] = v * PRESCALE;
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
                const int total = simd_sum(int(acc[m][r] - OFFSET * xsum[m]));
                const int n = n0 + r;
                if (lane == 0 && n < N) {
                    const float s = x_scale[m0 + m] * float(w_scale[n]);
                    out[size_t(m0 + m) * N + n] = T(s * float(total));
                }
            }
        }
    }
"""


# Activation pre-scales that undo each weight field's bit position within its
# ushort word; int2 groups of 4 alternate between the low and high byte.
_MATVEC_PRESCALE = {
    4: "float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f)",
    2: (
        "((i % 2) ? float4(1.0f / 256.0f, 1.0f / 1024.0f, 1.0f / 4096.0f, "
        "1.0f / 16384.0f) : float4(1.0f, 1.0f / 4.0f, 1.0f / 16.0f, 1.0f / 64.0f))"
    ),
}


def _matvec_source(bits: int) -> str:
    body = _MATVEC_BODY.replace("PRESCALE", _MATVEC_PRESCALE[bits]).replace(
        "OFFSET", f"{float(1 << (bits - 1))}f"
    )
    return (_MATVEC_PROLOGUE + _MATVEC_LOAD[bits] + body).replace("BITS", str(bits))


_FULL_K_SOURCES = {b: _full_k_source(b) for b in BITS}
_SPLIT_K_SOURCES = {b: _split_k_source(b) for b in BITS}
_MATVEC_SOURCES = {b: _matvec_source(b) for b in BITS}


def _lit(*values: int) -> list[IntOrVid]:
    return [IntOrVid.from_literal(v) for v in values]


@REGISTRY.register(target=[torch.ops.mlx.int8_intx_gemm.default])
def _int8_intx_gemm_handler(P: MLXProgramBuilder, n: Node) -> Slot:  # noqa: C901
    x_slot, w_packed_slot, w_scale_slot = P.args(n)
    x_node = n.args[0]
    x_meta = x_node.meta["val"]
    N, KP = n.args[1].meta["val"].shape
    K = x_meta.shape[-1]
    if not all(isinstance(v, int) for v in (N, KP, K)):
        raise NotImplementedError(
            "mlx::int8_intx_gemm: w_packed shape and K must be statically known"
        )
    bits = 8 * KP // K
    if bits not in BITS or K * bits != 8 * KP:
        raise ValueError(
            f"mlx::int8_intx_gemm: w_packed width {KP} is not K * bits // 8 for "
            f"K={K} and bits in {BITS}"
        )
    if K % _K_ALIGN != 0:
        raise NotImplementedError(
            f"mlx::int8_intx_gemm: K must be a multiple of {_K_ALIGN}; got K={K}"
        )

    dtype_int = torch_dtype_to_scalar_type(x_meta.dtype)
    out = P.make_or_get_slot(n)
    leading = emit_shape(P, x_node, x_slot, end_dim=-1)
    out_shape_flat = leading + [IntOrVid.from_literal(N)]
    m_iov = emit_product(P, leading)
    _, xq_slot = P.make_tmp_slot()
    _, x_scale_slot = P.make_tmp_slot()
    _, partial_slot = P.make_tmp_slot()
    int8 = torch_dtype_to_scalar_type(torch.int8)
    int32 = torch_dtype_to_scalar_type(torch.int32)
    fp32 = torch_dtype_to_scalar_type(torch.float32)
    k_lit = IntOrVid.from_literal(K)
    n_lit = IntOrVid.from_literal(N)
    tag = f"int8_int{bits}_gemm"

    def emit_quantize(rows: IntOrVid) -> None:
        # rows > M zero-pads xq to whole BM tiles; x_scale stays [M].
        P.emit(
            MetalKernelNode(
                name="int8_intx_gemm_quantize",
                source=_QUANT_SOURCE,
                header=_HEADER,
                inputs=[P.slot_to_tid(x_slot)],
                outputs=[P.slot_to_tid(xq_slot), P.slot_to_tid(x_scale_slot)],
                grid=[
                    IntOrVid.from_literal(_QUANT_THREADS),
                    rows,
                    IntOrVid.from_literal(1),
                ],
                threadgroup=_lit(_QUANT_THREADS, 1, 1),
                input_names=["x"],
                output_names=["xq", "x_scale"],
                output_shapes_flat=[rows, k_lit, m_iov],
                output_shape_lengths=[2, 1],
                output_dtypes=[int8, fp32],
                template_arg_names=["T", "K", "TPR"],
                template_arg_kinds=[2, 0, 0],
                template_arg_values=[dtype_int, K, _QUANT_THREADS],
            )
        )

    def emit_matvec(mr: int) -> None:
        emit_quantize(m_iov)
        rows_per_tg = _MATVEC_SIMDGROUPS * _MATVEC_ROWS_PER_SIMDGROUP
        P.emit(
            MetalKernelNode(
                name=f"{tag}_matvec",
                source=_MATVEC_SOURCES[bits],
                header=_HEADER,
                inputs=[
                    P.slot_to_tid(s)
                    for s in (xq_slot, x_scale_slot, w_packed_slot, w_scale_slot)
                ],
                outputs=[P.slot_to_tid(out)],
                grid=[
                    IntOrVid.from_literal(
                        (N + rows_per_tg - 1) // rows_per_tg * _MATVEC_SIMDGROUPS * 32
                    ),
                    emit_ceil_div(P, m_iov, mr),
                    IntOrVid.from_literal(1),
                ],
                threadgroup=_lit(_MATVEC_SIMDGROUPS * 32, 1, 1),
                input_names=["xq", "x_scale", "w_packed", "w_scale"],
                output_names=["out"],
                output_shapes_flat=out_shape_flat,
                output_shape_lengths=[len(out_shape_flat)],
                output_dtypes=[dtype_int],
                template_arg_names=["T", "N", "K", "MR"],
                template_arg_kinds=[2, 0, 0, 0],
                template_arg_values=[dtype_int, N, K, mr],
            )
        )

    def emit_mpp(bm: int, bn: int, splits: int) -> None:
        emit_quantize(
            emit_product(P, [emit_ceil_div(P, m_iov, bm), IntOrVid.from_literal(bm)])
        )
        grid = [
            IntOrVid.from_literal((N + bn - 1) // bn * _THREADS),
            emit_ceil_div(P, m_iov, bm),
            IntOrVid.from_literal(1),
        ]
        if splits == 1:
            P.emit(
                MetalKernelNode(
                    name=f"{tag}_full_k",
                    source=_FULL_K_SOURCES[bits],
                    header=_HEADER,
                    inputs=[
                        P.slot_to_tid(s)
                        for s in (xq_slot, x_scale_slot, w_packed_slot, w_scale_slot)
                    ],
                    outputs=[P.slot_to_tid(out)],
                    grid=grid,
                    threadgroup=_lit(_THREADS, 1, 1),
                    input_names=["xq", "x_scale", "w_packed", "w_scale"],
                    output_names=["out"],
                    output_shapes_flat=out_shape_flat,
                    output_shape_lengths=[len(out_shape_flat)],
                    output_dtypes=[dtype_int],
                    template_arg_names=["T", "N", "K", "BM", "BN"],
                    template_arg_kinds=[2, 0, 0, 0, 0],
                    template_arg_values=[dtype_int, N, K, bm, bn],
                )
            )
            return
        P.emit(
            MetalKernelNode(
                name=f"{tag}_split_k",
                source=_SPLIT_K_SOURCES[bits],
                header=_HEADER,
                inputs=[
                    P.slot_to_tid(s) for s in (xq_slot, x_scale_slot, w_packed_slot)
                ],
                outputs=[P.slot_to_tid(partial_slot)],
                grid=grid[:2] + [IntOrVid.from_literal(splits)],
                threadgroup=_lit(_THREADS, 1, 1),
                input_names=["xq", "x_scale", "w_packed"],
                output_names=["partial"],
                output_shapes_flat=[IntOrVid.from_literal(splits), m_iov, n_lit],
                output_shape_lengths=[3],
                output_dtypes=[int32],
                template_arg_names=["T", "N", "K", "BM", "BN", "SPLITS"],
                template_arg_kinds=[2, 0, 0, 0, 0, 0],
                template_arg_values=[dtype_int, N, K, bm, bn, splits],
            )
        )
        reduce_threads = emit_product(
            P,
            [
                emit_ceil_div(P, emit_product(P, [m_iov, n_lit]), 256),
                IntOrVid.from_literal(256),
            ],
        )
        P.emit(
            MetalKernelNode(
                name="int8_intx_gemm_split_k_reduce",
                source=_SPLIT_K_REDUCE_SOURCE,
                header=_HEADER,
                inputs=[
                    P.slot_to_tid(s) for s in (partial_slot, x_scale_slot, w_scale_slot)
                ],
                outputs=[P.slot_to_tid(out)],
                grid=[
                    reduce_threads,
                    IntOrVid.from_literal(1),
                    IntOrVid.from_literal(1),
                ],
                threadgroup=_lit(256, 1, 1),
                input_names=["partial", "x_scale", "w_scale"],
                output_names=["out"],
                output_shapes_flat=out_shape_flat,
                output_shape_lengths=[len(out_shape_flat)],
                output_dtypes=[dtype_int],
                template_arg_names=["T", "N", "SPLITS"],
                template_arg_kinds=[2, 0, 0],
                template_arg_values=[dtype_int, N, splits],
            )
        )

    def emit_plan(plan: Int8IntxPlan) -> None:
        if plan.bn == 0:
            emit_matvec(plan.bm)
        else:
            emit_mpp(plan.bm, plan.bn, plan.splits)

    # (M - 1) // max_m == 0 exactly when M <= max_m, so each IfNode's else
    # branch takes the first plan that covers M.
    m_minus_1 = emit_sub_int(P, m_iov, IntOrVid.from_literal(1))

    def emit_plans(plans: list[Int8IntxPlan]) -> None:
        plan, rest = plans[0], plans[1:]
        if not rest:
            emit_plan(plan)
            return
        emit_if_else(
            P,
            emit_floordiv(P, m_minus_1, IntOrVid.from_literal(plan.max_m)),
            emit_then=lambda: emit_plans(rest),
            emit_else=lambda: emit_plan(plan),
        )

    emit_plans(full_k_plans(N, K))
    return out
