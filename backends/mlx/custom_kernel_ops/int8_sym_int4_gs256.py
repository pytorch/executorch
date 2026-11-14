#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
``mlx::int8_sym_int4_gs256_gemm``: digantdesai's group-wise int8 x int4 MPP
kernel from the M5 Max study (manifold executorch/tree/digantdesai/
m5_gemm_latest, ``int8_int4_gemm.py`` at commit afc577d626 of its git bundle),
ported unchanged as its own op.

    x_scale[m] = max(max_k |x[m, k]|, eps) / 127              (symmetric, per token)
    xq[m, k]   = clamp(round(x[m, k] / x_scale[m]), -128, 127)
    W[n, k]    = w_s[g, n] * wq[n, k] + w_b[g, n],  g = k // 256
    Y[m, n]    = x_scale[m] * sum_k xq[m, k] * W[n, k]

The kernel sources below (``_HEADER``, ``_SG_ROWS_PER_THREAD`` /
``sg_rows_per_thread``, ``_GROUP_QUANT_VEC_SOURCE``, ``_GROUP_SG_HEADER``,
``_GROUP_INT8_SG_SOURCE``, ``_GROUP_SPLIT_REDUCE_SOURCE``) are copied verbatim.
The lowering launches them exactly as the study's benchmark did
(``bench_int4_decomposition.int8_group_sg_gemm``): the quantize kernel over
``ceil(M / BM) * BM`` rows (zeroing the padding rows), the per-SIMD-group GEMM,
then either the split-K reduce or a slice back to M rows. Tiles are the study's
pinned M5 Max configs per Muse Glimmer shape and M (``analysis/pinned_q``),
written as the study names them: ``sg<bias> BMxBN/SMxSN k<splits>``.

``w_packed`` is ``uint8 [N, K // 2]`` (``int8_intx_gemm.pack_int4``);
``w_s`` / ``w_b`` are fp16 ``[K // 256, N]`` planes, which
``Int8SymInt4Gs256Linear`` builds from ``int4_affine_gemm.quantize_int4_affine``
scales and zero points (``w_b = -scale * zero``).

Constraints for MLX lowering: K and N static, K % 256 == 0, N % 4 == 0, N at
least the plans' BN, fp16 planes, MPP-capable GPU (Metal 4).

Usage::

    import executorch.backends.mlx.custom_kernel_ops.int8_sym_int4_gs256  # noqa: F401

    y = torch.ops.mlx.int8_sym_int4_gs256_gemm(x, w_packed, w_s, w_b)
"""

from __future__ import annotations

import math

import torch
from executorch.backends.mlx.custom_kernel_ops.int4_affine_gemm import (
    dequantize_int4_affine,
    int4_affine_planes,
)
from executorch.backends.mlx.custom_kernel_ops.int8_intx_gemm import (
    quantize_per_token_int8,
)
from torch import nn, Tensor
from torch.fx.node import Node

GROUP_SIZE = 256


class Int8SymInt4Gs256Linear(nn.Module):
    """Bias-free linear computed by ``mlx::int8_sym_int4_gs256_gemm`` from
    ``quantize_int4_affine(..., group_size=256)``'s ``(w_packed, scale,
    zero)``; stores the fp16 ``[G, N]`` planes the kernel reads."""

    def __init__(self, w_packed: Tensor, scale: Tensor, zero: Tensor):
        super().__init__()
        self.in_features = 2 * w_packed.shape[1]
        self.out_features = w_packed.shape[0]
        if self.in_features != GROUP_SIZE * scale.shape[1]:
            raise ValueError(
                f"expected {GROUP_SIZE}-wide groups; got K={self.in_features} with "
                f"{scale.shape[1]} groups"
            )
        w_s, w_b = int4_affine_planes(scale, zero, torch.float16)
        self.register_buffer("w_packed", w_packed)
        self.register_buffer("w_s", w_s)
        self.register_buffer("w_b", w_b)

    def _apply(self, fn, recurse=True):
        # Keep the fp16 planes through model.to(activation_dtype).
        planes = (self.w_s, self.w_b)
        result = super()._apply(fn, recurse)
        device = self.w_packed.device
        self.w_s, self.w_b = (t.to(device) for t in planes)
        return result

    def forward(self, x: Tensor) -> Tensor:
        return torch.ops.mlx.int8_sym_int4_gs256_gemm(
            x, self.w_packed, self.w_s, self.w_b
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


@torch.library.custom_op("mlx::int8_sym_int4_gs256_gemm", mutates_args=())
def int8_sym_int4_gs256_gemm(
    x: Tensor, w_packed: Tensor, w_s: Tensor, w_b: Tensor
) -> Tensor:
    K = x.shape[-1]
    xq, x_scale = quantize_per_token_int8(x.reshape(-1, K))
    y = x_scale * (xq.float() @ dequantize_int4_affine(w_packed, w_s, w_b).T)
    return y.to(x.dtype).reshape(*x.shape[:-1], w_packed.shape[0])


@torch.library.register_fake("mlx::int8_sym_int4_gs256_gemm")
def int8_sym_int4_gs256_gemm_fake(
    x: Tensor, w_packed: Tensor, w_s: Tensor, w_b: Tensor
) -> Tensor:
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
    ReshapeNode,
    SliceNode,
)

# ---- Verbatim from the study's int8_int4_gemm.py ----

_EPS = 1e-10
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


# Distinct accumulator rows per thread of a single-SIMD-group int8 x int4b MPP
# op with an SM x SN tile; other tiles were not measured.
_SG_ROWS_PER_THREAD = {(32, 32): 4, (16, 64): 2, (16, 32): 2}


def sg_rows_per_thread(sm: int, sn: int) -> int:
    return _SG_ROWS_PER_THREAD.get((sm, sn), 4)


# Kernel 1 (template args T, K, TPR, GS; GS % 32 == 0, K % GS == 0): per-token
# int8 like _QUANT_SOURCE plus exact fp32 group sums xs[m, K / GS]; each lane
# quantizes a contiguous GS / 32 chunk of a group (vec4 loads, packed stores).
# Launched over more rows than x has, it zeroes xq and xs past them (the
# per-SIMD-group kernel's padding rows).
_GROUP_QUANT_VEC_SOURCE = """
    constexpr int SG = TPR / 32;
    constexpr int NG = K / GS;
    constexpr int VPL = GS / 32;
    threadgroup float sg_amax[SG];

    int M = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
        M *= x_shape[d];
    }
    const int m = int(threadgroup_position_in_grid.y);
    const int tid = int(thread_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int sg = int(simdgroup_index_in_threadgroup);
    if (m >= M) {
        for (int i = tid; i < K / 4; i += TPR) {
            ((device uint*)(xq + size_t(m) * K))[i] = 0;
        }
        for (int g = tid; g < NG; g += TPR) {
            xs[size_t(m) * NG + g] = 0.0f;
        }
        return;
    }
    device const vec<T, 4>* x4 = (device const vec<T, 4>*)(x + size_t(m) * K);

    float amax = 0.0f;
    for (int i = tid; i < K / 4; i += TPR) {
        const float4 v = abs(float4(x4[i]));
        amax = max(amax, max(max(v.x, v.y), max(v.z, v.w)));
    }
    amax = simd_max(amax);
    if (SG > 1) {
        if (lane == 0) {
            sg_amax[sg] = amax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        amax = sg_amax[0];
        for (int s = 1; s < SG; ++s) {
            amax = max(amax, sg_amax[s]);
        }
    }
    const float scale = int8_scale_from_amax(amax);
    const float inv_scale = precise::divide(1.0f, scale);

    device const T* xr = x + size_t(m) * K;
    device int8_t* qr = xq + size_t(m) * K;
    for (int g = sg; g < NG; g += SG) {
        const int k0 = g * GS + lane * VPL;
        int sum = 0;
        if (VPL % 4 == 0) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < VPL; j += 4) {
                const uint q = quantize_int8x4(
                    *(device const vec<T, 4>*)(xr + k0 + j), inv_scale);
                *(device uint*)(qr + k0 + j) = q;
                const char4 c = as_type<char4>(q);
                sum += (int(c.x) + int(c.y)) + (int(c.z) + int(c.w));
            }
        } else {
            #pragma clang loop unroll(full)
            for (int j = 0; j < VPL; ++j) {
                const int q = int(
                    clamp(rint(float(xr[k0 + j]) * inv_scale), -128.0f, 127.0f));
                qr[k0 + j] = int8_t(q);
                sum += q;
            }
        }
        sum = simd_sum(sum);
        if (lane == 0) {
            xs[size_t(m) * NG + g] = float(sum);
        }
    }
    if (tid == 0) {
        x_scale[m] = scale;
    }
"""

# Destination layout of a single-SIMD-group int8 x int4b MPP op (measured for
# 32 x 32, 16 x 64, 16 x 32): element i sits at column rn[(i / 8) % NRN] + i % 4
# and row rm[(i / (8 * NRN)) * 2 + (i / 4) % 2], with NM rows per thread and
# NRN = capacity / (4 * NM) runs of 4 adjacent columns.
_GROUP_SG_HEADER = (
    _HEADER
    + """
template <int CAP, int NRN, typename CI, typename CF>
inline void group_rescale(
    thread CI& acc, thread CF& facc, thread const half4 (&s)[NRN]) {
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < CAP; ++i) {
    facc[i] = fma(float(acc[i]), float(s[(i / 8) % NRN][i % 4]), facc[i]);
  }
}
"""
)

# Kernel 2 (template args T, N, K, BM, BN, SM, SN, GS, NM, BIAS, SPLITS;
# M % BM == 0, N >= BN, N % 4 == 0, no empty split): each SIMD group of a
# BM x BN threadgroup runs its own
# SM x SN MPP op (execution_simdgroup, so SIMD groups do not rescale in
# lockstep). Per group, MPP multiplies int8 [GS, SM] by int4b [GS, SN] into
# INT32 and facc += s * acc, with the group's scales for the thread's column
# runs loaded as half4 (one group ahead). The bias term sum_g xs[m, g] * b'[g, n]
# is one f32 x f16 MPP call into facc through a threadgroup tile (BIAS = 1) or
# one pass over the groups after the loop (BIAS = 3). The last column tile
# shifts back to N - BN (overlapping columns are written twice, identically).
# Each thread checks the layout above once; if it differs, a per-element path
# with the same math runs instead. SPLITS > 1 (split-K for small M): grid z
# picks ceil(NG / SPLITS) consecutive groups (fewer in the last split), and the
# unscaled fp32 sums go to
# out[split, m, n] for rows below x_scale's length (pass the unpadded x_scale)
# for _GROUP_SPLIT_REDUCE_SOURCE.
_GROUP_INT8_SG_SOURCE = """
    constexpr int NG = K / GS;
    constexpr int NGS = (NG + SPLITS - 1) / SPLITS;
    constexpr int SGN = BN / SN;
    constexpr int CAP = SM * SN / 32;
    constexpr int NRN = CAP / (4 * NM);
    const int M = xs_shape[0];
    const int MV = x_scale_shape[0];
    const int split = int(threadgroup_position_in_grid.z);
    const int g0 = split * NGS;
    const int ng = min(NGS, NG - g0);
    device float* part = (device float*)out + size_t(split) * MV * N;
    const int sg = int(simdgroup_index_in_threadgroup);
    const int n0 =
        min(int(threadgroup_position_in_grid.x) * BN, N - BN) + (sg % SGN) * SN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM + (sg / SGN) * SM;
    device const half* ws = (device const half*)w_s;
    device const half* wb = (device const half*)w_b;

    using XT = tensor<device int8_t, dextents<int, 2>, tensor_inline>;
    using WT = tensor<device int4b_format, dextents<int, 2>, tensor_inline>;
    XT X((device int8_t*)xq, dextents<int, 2>(K, M), array<int, 2>{1, K});
    WT W((device uchar*)w_packed, dextents<int, 2>(K, N), array<int, 2>{1, K});
    constexpr auto desc = matmul2d_descriptor(
        SM, SN, GS, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> mm;
    auto a0 = X.slice<GS, SM>(0, m0);
    auto b0 = W.slice<GS, SN>(0, n0);
    auto acc = mm.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), int>();
    auto facc = mm.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), float>();

    int rn[NRN];
    int rm[NM];
    bool fast = facc.get_capacity() == CAP;
    #pragma clang loop unroll(full)
    for (int r = 0; r < NRN; ++r) {
        rn[r] = n0 + int(facc.get_multidimensional_index(r * 8)[0]);
    }
    #pragma clang loop unroll(full)
    for (int t = 0; t < NM; ++t) {
        const int i = (t / 2) * NRN * 8 + (t % 2) * 4;
        rm[t] = m0 + int(facc.get_multidimensional_index(i)[1]);
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < CAP; ++i) {
        const auto c = facc.get_multidimensional_index(i);
        const int r = (i / 8) % NRN, t = (i / (8 * NRN)) * 2 + (i / 4) % 2;
        fast = fast && facc.is_valid_element(i) && n0 + int(c[0]) == rn[r] + i % 4 &&
            m0 + int(c[1]) == rm[t];
        facc[i] = 0.0f;
    }

    if constexpr (BIAS == 1) {
        // Cooperative tensors of different ops cannot be mixed, so the f32 x f16
        // result reaches facc through this SIMD group's slice of a fp32 tile.
        threadgroup float bias_tile[BM * BN];
        tensor<device float, dextents<int, 2>, tensor_inline> XS(
            (device float*)xs + g0, dextents<int, 2>(ng, M), array<int, 2>{1, NG});
        tensor<device half, dextents<int, 2>, tensor_inline> WB(
            (device half*)w_b + size_t(g0) * N, dextents<int, 2>(N, ng), array<int, 2>{1, N});
        constexpr auto bdesc = matmul2d_descriptor(
            SM, SN, static_cast<int>(dynamic_extent), false, false, false,
            matmul2d_descriptor::mode::multiply);
        matmul2d<bdesc, execution_simdgroup> mb;
        auto xa = XS.slice(0, m0);
        auto wbt = WB.slice(n0, 0);
        auto bacc = mb.template get_destination_cooperative_tensor<
            decltype(xa), decltype(wbt), float>();
        mb.run(xa, wbt, bacc);
        tensor<threadgroup float, extents<int, SN, SM>, tensor_inline> bt(
            bias_tile + sg * SM * SN, extents<int, SN, SM>{});
        bacc.store(bt);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        facc.load(bt);
    }

    if (fast) {
        half4 sv[NRN];
        #pragma clang loop unroll(full)
        for (int r = 0; r < NRN; ++r) {
            sv[r] = *(device const half4*)(ws + size_t(g0) * N + rn[r]);
        }
        for (int g = g0; g < g0 + ng; ++g) {
            half4 sc[NRN];
            const int gn = min(g + 1, NG - 1);
            #pragma clang loop unroll(full)
            for (int r = 0; r < NRN; ++r) {
                sc[r] = sv[r];
                sv[r] = *(device const half4*)(ws + size_t(gn) * N + rn[r]);
            }
            auto a = X.slice<GS, SM>(g * GS, m0);
            auto b = W.slice<GS, SN>(g * GS, n0);
            mm.run(a, b, acc);
            group_rescale<CAP, NRN>(acc, facc, sc);
        }
        if (BIAS == 3) {
            for (int g = g0; g < g0 + ng; ++g) {
                half4 bv[NRN];
                float xv[NM];
                #pragma clang loop unroll(full)
                for (int r = 0; r < NRN; ++r) {
                    bv[r] = *(device const half4*)(wb + size_t(g) * N + rn[r]);
                }
                #pragma clang loop unroll(full)
                for (int t = 0; t < NM; ++t) {
                    xv[t] = xs[size_t(rm[t]) * NG + g];
                }
                #pragma clang loop unroll(full)
                for (ushort i = 0; i < CAP; ++i) {
                    const int r = (i / 8) % NRN, t = (i / (8 * NRN)) * 2 + (i / 4) % 2;
                    facc[i] = fma(xv[t], float(bv[r][i % 4]), facc[i]);
                }
            }
        }
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < CAP / 4; ++j) {
            const int r = (j / 2) % NRN, t = (j / (2 * NRN)) * 2 + j % 2;
            if constexpr (SPLITS > 1) {
                if (rm[t] < MV) {
                    *(device float4*)(part + size_t(rm[t]) * N + rn[r]) = float4(
                        facc[4 * j], facc[4 * j + 1], facc[4 * j + 2], facc[4 * j + 3]);
                }
            } else {
                const float xsc = x_scale[rm[t]];
                *(device vec<T, 4>*)(out + size_t(rm[t]) * N + rn[r]) = vec<T, 4>(
                    T(xsc * facc[4 * j]), T(xsc * facc[4 * j + 1]),
                    T(xsc * facc[4 * j + 2]), T(xsc * facc[4 * j + 3]));
            }
        }
    } else {
        for (int g = g0; g < g0 + ng; ++g) {
            auto a = X.slice<GS, SM>(g * GS, m0);
            auto b = W.slice<GS, SN>(g * GS, n0);
            mm.run(a, b, acc);
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < CAP; ++i) {
                if (facc.is_valid_element(i)) {
                    const auto c = facc.get_multidimensional_index(i);
                    const int n = n0 + int(c[0]);
                    const int m = m0 + int(c[1]);
                    facc[i] += float(acc[i]) * float(ws[size_t(g) * N + n]);
                    if (BIAS == 3) {
                        facc[i] +=
                            xs[size_t(m) * NG + g] * float(wb[size_t(g) * N + n]);
                    }
                }
            }
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < CAP; ++i) {
            if (facc.is_valid_element(i)) {
                const auto c = facc.get_multidimensional_index(i);
                const int n = n0 + int(c[0]);
                const int m = m0 + int(c[1]);
                if constexpr (SPLITS > 1) {
                    if (m < MV) {
                        part[size_t(m) * N + n] = facc[i];
                    }
                } else {
                    out[size_t(m) * N + n] = T(x_scale[m] * facc[i]);
                }
            }
        }
    }
"""

# Split-K reduce for _GROUP_INT8_SG_SOURCE (template args T, N, SPLITS; N % 4
# == 0): out[m, n] = x_scale[m] * sum_s partial[s, m, n], four columns per
# thread.
_GROUP_SPLIT_REDUCE_SOURCE = """
    const int M = x_scale_shape[0];
    const size_t q = thread_position_in_grid.x;
    if (q >= size_t(M) * (N / 4)) {
        return;
    }
    const int m = int(q / (N / 4));
    const size_t base = size_t(m) * N + (q % (N / 4)) * 4;
    float4 acc = 0.0f;
    for (int s = 0; s < SPLITS; ++s) {
        acc += *(device const float4*)(partial + size_t(s) * M * N + base);
    }
    *(device vec<T, 4>*)(out + base) = vec<T, 4>(acc * x_scale[m]);
"""

# ---- End of verbatim sources ----

_MAX_M = 2**31 - 1

# The study's pinned M5 Max configs (analysis/pinned_q/<shape>.json, family
# int8_int4_g256_sg) per Muse Glimmer (N, K): (M, config), smallest M first.
# A config serves M up to its M (above the previous one); the last serves every
# larger M.
_PINNED: dict[tuple[int, int], list[tuple[int, str]]] = {
    (8704, 6656): [  # qkv
        (1, "sg1 16x128/16x64 k13"),
        (2, "sg1 16x64/16x64 k4"),
        (4, "sg1 16x64/16x64 k4"),
        (8, "sg1 16x64/16x64 k4"),
        (16, "sg1 32x64/32x32 k2"),
        (64, "sg1 32x64/32x32"),
        (256, "sg1 64x128/32x32"),
    ],
    (6656, 4096): [  # o
        (1, "sg1 16x64/16x64 k16"),
        (2, "sg1 16x128/16x64 k4"),
        (4, "sg1 16x64/16x64 k4"),
        (8, "sg1 16x64/16x64 k4"),
        (16, "sg1 16x128/16x64 k4"),
        (64, "sg1 32x128/32x32"),
        (256, "sg1 32x64/32x32"),
    ],
    (39936, 6656): [  # gate_up
        (1, "sg1 16x128/16x64 k13"),
        (2, "sg1 16x128/16x64 k13"),
        (4, "sg1 16x64/16x64"),
        (8, "sg1 16x128/16x64"),
        (16, "sg1 16x128/16x64"),
        (64, "sg3 32x128/32x32"),
        (256, "sg3 32x128/32x32"),
    ],
    (6656, 19968): [  # down
        (1, "sg1 16x64/16x64 k16"),
        (2, "sg1 16x128/16x64 k16"),
        (4, "sg1 16x128/16x64 k16"),
        (8, "sg1 16x128/16x64 k4"),
        (16, "sg1 16x64/16x64 k4"),
        (64, "sg1 32x64/32x32 k4"),
        (256, "sg1 32x128/32x32 k4"),
    ],
    (202048, 6656): [  # lm_head
        (1, "sg1 16x64/16x64 k13"),
        (2, "sg1 16x64/16x64 k4"),
        (4, "sg1 16x256/16x64 k2"),
        (8, "sg1 16x128/16x64"),
        (16, "sg1 16x256/16x64"),
        (64, "sg3 64x128/32x32"),
        (256, "sg3 32x64/32x32"),
    ],
}
# Other shapes (and tests): the study's most common small-M and large-M picks.
_DEFAULT_PLANS: list[tuple[int, str]] = [
    (16, "sg1 16x64/16x64 k4"),
    (256, "sg1 32x64/32x32"),
]


def parse_config(config: str) -> tuple[int, int, int, int, int, int]:
    """``"sg<bias> BMxBN/SMxSN[ k<splits>]"`` -> (BM, BN, SM, SN, bias, splits)."""
    parts = config.split()
    bias = int(parts[0][2:])
    tile, sub = parts[1].split("/")
    bm, bn = map(int, tile.split("x"))
    sm, sn = map(int, sub.split("x"))
    splits = int(parts[2][1:]) if len(parts) > 2 else 1
    return bm, bn, sm, sn, bias, splits


def plans(N: int, K: int) -> list[tuple[int, str]]:
    """(max_m, config) per M tier, smallest first; the last covers every M."""
    pinned = _PINNED.get((N, K), _DEFAULT_PLANS)
    return [(m, c) for m, c in pinned[:-1]] + [(_MAX_M, pinned[-1][1])]


def effective_splits(n_groups: int, splits: int) -> int:
    """The study's split count: ceil(NG / ceil(NG / splits)), so no split is
    empty."""
    return math.ceil(n_groups / math.ceil(n_groups / splits))


def _lit(*values: int) -> list[IntOrVid]:
    return [IntOrVid.from_literal(v) for v in values]


@REGISTRY.register(target=[torch.ops.mlx.int8_sym_int4_gs256_gemm.default])
def _int8_sym_int4_gs256_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    op_name = "mlx::int8_sym_int4_gs256_gemm"
    x_slot, w_packed_slot, w_s_slot, w_b_slot = P.args(n)
    x_node = n.args[0]
    x_meta = x_node.meta["val"]
    N, KP = n.args[1].meta["val"].shape
    K = x_meta.shape[-1]
    w_s_meta, w_b_meta = n.args[2].meta["val"], n.args[3].meta["val"]
    if not all(isinstance(v, int) for v in (N, KP, K)):
        raise NotImplementedError(f"{op_name}: K and N must be statically known")
    if K != 2 * KP or K % GROUP_SIZE:
        raise ValueError(
            f"{op_name}: need K = 2 * w_packed width and K % {GROUP_SIZE} == 0; "
            f"got K={K}, width {KP}"
        )
    ng = K // GROUP_SIZE
    for meta, name in ((w_s_meta, "w_s"), (w_b_meta, "w_b")):
        if tuple(meta.shape) != (ng, N) or meta.dtype != torch.float16:
            raise ValueError(
                f"{op_name}: {name} must be fp16 [{ng}, {N}]; got {meta.dtype} "
                f"{tuple(meta.shape)}"
            )
    tiers = plans(N, K)
    max_bn = max(parse_config(c)[1] for _, c in tiers)
    if N % 4 or N < max_bn:
        raise NotImplementedError(
            f"{op_name}: needs N % 4 == 0 and N >= {max_bn}; got N={N}"
        )

    dtype_int = torch_dtype_to_scalar_type(x_meta.dtype)
    int8 = torch_dtype_to_scalar_type(torch.int8)
    fp32 = torch_dtype_to_scalar_type(torch.float32)
    out = P.make_or_get_slot(n)
    leading = emit_shape(P, x_node, x_slot, end_dim=-1)
    out_shape_flat = leading + [IntOrVid.from_literal(N)]
    m_iov = emit_product(P, leading)
    _, xq_slot = P.make_tmp_slot()
    _, x_scale_slot = P.make_tmp_slot()
    _, xs_slot = P.make_tmp_slot()
    _, gemm_slot = P.make_tmp_slot()
    _, rows_out_slot = P.make_tmp_slot()
    n_lit = IntOrVid.from_literal(N)

    def emit_kernel(
        name: str,
        source: str,
        header: str,
        inputs: dict[str, Slot],
        outputs: dict[str, tuple[Slot, int, list[IntOrVid]]],
        grid: list[IntOrVid],
        threads: int,
        template_args: dict[str, int],
    ) -> None:
        names = list(template_args)
        P.emit(
            MetalKernelNode(
                name=name,
                source=source,
                header=header,
                inputs=[P.slot_to_tid(s) for s in inputs.values()],
                outputs=[P.slot_to_tid(s) for s, _, _ in outputs.values()],
                grid=grid,
                threadgroup=_lit(threads, 1, 1),
                input_names=list(inputs),
                output_names=list(outputs),
                output_shapes_flat=[
                    d for _, _, shape in outputs.values() for d in shape
                ],
                output_shape_lengths=[len(shape) for _, _, shape in outputs.values()],
                output_dtypes=[dt for _, dt, _ in outputs.values()],
                template_arg_names=names,
                template_arg_kinds=[2 if a == "T" else 0 for a in names],
                template_arg_values=list(template_args.values()),
            )
        )

    def emit_plan(config: str) -> None:
        bm, bn, sm, sn, bias, k = parse_config(config)
        splits = effective_splits(ng, k)
        tiles_m = emit_ceil_div(P, m_iov, bm)
        rows = emit_product(P, [tiles_m, IntOrVid.from_literal(bm)])
        emit_kernel(
            "int8_sym_int4_gs256_quant",
            _GROUP_QUANT_VEC_SOURCE,
            _HEADER,
            {"x": x_slot},
            {
                "xq": (xq_slot, int8, [rows, IntOrVid.from_literal(K)]),
                "x_scale": (x_scale_slot, fp32, [m_iov if splits > 1 else rows]),
                "xs": (xs_slot, fp32, [rows, IntOrVid.from_literal(ng)]),
            },
            [IntOrVid.from_literal(_QUANT_THREADS), rows, IntOrVid.from_literal(1)],
            _QUANT_THREADS,
            {"T": dtype_int, "K": K, "TPR": _QUANT_THREADS, "GS": GROUP_SIZE},
        )
        simdgroups = (bm // sm) * (bn // sn)
        gemm_inputs = {
            "xq": xq_slot,
            "x_scale": x_scale_slot,
            "xs": xs_slot,
            "w_packed": w_packed_slot,
            "w_s": w_s_slot,
            "w_b": w_b_slot,
        }
        gemm_targs = {
            "T": dtype_int,
            "N": N,
            "K": K,
            "BM": bm,
            "BN": bn,
            "SM": sm,
            "SN": sn,
            "GS": GROUP_SIZE,
            "NM": sg_rows_per_thread(sm, sn),
            "BIAS": bias,
            "SPLITS": splits,
        }
        grid = [
            IntOrVid.from_literal((N + bn - 1) // bn * 32 * simdgroups),
            tiles_m,
            IntOrVid.from_literal(splits),
        ]
        if splits > 1:
            emit_kernel(
                "int8_sym_int4_gs256_gemm",
                _GROUP_INT8_SG_SOURCE,
                _GROUP_SG_HEADER,
                gemm_inputs,
                {
                    "out": (
                        gemm_slot,
                        fp32,
                        [IntOrVid.from_literal(splits), m_iov, n_lit],
                    )
                },
                grid,
                32 * simdgroups,
                gemm_targs,
            )
            emit_kernel(
                "int8_sym_int4_gs256_split_reduce",
                _GROUP_SPLIT_REDUCE_SOURCE,
                _HEADER,
                {"partial": gemm_slot, "x_scale": x_scale_slot},
                {"out": (out, dtype_int, out_shape_flat)},
                [
                    emit_product(
                        P,
                        [
                            emit_ceil_div(
                                P,
                                emit_product(P, [m_iov, IntOrVid.from_literal(N // 4)]),
                                256,
                            ),
                            IntOrVid.from_literal(256),
                        ],
                    ),
                    IntOrVid.from_literal(1),
                    IntOrVid.from_literal(1),
                ],
                256,
                {"T": dtype_int, "N": N, "SPLITS": splits},
            )
            return
        # Unsplit, the kernel writes every padded row; keep the first M.
        emit_kernel(
            "int8_sym_int4_gs256_gemm",
            _GROUP_INT8_SG_SOURCE,
            _GROUP_SG_HEADER,
            gemm_inputs,
            {"out": (gemm_slot, dtype_int, [rows, n_lit])},
            grid,
            32 * simdgroups,
            gemm_targs,
        )
        P.emit(
            SliceNode(
                x=P.slot_to_tid(gemm_slot),
                out=P.slot_to_tid(rows_out_slot),
                axis=IntOrVid.from_literal(0),
                start=IntOrVid.from_literal(0),
                stop=m_iov,
            )
        )
        P.emit(
            ReshapeNode(
                x=P.slot_to_tid(rows_out_slot),
                out=P.slot_to_tid(out),
                shape=out_shape_flat,
            )
        )

    # (M - 1) // max_m == 0 exactly when M <= max_m, so each IfNode's else
    # branch takes the first tier that covers M.
    m_minus_1 = emit_sub_int(P, m_iov, IntOrVid.from_literal(1))

    def emit_tiers(rest: list[tuple[int, str]]) -> None:
        (max_m, config), rest = rest[0], rest[1:]
        if not rest:
            emit_plan(config)
            return
        emit_if_else(
            P,
            emit_floordiv(P, m_minus_1, IntOrVid.from_literal(max_m)),
            emit_then=lambda: emit_tiers(rest),
            emit_else=lambda: emit_plan(config),
        )

    emit_tiers(tiers)
    return out
