#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
``mlx::fp_int4_gemm``: fp16 / bf16 activation x 4-bit per-channel weight linear,
lowered to Metal Performance Primitives (Metal 4) kernels that multiply
``half`` (or ``bfloat``) by ``int4b_format`` with FP32 accumulation.

Weights are symmetric per output channel (no zero points, no groups), the same
format as ``mlx::int8_intx_gemm`` with ``bits=4``:

    W[n, k] = w_scale[n] * wq[n, k],  wq in [-8, 7]
    Y[m, n] = w_scale[n] * sum_k x[m, k] * wq[n, k]

``w_packed`` is ``uint8 [N, K // 2]`` from
:func:`executorch.backends.mlx.custom_kernel_ops.int8_intx_gemm.pack_int4`.

The kernel is picked at runtime from M (see :func:`full_k_plans`):
    * matvec for M <= 2: streams packed weights like MLX's qmv.
    * otherwise MPP over the full K straight from device memory, optionally
      split over K into FP32 partials plus a reduce kernel when that adds
      enough threadgroups.

Constraints for MLX lowering:
    * ``K`` must be statically known and a multiple of 64.
    * ``N`` must be statically known.
    * Activations must be fp16 or bf16.
    * Requires an MPP-capable GPU (Metal 4).

Usage::

    import executorch.backends.mlx.custom_kernel_ops.fp_int4_gemm  # noqa: F401

    y = torch.ops.mlx.fp_int4_gemm(x, w_packed, w_scale)
    # x:        (..., K)    fp16 / bf16
    # w_packed: (N, K // 2) uint8
    # w_scale:  (N,)        float
    # y:        (..., N)    x.dtype
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from executorch.backends.mlx.custom_kernel_ops.int8_intx_gemm import unpack_int4
from torch import nn, Tensor
from torch.fx.node import Node


class FpInt4Linear(nn.Module):
    """Bias-free linear computed by ``mlx::fp_int4_gemm``."""

    def __init__(self, w_packed: Tensor, w_scale: Tensor):
        super().__init__()
        self.in_features = 2 * w_packed.shape[1]
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
        return torch.ops.mlx.fp_int4_gemm(x, self.w_packed, self.w_scale)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


@torch.library.custom_op("mlx::fp_int4_gemm", mutates_args=())
def fp_int4_gemm(x: Tensor, w_packed: Tensor, w_scale: Tensor) -> Tensor:
    wq = unpack_int4(w_packed).float()
    y = (x.float() @ wq.T) * w_scale.float()
    return y.to(x.dtype)


@torch.library.register_fake("mlx::fp_int4_gemm")
def fp_int4_gemm_fake(x: Tensor, w_packed: Tensor, w_scale: Tensor) -> Tensor:
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
_MATVEC_SIMDGROUPS = 2
_MATVEC_ROWS_PER_SIMDGROUP = 4
_MATVEC_BLOCK = 512
_MAX_M = 2**31 - 1

# Muse Glimmer (N, K) -> plans as (max_m, bm, bn, splits), from the M4 Max fp16
# sweeps (test/bench_muse_glimmer_gemm.py --kernels fp_int4). bn == 0 is the
# matvec with bm rows per threadgroup; bn < 0 the qmv_wide-style kernel with bm
# rows per threadgroup and -bn lanes per output channel; M <= 4 runs
# ceil(M / bm) threadgroup rows. M = 1 / 2 / 4 use the fastest matvec or wide
# config across sweeps; the M <= 8 / <= 16 / larger tiles are the M = 8 / 16 /
# 256 winners.
_MG_PLANS: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {
    (8704, 6656): [  # qkv
        (1, 1, -16, 1),
        (2, 1, 0, 1),
        (4, 2, 0, 1),
        (8, 8, 128, 8),
        (16, 16, 64, 4),
        (_MAX_M, 128, 32, 4),
    ],
    (6656, 4096): [  # o
        (1, 1, -16, 1),
        (2, 2, -32, 1),
        (4, 2, -16, 1),
        (8, 8, 64, 8),
        (16, 16, 32, 4),
        (_MAX_M, 64, 32, 1),
    ],
    (39936, 6656): [  # gate_up
        (1, 1, -32, 1),
        (2, 2, 0, 1),
        (4, 2, -16, 1),
        (8, 8, 128, 4),
        (16, 16, 128, 8),
        (_MAX_M, 64, 64, 1),
    ],
    (6656, 19968): [  # down
        (1, 1, -32, 1),
        (4, 2, 0, 1),
        (8, 8, 64, 8),
        (16, 16, 64, 8),
        (_MAX_M, 64, 64, 4),
    ],
    (202048, 6656): [  # lm_head
        (1, 1, -8, 1),
        (2, 2, -16, 1),
        (4, 4, -32, 1),
        (8, 8, 128, 2),
        (16, 16, 128, 1),
        (_MAX_M, 64, 64, 1),
    ],
}


class FpInt4Plan(NamedTuple):
    """Kernel choice for ``M <= max_m``. ``bn == 0`` selects the matvec kernel
    with ``bm`` activation rows per threadgroup; ``bn < 0`` the qmv_wide-style
    kernel with ``bm`` rows per threadgroup and ``-bn`` lanes per output
    channel; otherwise a full-K MPP GEMM with a ``bm x bn`` tile, split over K
    when ``splits > 1``."""

    max_m: int
    bm: int
    bn: int
    splits: int = 1


def full_k_plans(N: int, K: int) -> list[FpInt4Plan]:
    """Kernel plans, smallest ``max_m`` first; the last covers every larger M.
    Muse Glimmer shapes use the swept ``_MG_PLANS``. Other shapes: matvec for
    M <= 4 (2 rows per threadgroup past M = 1), then padded MPP tiles with more
    splits when N gives few output tiles."""
    if (N, K) in _MG_PLANS and K % _MATVEC_BLOCK == 0:
        return [FpInt4Plan(*p) for p in _MG_PLANS[(N, K)]]
    plans = []
    if K % _MATVEC_BLOCK == 0:
        plans += [FpInt4Plan(1, 1, 0), FpInt4Plan(4, 2, 0)]
    small_bn, small_splits = (64, 8) if N <= 16384 else (128, 2)
    tiles = [(8, small_bn, small_splits), (16, small_bn, small_splits), (32, 128, 1)]
    for max_m, (bm, bn, splits) in zip((8, 16, _MAX_M), tiles):
        while K % (_K_ALIGN * splits):
            splits //= 2
        plans.append(FpInt4Plan(max_m, bm, bn, splits))
    return plans


_HEADER = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

# Template args T, N, K, BM, BN. MPP multiplies a T [K, M] tensor by an int4b
# [K, N] tensor (W transposed) over the whole K for each BM x BN output tile.
_FULL_K_SOURCE = """
    int M = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
        M *= x_shape[d];
    }
    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;

    tensor<device T, dextents<int, 2>, tensor_inline> X(
        (device T*)x, dextents<int, 2>(K, M));
    tensor<device int4b_format, dextents<int, 2>, tensor_inline> W(
        (device uchar*)w_packed, dextents<int, 2>(K, N), array<int, 2>{1, K});
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto mA = X.slice(0, m0);
    auto mB = W.slice(0, n0);
    auto acc = mm.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    mm.run(mA, mB, acc);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            const auto coord = acc.get_multidimensional_index(i);
            const int n = n0 + int(coord[0]);
            const int m = m0 + int(coord[1]);
            if (m < M && n < N) {
                out[size_t(m) * N + n] = T(acc[i] * float(w_scale[n]));
            }
        }
    }
"""

# Template args T, N, K, BM, BN, SPLITS (K % (64 * SPLITS) == 0): grid z picks
# one of SPLITS contiguous K ranges; each threadgroup writes FP32 partial sums
# to partial[s, m, n].
_SPLIT_K_SOURCE = """
    constexpr int KS = K / SPLITS;
    int M = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
        M *= x_shape[d];
    }
    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;
    const int s = int(threadgroup_position_in_grid.z);
    const int k0 = s * KS;

    tensor<device T, dextents<int, 2>, tensor_inline> X(
        (device T*)x + k0, dextents<int, 2>(KS, M), array<int, 2>{1, K});
    tensor<device int4b_format, dextents<int, 2>, tensor_inline> W(
        (device uchar*)w_packed + k0 / 2, dextents<int, 2>(KS, N),
        array<int, 2>{1, K});
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto mA = X.slice(0, m0);
    auto mB = W.slice(0, n0);
    auto acc = mm.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
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

# Template args T, N, SPLITS: one thread per output sums the partials and
# applies the channel scale.
_SPLIT_K_REDUCE_SOURCE = """
    const int M = partial_shape[1];
    const size_t idx = thread_position_in_grid.x;
    if (idx >= size_t(M) * N) {
        return;
    }
    float sum = 0.0f;
    for (int s = 0; s < SPLITS; ++s) {
        sum += partial[size_t(s) * M * N + idx];
    }
    out[idx] = T(sum * float(w_scale[idx % N]));
"""

# Template args T, N, K, MR (K % 512 == 0) for small M. Each SIMD group
# computes 4 output channels for up to MR activation rows; each lane covers 16
# K values per 512-wide step. XOR 0x88 makes nibbles unsigned (q = w + 8) so
# masked ushort words convert directly to float against activations pre-scaled
# by 1, 1/16, 1/256, 1/4096 (exact powers of two); subtracting 8 * sum(x)
# restores the signed dot product.
_MATVEC_SOURCE = """
    constexpr int SG = 2;
    constexpr int RPS = 4;
    constexpr int VPT = 16;
    constexpr int BLOCK = 32 * VPT;
    constexpr int KP = K / 2;

    int M = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
        M *= x_shape[d];
    }
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
    device const T* x_base = x + size_t(m0) * K + lane * VPT;
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
                device const vec<T, 4>* xv =
                    (device const vec<T, 4>*)(x_base + size_t(m) * K + k);
                float4 xs[4];
                #pragma clang loop unroll(full)
                for (int i = 0; i < 4; ++i) {
                    const float4 v = float4(xv[i]);
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
                const float total = simd_sum(acc[m][r] - 8.0f * xsum[m]);
                const int n = n0 + r;
                if (lane == 0 && n < N) {
                    out[size_t(m0 + m) * N + n] = T(total * float(w_scale[n]));
                }
            }
        }
    }
"""

# Template args T, N, K, NV, KL (K % (16 * KL) == 0) for small M, laid out like
# MLX's qmv_wide: KL lanes split K for one output channel (32 / KL channels per
# SIMD group, 2 SIMD groups per threadgroup) and each threadgroup streams NV
# activation rows, so each lane decodes one 16-value chunk of one channel per
# step and reuses it for every row. Activations are read from device memory
# per row instead of held in registers, which keeps register use low. The XOR /
# mask decode is undone on the weight side (fma with 1, 1/16, 1/256, 1/4096
# and -8, exact in fp32), so no sum(x) correction is needed.
_WIDE_SOURCE = """
    constexpr int SG = 2;
    constexpr int CPS = 32 / KL;
    constexpr int VPL = 16;
    constexpr int STEP = KL * VPL;
    constexpr int KP = K / 2;

    int M = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
        M *= x_shape[d];
    }
    const int lane = int(thread_index_in_simdgroup);
    const int kl = lane % KL;
    const int n = (int(threadgroup_position_in_grid.x) * SG
                   + int(simdgroup_index_in_threadgroup)) * CPS + lane / KL;
    const int v0 = int(threadgroup_position_in_grid.y) * NV;

    device const uchar* w = w_packed + size_t(min(n, N - 1)) * KP + kl * (VPL / 2);
    device const T* xv[NV];
    #pragma clang loop unroll(full)
    for (int v = 0; v < NV; ++v) {
        xv[v] = x + size_t(min(v0 + v, M - 1)) * K + kl * VPL;
    }
    const float4 pre = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

    float acc[NV];
    #pragma clang loop unroll(full)
    for (int v = 0; v < NV; ++v) {
        acc[v] = 0.0f;
    }

    for (int k = 0; k < K; k += STEP) {
        const ushort4 q = as_type<ushort4>(
            *(device const uint2*)(w + k / 2) ^ uint2(0x88888888u));
        float4 wf[4];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
            wf[i] = fma(
                float4(q[i] & 0x000f, q[i] & 0x00f0, q[i] & 0x0f00, q[i] & 0xf000),
                pre,
                float4(-8.0f));
        }
        #pragma clang loop unroll(full)
        for (int v = 0; v < NV; ++v) {
            device const vec<T, 4>* x4 = (device const vec<T, 4>*)(xv[v] + k);
            acc[v] += (dot(wf[0], float4(x4[0])) + dot(wf[1], float4(x4[1])))
                + (dot(wf[2], float4(x4[2])) + dot(wf[3], float4(x4[3])));
        }
    }

    // Reduce within this channel's KL lanes; simd_sum would mix channels.
    #pragma clang loop unroll(full)
    for (int v = 0; v < NV; ++v) {
        #pragma clang loop unroll(full)
        for (int off = KL / 2; off > 0; off /= 2) {
            acc[v] += simd_shuffle_down(acc[v], ushort(off));
        }
    }
    if (kl == 0 && n < N) {
        const float s = float(w_scale[n]);
        #pragma clang loop unroll(full)
        for (int v = 0; v < NV; ++v) {
            if (v0 + v < M) {
                out[size_t(v0 + v) * N + n] = T(acc[v] * s);
            }
        }
    }
"""


def _lit(*values: int) -> list[IntOrVid]:
    return [IntOrVid.from_literal(v) for v in values]


@REGISTRY.register(target=[torch.ops.mlx.fp_int4_gemm.default])
def _fp_int4_gemm_handler(P: MLXProgramBuilder, n: Node) -> Slot:  # noqa: C901
    x_slot, w_packed_slot, w_scale_slot = P.args(n)
    x_node = n.args[0]
    x_meta = x_node.meta["val"]
    N, KP = n.args[1].meta["val"].shape
    if not isinstance(N, int) or not isinstance(KP, int):
        raise NotImplementedError(
            "mlx::fp_int4_gemm: w_packed shape must be statically known"
        )
    K = 2 * KP
    if K % _K_ALIGN != 0:
        raise NotImplementedError(
            f"mlx::fp_int4_gemm: K must be a multiple of {_K_ALIGN}; got K={K}"
        )
    if x_meta.shape[-1] != K:
        raise ValueError(
            f"mlx::fp_int4_gemm: x last dim {x_meta.shape[-1]} does not match "
            f"K={K} from w_packed"
        )
    if x_meta.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError(
            f"mlx::fp_int4_gemm: activations must be fp16 or bf16; got {x_meta.dtype}"
        )

    dtype_int = torch_dtype_to_scalar_type(x_meta.dtype)
    out = P.make_or_get_slot(n)
    leading = emit_shape(P, x_node, x_slot, end_dim=-1)
    out_shape_flat = leading + [IntOrVid.from_literal(N)]
    m_iov = emit_product(P, leading)
    _, partial_slot = P.make_tmp_slot()
    n_lit = IntOrVid.from_literal(N)

    def emit_matvec(mr: int) -> None:
        rows_per_tg = _MATVEC_SIMDGROUPS * _MATVEC_ROWS_PER_SIMDGROUP
        P.emit(
            MetalKernelNode(
                name="fp_int4_gemm_matvec",
                source=_MATVEC_SOURCE,
                header=_HEADER,
                inputs=[
                    P.slot_to_tid(s) for s in (x_slot, w_packed_slot, w_scale_slot)
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
                input_names=["x", "w_packed", "w_scale"],
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
        grid = [
            IntOrVid.from_literal((N + bn - 1) // bn * _THREADS),
            emit_ceil_div(P, m_iov, bm),
            IntOrVid.from_literal(1),
        ]
        if splits == 1:
            P.emit(
                MetalKernelNode(
                    name="fp_int4_gemm_full_k",
                    source=_FULL_K_SOURCE,
                    header=_HEADER,
                    inputs=[
                        P.slot_to_tid(s) for s in (x_slot, w_packed_slot, w_scale_slot)
                    ],
                    outputs=[P.slot_to_tid(out)],
                    grid=grid,
                    threadgroup=_lit(_THREADS, 1, 1),
                    input_names=["x", "w_packed", "w_scale"],
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
                name="fp_int4_gemm_split_k",
                source=_SPLIT_K_SOURCE,
                header=_HEADER,
                inputs=[P.slot_to_tid(s) for s in (x_slot, w_packed_slot)],
                outputs=[P.slot_to_tid(partial_slot)],
                grid=grid[:2] + [IntOrVid.from_literal(splits)],
                threadgroup=_lit(_THREADS, 1, 1),
                input_names=["x", "w_packed"],
                output_names=["partial"],
                output_shapes_flat=[IntOrVid.from_literal(splits), m_iov, n_lit],
                output_shape_lengths=[3],
                output_dtypes=[torch_dtype_to_scalar_type(torch.float32)],
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
                name="fp_int4_gemm_split_k_reduce",
                source=_SPLIT_K_REDUCE_SOURCE,
                header=_HEADER,
                inputs=[P.slot_to_tid(s) for s in (partial_slot, w_scale_slot)],
                outputs=[P.slot_to_tid(out)],
                grid=[
                    reduce_threads,
                    IntOrVid.from_literal(1),
                    IntOrVid.from_literal(1),
                ],
                threadgroup=_lit(256, 1, 1),
                input_names=["partial", "w_scale"],
                output_names=["out"],
                output_shapes_flat=out_shape_flat,
                output_shape_lengths=[len(out_shape_flat)],
                output_dtypes=[dtype_int],
                template_arg_names=["T", "N", "SPLITS"],
                template_arg_kinds=[2, 0, 0],
                template_arg_values=[dtype_int, N, splits],
            )
        )

    def emit_wide(nv: int, kl: int) -> None:
        if K % (16 * kl):
            raise NotImplementedError(
                f"mlx::fp_int4_gemm: wide kernel needs K % {16 * kl} == 0; got K={K}"
            )
        channels_per_tg = _MATVEC_SIMDGROUPS * (32 // kl)
        P.emit(
            MetalKernelNode(
                name="fp_int4_gemm_wide",
                source=_WIDE_SOURCE,
                header=_HEADER,
                inputs=[
                    P.slot_to_tid(s) for s in (x_slot, w_packed_slot, w_scale_slot)
                ],
                outputs=[P.slot_to_tid(out)],
                grid=[
                    IntOrVid.from_literal(
                        (N + channels_per_tg - 1)
                        // channels_per_tg
                        * _MATVEC_SIMDGROUPS
                        * 32
                    ),
                    emit_ceil_div(P, m_iov, nv),
                    IntOrVid.from_literal(1),
                ],
                threadgroup=_lit(_MATVEC_SIMDGROUPS * 32, 1, 1),
                input_names=["x", "w_packed", "w_scale"],
                output_names=["out"],
                output_shapes_flat=out_shape_flat,
                output_shape_lengths=[len(out_shape_flat)],
                output_dtypes=[dtype_int],
                template_arg_names=["T", "N", "K", "NV", "KL"],
                template_arg_kinds=[2, 0, 0, 0, 0],
                template_arg_values=[dtype_int, N, K, nv, kl],
            )
        )

    def emit_plan(plan: FpInt4Plan) -> None:
        if plan.bn == 0:
            emit_matvec(plan.bm)
        elif plan.bn < 0:
            emit_wide(plan.bm, -plan.bn)
        else:
            emit_mpp(plan.bm, plan.bn, plan.splits)

    # (M - 1) // max_m == 0 exactly when M <= max_m, so each IfNode's else
    # branch takes the first plan that covers M.
    m_minus_1 = emit_sub_int(P, m_iov, IntOrVid.from_literal(1))

    def emit_plans(plans: list[FpInt4Plan]) -> None:
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
