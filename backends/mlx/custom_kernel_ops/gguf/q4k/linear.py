#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF **Q4_K** linear implementation.

Same structure as :mod:`..q6k.linear`: mat-vec (M==1), mat-mat (M>1), IfNode (dynamic M).
Kernels ported from llama.cpp ``kernel_mul_mv_q4_K_f32_impl`` and ``kernel_mul_mm`` (Q4_K).
"""

from __future__ import annotations

from typing import Optional

from executorch.backends.mlx.builder.op_helpers import (
    emit_ceil_div,
    emit_if_else,
    emit_product,
    emit_shape,
    emit_sub_int,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.custom_kernel_ops.gguf.q4k.common import (
    _Q4K_HEADER,
    Q4K_BLOCK_BYTES,
    QK_K,
)
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    IntOrVid,
    MetalKernelNode,
    MultiplyIntNode,
)
from torch.fx.node import Node


# ---------------------------------------------------------------------------
# Metal kernel sources
# ---------------------------------------------------------------------------


# Decode mat-vec kernel, ported from llama.cpp kernel_mul_mv_q4_K_f32_impl.
# Threadgroup = (32 * NSG, 1, 1): NSG simdgroups, each computing N_R0 output
# rows for one activation row (grid.y). Accumulate in float, reduce via simd_sum.
def _q4k_matvec_source(has_bias: bool) -> str:
    write = "out[(uint)m * N + r] = (OutT)(tot"
    write += " + (float)bias[r]);" if has_bias else ");"
    return f"""
    constexpr short N_R0 = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort tiisg = thread_index_in_simdgroup;
    const ushort sgitg = simdgroup_index_in_threadgroup;
    const uint   m     = thread_position_in_grid.y;
    const uint   tgx   = thread_position_in_grid.x / (32u * NSG);
    const int    nb    = K / QK_K;
    const int    first_row = (int)(tgx * NSG + sgitg) * N_R0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    device const block_q4_K * xrows = (device const block_q4_K *) weight;
    device const InT * yy = x + (uint)m * (uint)K;
    device const InT * y4 = yy + ix * QK_K + 64 * iq + 8 * ir;

    float sumf[N_R0];
    for (short row = 0; row < N_R0; ++row) {{ sumf[row] = 0.f; }}

    float yl[16];
    float yh[16];
    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {{
        float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
        for (short i = 0; i < 8; ++i) {{
            yl[i+0] = (float) y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = (float) y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = (float) y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = (float) y4[i+160]; sumy[3] += yh[i+8];
        }}

        for (short row = 0; row < N_R0; ++row) {{
            const int r = first_row + row;
            device const block_q4_K * blk = xrows + (uint)r * nb + ib;
            device const uint16_t * sc = (device const uint16_t *)blk->scales + iq;
            device const uint16_t * q1 = (device const uint16_t *)blk->qs + 16 * iq + 4 * ir;
            device const uint16_t * q2 = q1 + 32;
            const float d  = (float) blk->d;
            const float dm = (float) blk->dmin;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            float4 acc1 = {{0.f, 0.f, 0.f, 0.f}};
            float4 acc2 = {{0.f, 0.f, 0.f, 0.f}};
            for (short i = 0; i < 4; ++i) {{
                acc1[0] += yl[2*i + 0] * (float)(q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (float)(q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (float)(q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (float)(q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (float)(q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (float)(q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (float)(q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (float)(q2[i] & 0xF000);
            }}

            sumf[row] += d * ((acc1[0] + acc1[1] / 256.f) * (float)sc8[0] +
                              (acc1[2] + acc1[3] / 256.f) * (float)sc8[1] / 16.f +
                              (acc2[0] + acc2[1] / 256.f) * (float)sc8[4] +
                              (acc2[2] + acc2[3] / 256.f) * (float)sc8[5] / 16.f) -
                         dm * (sumy[0] * (float)sc8[2] + sumy[1] * (float)sc8[3] +
                               sumy[2] * (float)sc8[6] + sumy[3] * (float)sc8[7]);
        }}

        y4 += 4 * QK_K;
    }}

    for (short row = 0; row < N_R0; ++row) {{
        const int r = first_row + row;
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r < N) {{
            {write}
        }}
    }}
"""


# Prefill mat-mat kernel, ported from llama.cpp kernel_mul_mm (Q4_K variant).
# 64x32 output tiles, 4 simdgroups / 128 threads per threadgroup.
# Uses vectorized dequantize_q4_K_16 to decode 16 weight values per thread
# into threadgroup memory, then runs simdgroup_multiply_accumulate on 8x8
# tiles. NL=16 for Q4_K (QK_K / 16 = 16 dequant steps per super-block).
def _q4k_matmul_source(has_bias: bool) -> str:
    bias_add = "+ (float) bias[r0 + i]" if has_bias else ""
    return f"""
    constexpr short NR0 = 64;   // weight/output rows per tile (N dim)
    constexpr short NR1 = 32;   // activation rows per tile (M dim)
    constexpr short NK  = 32;   // K-chunk per iteration
    constexpr short NL  = 16;   // Q4_K: QK_K / 16
    constexpr short NL0 = NK / 16;  // = 2 — dequant iterations per thread for weight
    constexpr short NL1 = NK / 8;   // = 4 — load iterations per thread for activation

    threadgroup half sa[4096];  // NR0 * NK weight tile; reused as NR1*NR0 float output staging (8 KB)
    threadgroup half sb[1024];  // NR1 * NK activation tile (strided by 64): 16 ib slots * 64

    const ushort tid   = thread_index_in_threadgroup;   // 0..127
    const ushort sgitg = simdgroup_index_in_threadgroup; // 0..3

    const uint r0 = thread_position_in_grid.y * NR0;  // first weight row
    const uint r1 = (thread_position_in_grid.x / 128u) * NR1;  // first activation row

    // M (number of activation rows) read at runtime.
    int M = 1;
    for (uint d = 0; d + 1 < x_ndim; ++d) {{ M *= (int) x_shape[d]; }}

    const int nb = K / QK_K;

    // Clamp tile edges.
    const short nr0 = (N - (int)r0 < NR0) ? (N - (int)r0) : NR0;
    const short nr1 = (M - (int)r1 < NR1) ? (M - (int)r1) : NR1;

    // Thread → element mapping for cooperative loads.
    const short lr0 = ((short)(tid / NL0) < nr0) ? (short)(tid / NL0) : (nr0 - 1);  // 0..63
    const short lr1 = ((short)(tid / NL1) < nr1) ? (short)(tid / NL1) : (nr1 - 1);  // 0..31

    short il0 = tid % NL0;
    short il  = il0;  // current dequant sub-block index within Q4_K block
    
    const short offset1 = il0 / NL;  // always 0 (il0 < NL0=2, NL=16)

    // Pointer to weight block for this thread's assigned row.
    device const block_q4_K * wblk = (device const block_q4_K *) weight
        + (uint)(r0 + lr0) * nb + offset1;

    // Pointer to activation row for this thread.
    const short iy = 8 * (tid % NL1);
    device const InT * yp = x + (uint)(r1 + lr1) * (uint)K + iy;

    // Accumulator: 8 simdgroup 8x8 matrices (4 sgitg configs x 2 sub-tiles).
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {{
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }}

    for (int loop_k = 0; loop_k < K; loop_k += NK) {{
        // --- Cooperative load: dequantized weight tile (NR0 x NK) into sa ---
        half4x4 temp_a;
        dequantize_q4_K_16(wblk, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; ++i) {{
            const short sx = 2 * il0 + i / 8;
            const short sy = (tid / NL0) / 8;
            const short lx = (tid / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }}

        // --- Cooperative load: activation tile (NR1 x NK) into sb ---
        const short sx_b = tid % NL1;
        const short sy_b = (tid / NL1) / 8;
        const short ly_b = (tid / NL1) % 8;
        const short ib_b = 4 * sx_b + sy_b;

        for (short i = 0; i < 8; ++i) {{
            *(sb + 64 * ib_b + 8 * ly_b + i) = (half) *(yp + i);
        }}

        // Advance weight pointer through Q4_K sub-blocks.
        il = (il + 2 < NL) ? il + 2 : il % 2;
        wblk = (il < 2) ? wblk + (2 + NL - 1) / NL : wblk;

        yp += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Simdgroup matmul on loaded tiles ---
        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < NK / 8; ++ik) {{
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {{
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }}
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {{
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }}
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {{
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }}
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }}
    }}

    // --- Write results: stage the output tile in threadgroup memory, then
    // drain it to device. Staging is required for the float->OutT cast and the
    // optional bias add.
    // Barrier needed: sa was used for weight tiles during the K-loop and is now
    // reused as float staging for the output. Without this barrier, a fast
    // simdgroup could start writing mc[] into sa while a slower one is still
    // reading the last weight tile via simdgroup_load(ma[]).
    // (Mirrors the barrier in llama.cpp kernel_mul_mm's bounds-checked write path.)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {{
        threadgroup float * temp_str = ((threadgroup float *) sa)
            + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; ++i) {{
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                            NR0, ulong2(0, 0), false);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Drain all NR1*NR0 tile elements across the threadgroup's 128 threads.
        // NR0/NR1 are compile-time constants, so idx / NR0 and idx % NR0 fold
        // to a shift/mask.
        for (int idx = tid; idx < NR1 * NR0; idx += 128) {{
            const int j = idx / NR0;
            const int i = idx % NR0;
            if (j < nr1 && i < nr0) {{
                const float v = ((threadgroup float *) sa)[j * NR0 + i];
                out[(uint)(r1 + j) * (uint)N + (r0 + i)] = (OutT)(v {bias_add});
            }}
        }}
    }}
"""


# Number of simdgroups per threadgroup for the mat-vec kernel.
# Matches llama.cpp N_SG_Q4_K=2; nsg=4 launched half as many (fatter) threadgroups,
# hurting occupancy / wave-quantization tail on the bandwidth-bound decode matvec.
_Q4K_MV_NSG = 2
# Tile sizes for the mat-mat kernel (from llama.cpp kernel_mul_mm).
_Q4K_MM_NR0 = 64  # weight/output rows (N dim) per threadgroup
_Q4K_MM_NR1 = 32  # activation rows (M dim) per threadgroup


def _emit_q4k_matvec(
    P: MLXProgramBuilder,
    x_node: Node,
    x_slot: Slot,
    weight_slot: Slot,
    bias_slot: Optional[Slot],
    N: int,
    K: int,
    out_dtype_int: int,
    out: Slot,
) -> None:
    in_dtype_int = torch_dtype_to_scalar_type(x_node.meta["val"].dtype)

    leading = emit_shape(P, x_node, x_slot, end_dim=-1)
    M_iov = emit_product(P, leading)
    out_shape_flat = leading + [IntOrVid.from_literal(N)]

    n_r0 = 2
    nsg = _Q4K_MV_NSG
    num_row_groups = (N + nsg * n_r0 - 1) // (nsg * n_r0)
    grid_x = num_row_groups * 32 * nsg

    has_bias = bias_slot is not None
    inputs = [P.slot_to_tid(x_slot), P.slot_to_tid(weight_slot)]
    input_names = ["x", "weight"]
    if has_bias:
        inputs.append(P.slot_to_tid(bias_slot))
        input_names.append("bias")

    P.emit(
        MetalKernelNode(
            name="gguf_q4k_matvec",
            source=_q4k_matvec_source(has_bias),
            header=_Q4K_HEADER,
            inputs=inputs,
            outputs=[P.slot_to_tid(out)],
            grid=[
                IntOrVid.from_literal(grid_x),
                M_iov,
                IntOrVid.from_literal(1),
            ],
            threadgroup=[
                IntOrVid.from_literal(32 * nsg),
                IntOrVid.from_literal(1),
                IntOrVid.from_literal(1),
            ],
            input_names=input_names,
            output_names=["out"],
            output_shapes_flat=out_shape_flat,
            output_shape_lengths=[len(out_shape_flat)],
            output_dtypes=[out_dtype_int],
            template_arg_names=["InT", "OutT", "N", "K", "NSG"],
            template_arg_kinds=[2, 2, 0, 0, 0],  # dtype, dtype, int, int, int
            template_arg_values=[in_dtype_int, out_dtype_int, N, K, nsg],
        )
    )


def _emit_q4k_matmul(
    P: MLXProgramBuilder,
    x_node: Node,
    x_slot: Slot,
    weight_slot: Slot,
    bias_slot: Optional[Slot],
    N: int,
    K: int,
    blocks_m_iov: IntOrVid,
    out_dtype_int: int,
    out: Slot,
) -> None:
    in_dtype_int = torch_dtype_to_scalar_type(x_node.meta["val"].dtype)

    leading = emit_shape(P, x_node, x_slot, end_dim=-1)
    out_shape_flat = leading + [IntOrVid.from_literal(N)]

    # grid.x = ceil(M / NR1) * 128 threads (activation tiles)
    # grid.y = ceil(N / NR0) (weight tiles)
    blocks_n = (N + _Q4K_MM_NR0 - 1) // _Q4K_MM_NR0

    has_bias = bias_slot is not None
    inputs = [P.slot_to_tid(x_slot), P.slot_to_tid(weight_slot)]
    input_names = ["x", "weight"]
    if has_bias:
        inputs.append(P.slot_to_tid(bias_slot))
        input_names.append("bias")

    # blocks_m_iov = ceil(M / NR1); multiply by 128 for grid.x
    _, grid_x_slot = P.make_tmp_value_slot()
    P.emit(
        MultiplyIntNode(
            a=blocks_m_iov,
            b=IntOrVid.from_literal(128),
            out=P.slot_to_vid(grid_x_slot),
        )
    )
    grid_x_iov = IntOrVid.from_vid(P.slot_to_vid(grid_x_slot))

    P.emit(
        MetalKernelNode(
            name="gguf_q4k_matmul",
            source=_q4k_matmul_source(has_bias),
            header=_Q4K_HEADER,
            inputs=inputs,
            outputs=[P.slot_to_tid(out)],
            grid=[
                grid_x_iov,
                IntOrVid.from_literal(blocks_n),
                IntOrVid.from_literal(1),
            ],
            threadgroup=[
                IntOrVid.from_literal(128),
                IntOrVid.from_literal(1),
                IntOrVid.from_literal(1),
            ],
            input_names=input_names,
            output_names=["out"],
            output_shapes_flat=out_shape_flat,
            output_shape_lengths=[len(out_shape_flat)],
            output_dtypes=[out_dtype_int],
            template_arg_names=["InT", "OutT", "N", "K"],
            template_arg_kinds=[2, 2, 0, 0],
            template_arg_values=[in_dtype_int, out_dtype_int, N, K],
        )
    )


def _emit_linear_fused(
    P: MLXProgramBuilder,
    head: Node,
    x_node: Node,
    weight_node: Node,
    bias_node: Optional[Node],
) -> Slot:
    """Lower a Q4_K ``dequantize_gguf`` -> ``linear`` pattern to fused kernels.

    ``weight_node`` is the raw GGUF blob (the dequantize op's weight input) and
    ``head`` is the ``aten.linear`` node that owns the output slot.
    """
    x_slot, weight_slot, bias_slot = P.slot_map([x_node, weight_node, bias_node])

    weight_meta = weight_node.meta["val"]
    if weight_meta.dim() != 2:
        raise NotImplementedError(
            f"gguf q4k linear: weight must be 2-D (N, row_bytes); got "
            f"shape {tuple(weight_meta.shape)}"
        )
    N = weight_meta.shape[0]
    row_bytes = weight_meta.shape[1]
    if not isinstance(N, int) or not isinstance(row_bytes, int):
        raise NotImplementedError(
            "gguf q4k linear: weight shape must be statically known"
        )
    if row_bytes % Q4K_BLOCK_BYTES != 0:
        raise ValueError(
            f"gguf q4k linear: weight row bytes {row_bytes} must be a multiple of "
            f"{Q4K_BLOCK_BYTES}"
        )
    K = (row_bytes // Q4K_BLOCK_BYTES) * QK_K

    out = P.make_or_get_slot(head)
    out_dtype_int = torch_dtype_to_scalar_type(head.meta["val"].dtype)
    tile = _Q4K_MM_NR1  # M-dimension tile (activation rows per threadgroup)

    m_iov = emit_product(P, emit_shape(P, x_node, x_slot, end_dim=-1))
    emit_if_else(
        P,
        emit_sub_int(P, m_iov, IntOrVid.from_literal(1)),
        emit_then=lambda: _emit_q4k_matmul(
            P,
            x_node,
            x_slot,
            weight_slot,
            bias_slot,
            N,
            K,
            emit_ceil_div(P, m_iov, tile),
            out_dtype_int,
            out,
        ),
        emit_else=lambda: _emit_q4k_matvec(
            P, x_node, x_slot, weight_slot, bias_slot, N, K, out_dtype_int, out
        ),
    )
    return out


def emit_linear(
    P: MLXProgramBuilder,
    head: Node,
    x_node: Node,
    weight_node: Node,
    bias_node: Optional[Node],
) -> Slot:
    """Dispatch to fused Metal kernels or the legacy MLX-native repack path."""
    from executorch.backends.mlx.custom_kernel_ops.gguf.q4k import emit_direct_gguf

    if emit_direct_gguf():
        return _emit_linear_fused(P, head, x_node, weight_node, bias_node)

    from executorch.backends.mlx.custom_kernel_ops.gguf.q4k.linear_mlx_native import (
        emit_linear as emit_linear_mlx_native,
    )

    return emit_linear_mlx_native(P, head, x_node, weight_node, bias_node)
