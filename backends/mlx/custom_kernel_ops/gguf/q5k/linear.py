#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF **Q5_K** linear implementation.

Same structure as :mod:`..q6k.linear` / :mod:`..q4k.linear`: mat-vec (M==1),
mat-mat (M>1), IfNode (dynamic M).

Q5_K is the affine super-block of Q4_K (``d``/``dmin`` + 6-bit packed
scales/mins) plus a Q6_K-style 5th bit from ``qh``:

* the mat-vec is a faithful port of llama.cpp ``kernel_mul_mv_q5_K_f32_impl``
  (``N_R0 = 1``, ``NSG = 2``): inline kmask scale/min unpacking, with the nibble
  sum and the 5th-bit indicator sum accumulated separately and folded with
  ``d*scale`` / ``dmin*min``;
* the mat-mat reuses the shared tiled simdgroup kernel and only swaps in
  ``dequantize_q5_K_16`` for the weight tile decode.

Attribution
-----------
The Q5_K Metal kernels here are ported from llama.cpp
(``ggml/src/ggml-metal/ggml-metal.metal`` -- ``kernel_mul_mv_q5_K_f32_impl``,
``kernel_mul_mm``, ``dequantize_q5_K``), which is MIT-licensed
(Copyright (c) 2023-2024 The ggml authors).
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
from executorch.backends.mlx.custom_kernel_ops.gguf.q5k.common import (
    _Q5K_HEADER,
    Q5K_BLOCK_BYTES,
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


# Decode mat-vec kernel, a faithful port of llama.cpp kernel_mul_mv_q5_K_f32_impl
# (N_R0 = 1, NSG = 2). Threadgroup = (32 * NSG, 1, 1): NSG simdgroups, each
# computing N_R0 output rows for one activation row (grid.y). Accumulate in float,
# reduce via simd_sum. Thread map: tid = tiisg/4, ix = tiisg%4 (block stride),
# iq = tid/4, ir = tid%4. Scales/mins are unpacked inline via the kmask trick; the
# nibble sum (acc1) and the 5th-bit indicator sum (acc2, y where qh bit set) are
# accumulated separately and folded with d*scale / dmin*min.
def _q5k_matvec_source(has_bias: bool) -> str:
    write = "out[(uint)m * N + r] = (OutT)(tot"
    write += " + (float)bias[r]);" if has_bias else ");"
    return f"""
    constexpr short N_R0 = 1;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort tiisg = thread_index_in_simdgroup;
    const ushort sgitg = simdgroup_index_in_threadgroup;
    const uint   m     = thread_position_in_grid.y;
    const uint   tgx   = thread_position_in_grid.x / (32u * NSG);
    const int    nb    = K / QK_K;
    const int    first_row = (int)(tgx * NSG + sgitg) * N_R0;

    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;   // block stride (i = ix, ix+4, ...)
    const short iq  = tid / 4;     // 0..1
    const short ir  = tid % 4;     // 0..3
    const short l0  = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uint8_t hm1 = 1u << (2 * iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    device const block_q5_K * xrows = (device const block_q5_K *) weight;
    device const InT * yy = x + (uint)m * (uint)K;
    device const InT * y1 = yy + ix * QK_K + y_offset;

    float sumf[N_R0];
    for (short row = 0; row < N_R0; ++row) {{ sumf[row] = 0.f; }}

    float yl[16];
    float yh[16];
    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *) sc16;

    for (int i = ix; i < nb; i += 4) {{
        device const InT * y2 = y1 + 128;
        float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
        for (short l = 0; l < 8; ++l) {{
            yl[l + 0] = (float) y1[l +  0]; sumy[0] += yl[l + 0];
            yl[l + 8] = (float) y1[l + 32]; sumy[1] += yl[l + 8];
            yh[l + 0] = (float) y2[l +  0]; sumy[2] += yh[l + 0];
            yh[l + 8] = (float) y2[l + 32]; sumy[3] += yh[l + 8];
        }}

        for (short row = 0; row < N_R0; ++row) {{
            const int r = first_row + row;
            if (r >= N) {{ break; }}
            device const block_q5_K * blk = xrows + (uint)r * nb + i;
            device const uint8_t  * q1 = blk->qs + q_offset;
            device const uint8_t  * q2 = q1 + 64;
            device const uint8_t  * qh = blk->qh + l0;
            device const uint16_t * a  = (device const uint16_t *) blk->scales + iq;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {{0.f, 0.f, 0.f, 0.f}};
            float4 acc2 = {{0.f, 0.f, 0.f, 0.f}};
            FOR_UNROLL (short l = 0; l < 8; ++l) {{
                const uint8_t h = qh[l];
                acc1[0] += yl[l + 0] * (float)(q1[l] & 0x0F);
                acc1[1] += yl[l + 8] * (float)(q1[l] & 0xF0);
                acc1[2] += yh[l + 0] * (float)(q2[l] & 0x0F);
                acc1[3] += yh[l + 8] * (float)(q2[l] & 0xF0);
                acc2[0] += (h & hm1) ? yl[l + 0] : 0.f;
                acc2[1] += (h & hm2) ? yl[l + 8] : 0.f;
                acc2[2] += (h & hm3) ? yh[l + 0] : 0.f;
                acc2[3] += (h & hm4) ? yh[l + 8] : 0.f;
            }}

            sumf[row] += (float) blk->d * (
                             (float) sc8[0] * (acc1[0]        + 16.f * acc2[0]) +
                             (float) sc8[1] * (acc1[1] / 16.f + 16.f * acc2[1]) +
                             (float) sc8[4] * (acc1[2]        + 16.f * acc2[2]) +
                             (float) sc8[5] * (acc1[3] / 16.f + 16.f * acc2[3])) -
                         (float) blk->dmin * (
                             sumy[0] * (float) sc8[2] + sumy[1] * (float) sc8[3] +
                             sumy[2] * (float) sc8[6] + sumy[3] * (float) sc8[7]);
        }}

        y1 += 4 * QK_K;
    }}

    for (short row = 0; row < N_R0; ++row) {{
        const int r = first_row + row;
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r < N) {{
            {write}
        }}
    }}
"""


# Prefill mat-mat kernel, ported from llama.cpp kernel_mul_mm (Q5_K variant).
# 64x32 output tiles, 4 simdgroups / 128 threads per threadgroup.
# Uses vectorized dequantize_q5_K_16 to decode 16 weight values per thread into
# threadgroup memory, then runs simdgroup_multiply_accumulate on 8x8 tiles.
# NL=16 for Q5_K (QK_K / 16 = 16 dequant steps per super-block). Identical tiling
# to the Q4_K / Q6_K mat-mat kernels; only the weight decode differs.
def _q5k_matmul_source(has_bias: bool) -> str:
    bias_add = "+ (float) bias[r0 + i]" if has_bias else ""
    return f"""
    constexpr short NR0 = 64;   // weight/output rows per tile (N dim)
    constexpr short NR1 = 32;   // activation rows per tile (M dim)
    constexpr short NK  = 32;   // K-chunk per iteration
    constexpr short NL  = 16;   // Q5_K: QK_K / 16
    constexpr short NL0 = NK / 16;  // = 2 — dequant iterations per thread for weight
    constexpr short NL1 = NK / 8;   // = 4 — load iterations per thread for activation

    threadgroup half sa[4096];  // NR0 * NK weight tile (strided by 64); also reused
                                // as the 8 KB float output-staging buffer below
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
    short il  = il0;  // current dequant sub-block index within Q5_K block

    const short offset1 = il0 / NL;  // always 0 (il0 < NL0=2, NL=16)

    // Pointer to weight block for this thread's assigned row.
    device const block_q5_K * wblk = (device const block_q5_K *) weight
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
        dequantize_q5_K_16(wblk, il, temp_a);

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

        using InT2x4 = typename vec2x4<InT>::type;
        *(threadgroup half2x4 *)(sb + 64 * ib_b + 8 * ly_b) =
            (half2x4)(*(device const InT2x4 *)(yp));

        // Advance weight pointer through Q5_K sub-blocks.
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
    // drain it to device across all 128 threads. Staging is required for the
    // float->OutT cast and the optional bias add.
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


# Mat-vec launch params, matching llama.cpp Q5_K (N_R0_Q5_K = 1, N_SG_Q5_K = 2).
_Q5K_MV_N_R0 = 1
_Q5K_MV_NSG = 2
# Tile sizes for the mat-mat kernel (from llama.cpp kernel_mul_mm).
_Q5K_MM_NR0 = 64  # weight/output rows (N dim) per threadgroup
_Q5K_MM_NR1 = 32  # activation rows (M dim) per threadgroup


def _emit_q5k_matvec(
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

    n_r0 = _Q5K_MV_N_R0
    nsg = _Q5K_MV_NSG
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
            name="gguf_q5k_matvec",
            source=_q5k_matvec_source(has_bias),
            header=_Q5K_HEADER,
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


def _emit_q5k_matmul(
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
    blocks_n = (N + _Q5K_MM_NR0 - 1) // _Q5K_MM_NR0

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
            name="gguf_q5k_matmul",
            source=_q5k_matmul_source(has_bias),
            header=_Q5K_HEADER,
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


def emit_linear(
    P: MLXProgramBuilder,
    head: Node,
    x_node: Node,
    weight_node: Node,
    bias_node: Optional[Node],
) -> Slot:
    """Lower a Q5_K ``dequantize_gguf`` -> ``linear`` pattern to fused kernels.

    ``weight_node`` is the raw GGUF blob (the dequantize op's weight input) and
    ``head`` is the ``aten.linear`` node that owns the output slot.
    """
    x_slot, weight_slot, bias_slot = P.slot_map([x_node, weight_node, bias_node])

    weight_meta = weight_node.meta["val"]
    if weight_meta.dim() != 2:
        raise NotImplementedError(
            f"gguf q5k linear: weight must be 2-D (N, row_bytes); got "
            f"shape {tuple(weight_meta.shape)}"
        )
    N = weight_meta.shape[0]
    row_bytes = weight_meta.shape[1]
    if not isinstance(N, int) or not isinstance(row_bytes, int):
        raise NotImplementedError(
            "gguf q5k linear: weight shape must be statically known"
        )
    if row_bytes % Q5K_BLOCK_BYTES != 0:
        raise ValueError(
            f"gguf q5k linear: weight row bytes {row_bytes} must be a multiple of "
            f"{Q5K_BLOCK_BYTES}"
        )
    K = (row_bytes // Q5K_BLOCK_BYTES) * QK_K

    out = P.make_or_get_slot(head)
    out_dtype_int = torch_dtype_to_scalar_type(head.meta["val"].dtype)
    tile = _Q5K_MM_NR1  # M-dimension tile (activation rows per threadgroup)

    m_iov = emit_product(P, emit_shape(P, x_node, x_slot, end_dim=-1))
    emit_if_else(
        P,
        emit_sub_int(P, m_iov, IntOrVid.from_literal(1)),
        emit_then=lambda: _emit_q5k_matmul(
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
        emit_else=lambda: _emit_q5k_matvec(
            P, x_node, x_slot, weight_slot, bias_slot, N, K, out_dtype_int, out
        ),
    )
    return out
