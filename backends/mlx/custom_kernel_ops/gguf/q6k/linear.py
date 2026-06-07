#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF **Q6_K** linear implementation.

Provides the Q6_K linear pieces used by the MLX GGUF pattern handler
(:mod:`..patterns`):

* :func:`eager_linear` -- pure-torch reference (``x @ dequant(weight)^T``).
* :func:`emit_linear`  -- lowers a ``gguf_dequantize -> linear`` pattern to fused
  Q6_K Metal kernels.

Compute is keyed on the activation dtype (matching GGUF/llama.cpp): the Metal
kernels are templated on ``InT``, accumulate in ``float32``, read ``d`` as
``half``, and produce output in the activation dtype.

Two kernels are emitted depending on the number of activation rows ``M``:

    * ``M == 1`` (decode): a fused mat-vec kernel ported from llama.cpp
      ``kernel_mul_mv_q6_K_f32_impl``.
    * static ``M > 1`` (prefill): a tiled simdgroup mat-mat kernel that
      dequantizes weight tiles into threadgroup memory and reuses them across
      the activation rows.
    * dynamic/symbolic ``M`` (single program serving both prefill and decode):
      both kernels are emitted into separate instruction chains and selected at
      runtime via an ``IfNode`` on ``M`` (``M > 1`` -> mat-mat, ``M == 1`` ->
      mat-vec).

Attribution
-----------
The Q6_K Metal kernels and dequant routines here are ported from llama.cpp
(``ggml/src/ggml-metal/ggml-metal.metal`` -- ``kernel_mul_mv_q6_K_f32_impl``,
``kernel_mul_mm``, ``dequantize_q6_K``), which is MIT-licensed
(Copyright (c) 2023-2024 The ggml authors). Inline ``ported from ...`` notes
point at the specific upstream function for each kernel.
"""

from __future__ import annotations

from typing import Optional

from executorch.backends.mlx.builder.op_helpers import (
    emit_product,
    emit_shape,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.custom_kernel_ops.gguf.q6k.common import (
    _Q6K_HEADER,
    Q6K_BLOCK_BYTES,
    QK_K,
)
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    AddIntNode,
    FloorDivideIntNode,
    IfNode,
    IntOrVid,
    MetalKernelNode,
    MultiplyIntNode,
    SubtractIntNode,
)
from torch.fx.node import Node


# ---------------------------------------------------------------------------
# Metal kernel sources
# ---------------------------------------------------------------------------


# Decode mat-vec kernel, ported from llama.cpp kernel_mul_mv_q6_K_f32_impl.
# Threadgroup = (32 * NSG, 1, 1): NSG simdgroups, each computing N_R0 output
# rows for one activation row (grid.y). Accumulate in float, reduce via simd_sum.
def _q6k_matvec_source(has_bias: bool) -> str:
    write = "out[(uint)m * N + r] = (InT)(tot"
    write += " + (float)bias[r]);" if has_bias else ");"
    return f"""
    constexpr short N_R0 = 2;

    const ushort tiisg = thread_index_in_simdgroup;
    const ushort sgitg = simdgroup_index_in_threadgroup;
    const uint   m     = thread_position_in_grid.y;
    const uint   tgx   = thread_position_in_grid.x / (32u * NSG);
    const int    nb    = K / QK_K;
    const int    first_row = (int)(tgx * NSG + sgitg) * N_R0;

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;          // 0 or 1 (which 128-half)
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;

    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    device const block_q6_K * xrows = (device const block_q6_K *) weight;
    device const InT * yy = x + (uint)m * (uint)K;

    float sumf[N_R0];
    for (short r = 0; r < N_R0; ++r) {{ sumf[r] = 0.f; }}

    float yl[16];
    for (int i = ix; i < nb; i += 2) {{
        device const InT * yb = yy + i * QK_K + y_offset;
        for (short l = 0; l < 4; ++l) {{
            yl[4*l + 0] = (float) yb[l +  0];
            yl[4*l + 1] = (float) yb[l + 32];
            yl[4*l + 2] = (float) yb[l + 64];
            yl[4*l + 3] = (float) yb[l + 96];
        }}

        for (short row = 0; row < N_R0; ++row) {{
            const int r = first_row + row;
            if (r >= N) {{ break; }}
            device const block_q6_K * blk = xrows + (uint)r * nb + i;
            device const uint8_t * q1 = blk->ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = blk->qh + q_offset_h;
            device const int8_t  * sc = blk->scales + is;
            const float d = (float) blk->d;

            float4 sums = {{0.f, 0.f, 0.f, 0.f}};
            for (short l = 0; l < 4; ++l) {{
                sums[0] += yl[4*l + 0] * (float)((int8_t)((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32);
                sums[1] += yl[4*l + 1] * (float)((int8_t)((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32);
                sums[2] += yl[4*l + 2] * (float)((int8_t)((q1[l] >> 4)  | ((qh[l] & 0x30) << 0)) - 32);
                sums[3] += yl[4*l + 3] * (float)((int8_t)((q2[l] >> 4)  | ((qh[l] & 0xC0) >> 2)) - 32);
            }}
            sumf[row] += d * (sums[0]*sc[0] + sums[1]*sc[2] + sums[2]*sc[4] + sums[3]*sc[6]);
        }}
    }}

    for (short row = 0; row < N_R0; ++row) {{
        const int r = first_row + row;
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r < N) {{
            {write}
        }}
    }}
"""


# Prefill mat-mat kernel, ported from llama.cpp kernel_mul_mm (Q6_K variant).
# 64x32 output tiles, 4 simdgroups / 128 threads per threadgroup.
# Uses vectorized dequantize_q6_K_16 to decode 16 weight values per thread
# into threadgroup memory, then runs simdgroup_multiply_accumulate on 8x8
# tiles. NL=16 for Q6_K (QK_K / 16 = 16 dequant steps per super-block).
# C[m, n] = sum_k x[m, k] * dequant(weight)[n, k] (+ bias[n]).
def _q6k_matmul_source(has_bias: bool) -> str:
    bias_add = "+ (float) bias[r0 + i]" if has_bias else ""
    return f"""
    constexpr short NR0 = 64;   // weight/output rows per tile (N dim)
    constexpr short NR1 = 32;   // activation rows per tile (M dim)
    constexpr short NK  = 32;   // K-chunk per iteration
    constexpr short NL  = 16;   // Q6_K: QK_K / 16
    constexpr short NL0 = NK / 16;  // = 2 — dequant iterations per thread for weight
    constexpr short NL1 = NK / 8;   // = 4 — load iterations per thread for activation

    threadgroup half sa[4096];  // NR0 * NK storage (strided by 64)
    threadgroup half sb[4096];  // NR1 * NK storage (strided by 64)

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
    short il  = il0;  // current dequant sub-block index within Q6_K block

    const short offset1 = il0 / NL;  // always 0 for NL=16, NL0=2

    // Pointer to weight block for this thread's assigned row.
    device const block_q6_K * wblk = (device const block_q6_K *) weight
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
        dequantize_q6_K_16(wblk, il, temp_a);

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

        // Advance weight pointer through Q6_K sub-blocks.
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

    // --- Write results: always via threadgroup memory for float→InT cast ---
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

        if (sgitg == 0) {{
            for (int j = tid; j < nr1; j += NR1) {{
                device InT * D = out + (uint)(r1 + j) * (uint)N + r0;
                threadgroup float * Cp = ((threadgroup float *) sa) + j * NR0;
                for (int i = 0; i < nr0; ++i) {{
                    float v = Cp[i];
                    D[i] = (InT)(v {bias_add});
                }}
            }}
        }}
    }}
"""


# Number of simdgroups per threadgroup for the mat-vec kernel.
_Q6K_MV_NSG = 4
# Tile sizes for the mat-mat kernel (from llama.cpp kernel_mul_mm).
_Q6K_MM_NR0 = 64  # weight/output rows (N dim) per threadgroup
_Q6K_MM_NR1 = 32  # activation rows (M dim) per threadgroup


def _emit_q6k_matvec(
    P: MLXProgramBuilder,
    x_node: Node,
    x_slot: Slot,
    weight_slot: Slot,
    bias_slot: Optional[Slot],
    N: int,
    K: int,
    out: Slot,
) -> None:
    in_dtype_int = torch_dtype_to_scalar_type(x_node.meta["val"].dtype)

    leading = emit_shape(P, x_node, x_slot, end_dim=-1)
    M_iov = emit_product(P, leading)
    out_shape_flat = leading + [IntOrVid.from_literal(N)]

    n_r0 = 2
    nsg = _Q6K_MV_NSG
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
            name="gguf_q6k_matvec",
            source=_q6k_matvec_source(has_bias),
            header=_Q6K_HEADER,
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
            output_dtypes=[in_dtype_int],
            template_arg_names=["InT", "N", "K", "NSG"],
            template_arg_kinds=[2, 0, 0, 0],  # dtype, int, int, int
            template_arg_values=[in_dtype_int, N, K, nsg],
        )
    )


def _emit_q6k_matmul(
    P: MLXProgramBuilder,
    x_node: Node,
    x_slot: Slot,
    weight_slot: Slot,
    bias_slot: Optional[Slot],
    N: int,
    K: int,
    blocks_m_iov: IntOrVid,
    out: Slot,
) -> None:
    in_dtype_int = torch_dtype_to_scalar_type(x_node.meta["val"].dtype)

    leading = emit_shape(P, x_node, x_slot, end_dim=-1)
    out_shape_flat = leading + [IntOrVid.from_literal(N)]

    # grid.x = ceil(M / NR1) * 128 threads (activation tiles)
    # grid.y = ceil(N / NR0) (weight tiles)
    blocks_n = (N + _Q6K_MM_NR0 - 1) // _Q6K_MM_NR0

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
            name="gguf_q6k_matmul",
            source=_q6k_matmul_source(has_bias),
            header=_Q6K_HEADER,
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
            output_dtypes=[in_dtype_int],
            template_arg_names=["InT", "N", "K"],
            template_arg_kinds=[2, 0, 0],
            template_arg_values=[in_dtype_int, N, K],
        )
    )


def emit_linear(
    P: MLXProgramBuilder,
    head: Node,
    x_node: Node,
    weight_node: Node,
    bias_node: Optional[Node],
) -> Slot:
    """Lower a Q6_K ``gguf_dequantize`` -> ``linear`` pattern to fused kernels.

    ``weight_node`` is the raw GGUF blob (the dequantize op's weight input) and
    ``head`` is the ``aten.linear`` node that owns the output slot.
    """
    x_slot, weight_slot, bias_slot = P.slot_map([x_node, weight_node, bias_node])

    weight_meta = weight_node.meta["val"]
    if weight_meta.dim() != 2:
        raise NotImplementedError(
            f"gguf q6k linear: weight must be 2-D (N, row_bytes); got "
            f"shape {tuple(weight_meta.shape)}"
        )
    N = weight_meta.shape[0]
    row_bytes = weight_meta.shape[1]
    if not isinstance(N, int) or not isinstance(row_bytes, int):
        raise NotImplementedError(
            "gguf q6k linear: weight shape must be statically known"
        )
    if row_bytes % Q6K_BLOCK_BYTES != 0:
        raise ValueError(
            f"gguf q6k linear: weight row bytes {row_bytes} must be a multiple of "
            f"{Q6K_BLOCK_BYTES}"
        )
    K = (row_bytes // Q6K_BLOCK_BYTES) * QK_K

    # Determine M (product of x's leading dims). Static M lets us pick the
    # optimal kernel and (for mat-mat) compute a literal launch grid.
    x_meta = x_node.meta["val"]
    leading_dims = x_meta.shape[:-1]
    M: Optional[int] = 1
    for d in leading_dims:
        if isinstance(d, int):
            M *= d
        else:
            M = None  # dynamic / symbolic
            break

    out = P.make_or_get_slot(head)
    tile = _Q6K_MM_NR1  # M-dimension tile (activation rows per threadgroup)
    if M == 1:
        # Static decode -> mat-vec.
        _emit_q6k_matvec(P, x_node, x_slot, weight_slot, bias_slot, N, K, out)
    elif M is not None:
        # Static prefill -> tiled simdgroup mat-mat (literal grid).
        blocks_m = (M + tile - 1) // tile
        _emit_q6k_matmul(
            P,
            x_node,
            x_slot,
            weight_slot,
            bias_slot,
            N,
            K,
            IntOrVid.from_literal(blocks_m),
            out,
        )
    else:
        # Dynamic seqlen -> emit both kernels in separate chains and select at
        # runtime with an IfNode. cond = M - 1: nonzero (M>1) runs the mat-mat
        # (then) chain, zero (M==1) runs the mat-vec (else) chain.
        leading = emit_shape(P, x_node, x_slot, end_dim=-1)
        m_iov = emit_product(P, leading)

        _, cond_slot = P.make_tmp_value_slot()
        P.emit(
            SubtractIntNode(
                a=m_iov,
                b=IntOrVid.from_literal(1),
                out=P.slot_to_vid(cond_slot),
            )
        )
        cond_iov = IntOrVid.from_vid(P.slot_to_vid(cond_slot))

        # blocks_m = (M + tile - 1) // tile  (mat-mat grid.y).
        _, sum_slot = P.make_tmp_value_slot()
        P.emit(
            AddIntNode(
                a=m_iov,
                b=IntOrVid.from_literal(tile - 1),
                out=P.slot_to_vid(sum_slot),
            )
        )
        _, blocks_m_slot = P.make_tmp_value_slot()
        P.emit(
            FloorDivideIntNode(
                a=IntOrVid.from_vid(P.slot_to_vid(sum_slot)),
                b=IntOrVid.from_literal(tile),
                out=P.slot_to_vid(blocks_m_slot),
            )
        )
        blocks_m_iov = IntOrVid.from_vid(P.slot_to_vid(blocks_m_slot))

        with P.new_chain() as then_idx:  # prefill / mat-mat
            _emit_q6k_matmul(
                P,
                x_node,
                x_slot,
                weight_slot,
                bias_slot,
                N,
                K,
                blocks_m_iov,
                out,
            )
        with P.new_chain() as else_idx:  # decode / mat-vec
            _emit_q6k_matvec(P, x_node, x_slot, weight_slot, bias_slot, N, K, out)

        P.emit(
            IfNode(
                cond=cond_iov,
                then_chain_idx=then_idx,
                else_chain_idx=else_idx,
            )
        )
    return out
