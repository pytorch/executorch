#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark ``mlx::int8_intx_gemm`` (dynamic per-token int8 activations x
per-channel int4 or int2 weights, INT32 accumulation; kernels ``int8_int4`` and
``int8_int2``) and ``mlx::fp_int4_gemm`` (fp16 / bf16 activations x per-channel
int4 weights, FP32 accumulation) against MLX ``quantized_matmul`` (4-bit affine,
group size 32) on every Muse Glimmer GEMM shape, and estimate total Muse
Glimmer GEMM time per forward pass.

For each activation dtype, shape, M (token batch size), and kernel
(``--kernels``), every tile config is checked against a reference and timed.
The sweep matches what the op lowers to:
    * M <= ``--matvec-max-m`` (4): the matvec kernel, which streams packed
      weights like MLX's qmv. Rows per threadgroup MR in {1, 2, 4} up to M are
      tried; MR < M launches ceil(M / MR) threadgroup rows (MR = 3 compiles
      badly and is skipped). fp_int4 also sweeps the qmv_wide-style kernel:
      ``--wide-kl`` lanes per channel, M or ceil(M / 2) rows per threadgroup.
    * M >= ``--mpp-min-m`` (3), or no matvec (K % 512 != 0): MPP tiles (BM x
      BN, BM capped at max(8, M) since larger
      tiles only add padding rows, and BM >= 32 for M >= 128). Full-K
      (``--splits`` 1) runs MPP over all
      of K; split-K (S > 1) gives each of S threadgroups per output tile K / S
      and a reduce kernel adds the partials and applies the scales.
    M in both ranges sweeps both and keeps the fastest, which finds the
    decode / MPP crossover (int8 plans switch to MPP at M = 3, fp at M = 5).
int8_int4 / int8_int2 quantize activations first (included in their times).

``mlx::int8_int4_affine_gemm`` / ``mlx::fp_int4_affine_gemm`` (asymmetric
activations and int4 weights with a scale and bias per group) are opt-in
kernels ``{int8,fp}_affine_{gs64,gs128,gs256,pc}`` (``pc`` = per channel).
Small M sweeps their matvec and qmv_wide-style kernels; larger M sweeps
per-SIMD-group MPP tiles (16 x 64 / 16 x 32 sub-tiles below BM = 32, 32 x 32
above), splits that cover whole groups, and for grouped weights the bias term
as an MPP call (b1) or after the group loop (b3). Their prologue (int8 quantize
or fp group sums) is included in their times; scale / bias planes are
``--affine-qparam-dtype``.
Tile selection has two rounds: every config is timed with ``--iters``, then
the ``--refine-top`` fastest configs of each lowering (full-K, split-K) are
re-timed with ``--refine-iters`` and the fastest is kept. Each kernel's winner
is then re-timed against MLX: they alternate ``--final-repeats`` times and the
median of each is the reported (official) number, so all are measured under
the same GPU conditions. The Muse Glimmer estimate is ``sum over shapes of
(official time x times the shape occurs in the model)``.

``PRESETS`` combine the per-shape winners into mixed-scheme forwards (a kernel
per layer group: attention, FFN, lm_head; MLX 4-bit or our affine kernels with
int8 and fp activations). ``--save-json`` keeps the per-shape results so
``--load-json`` can re-print the reports and presets without re-running.

Just run::

    python -m executorch.backends.mlx.custom_kernel_ops.test.bench_muse_glimmer_gemm

and send back the output. ``--verbose`` prints every tile config.

Timing: each timing first runs ``--warmup`` untimed calls (so the kernel is
compiled), then N calls built lazily as a dependent chain (each call's input
is ``x0 + 0 * previous_output[:, :1]``) and timed as a single ``mx.eval``.
The chain serializes calls like consecutive layers in a model; independent
calls would let the GPU overlap them and hide per-call latency. The same chain
without the GEMM is timed and subtracted, and the result is the average per
call. N targets ``--time-budget-ms`` of GPU time per timing (estimated from one
call), between ``--min-iters`` and a per-round cap: ``--iters`` for the tile
sweep, ``--refine-iters`` for the refine round, and ``--final-iters`` for the
official re-run. ``--shape-pause`` seconds between shapes and
``--dtype-pause`` seconds between activation dtypes.
Calls rotate across enough weight copies to exceed ``--rotate-mb`` so weights
are not served from cache.

Requires MLX and an MPP-capable GPU (Metal 4).
"""

import argparse
import json
import math
import statistics
import time
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx
from executorch.backends.mlx.custom_kernel_ops import (
    fp_int4_gemm as fp4,
    int4_affine_gemm as aff,
    int8_intx_gemm as op,
)

# Muse Glimmer (MuseGlimmerConfig defaults): 52 identical decoder layers with
# fused Q/K/V/output-gate and fused gate/up projections, plus lm_head.
_DIM = 6656
_N_LAYERS = 52
_Q_DIM = 32 * 128
_KV_DIM = 2 * 128
_FFN_DIM = 3 * _DIM
_VOCAB = 202048

# name -> (N, K, occurrences per forward) for y[M, N] = x[M, K] @ W[N, K].T
SHAPES: Dict[str, Tuple[int, int, int]] = {
    "qkv": (_Q_DIM + 2 * _KV_DIM + _Q_DIM, _DIM, _N_LAYERS),
    "o": (_DIM, _Q_DIM, _N_LAYERS),
    "gate_up": (2 * _FFN_DIM, _DIM, _N_LAYERS),
    "down": (_DIM, _FFN_DIM, _N_LAYERS),
    "lm_head": (_VOCAB, _DIM, 1),
}
DTYPES = {"fp16": mx.float16, "bf16": mx.bfloat16}

# Layer groups of a forward pass, for presets that quantize them differently.
GROUPS: Dict[str, Tuple[str, ...]] = {
    "attn": ("qkv", "o"),
    "ffn": ("gate_up", "down"),
    "lm_head": ("lm_head",),
}
# Mixed-scheme presets: kernel per layer group, "{act}" filled with int8 and
# fp, "mlx" = MLX 4-bit affine at --group-size. Numbers refer to the
# experiment list; KV-cache quantization (bf16, MLX Q4 at 5.0 bpw, Q4 gs=32 at
# 4.5 bpw) only changes SDPA, so presets that differ only in KV share a GEMM
# time. lm_head follows FFN.
PRESETS: Dict[str, Dict[str, str]] = {
    "1/2/5 pc FFN + pc ATTN": {
        "ffn": "{act}_affine_pc",
        "attn": "{act}_affine_pc",
        "lm_head": "{act}_affine_pc",
    },
    "3/6 pc FFN + mlx ATTN": {
        "ffn": "{act}_affine_pc",
        "attn": "mlx",
        "lm_head": "{act}_affine_pc",
    },
    "4/7 gs256 FFN + gs256 ATTN": {
        "ffn": "{act}_affine_gs256",
        "attn": "{act}_affine_gs256",
        "lm_head": "{act}_affine_gs256",
    },
    "mlx FFN + mlx ATTN": {"ffn": "mlx", "attn": "mlx", "lm_head": "mlx"},
}
DEFAULT_KERNELS = ("int8_int4", "int8_int2", "fp_int4")
AFFINE_KERNELS = tuple(
    f"{act}_affine_{'pc' if gs is None else f'gs{gs}'}"
    for act in aff.ACTS
    for gs in aff.GROUP_SIZES
)
KERNELS = DEFAULT_KERNELS + AFFINE_KERNELS
# Weight bits per kernel.
KERNEL_BITS = {
    "int8_int4": 4,
    "int8_int2": 2,
    "fp_int4": 4,
    **{k: 4 for k in AFFINE_KERNELS},
}
MATVEC = 0

# (BM, BN, splits): an MPP tile, split over K when splits > 1. BN == MATVEC
# selects the matvec kernel with BM activation rows per threadgroup; BN < 0
# (fp_int4 and affine only) the qmv_wide-style kernel with BM rows per
# threadgroup and -BN lanes per output channel. Affine MPP tiles append
# (SM, SN, BIAS): the per-SIMD-group sub-tile and the bias mode.
Tile = Tuple[int, ...]
# (tile, seconds or None on failure, check error or None)
Entry = Tuple[Tile, Optional[float], Optional[float]]

_quant_kernel = mx.fast.metal_kernel(
    name="int8_intx_gemm_quantize",
    input_names=["x"],
    output_names=["xq", "x_scale"],
    source=op._QUANT_SOURCE,
    header=op._HEADER,
)
_split_k_reduce_kernel = mx.fast.metal_kernel(
    name="int8_intx_gemm_split_k_reduce",
    input_names=["partial", "x_scale", "w_scale"],
    output_names=["out"],
    source=op._SPLIT_K_REDUCE_SOURCE,
    header=op._HEADER,
)
_full_k_kernels = {
    b: mx.fast.metal_kernel(
        name=f"int8_int{b}_gemm_full_k",
        input_names=["xq", "x_scale", "w_packed", "w_scale"],
        output_names=["out"],
        source=op._FULL_K_SOURCES[b],
        header=op._HEADER,
    )
    for b in op.BITS
}
_split_k_kernels = {
    b: mx.fast.metal_kernel(
        name=f"int8_int{b}_gemm_split_k",
        input_names=["xq", "x_scale", "w_packed"],
        output_names=["partial"],
        source=op._SPLIT_K_SOURCES[b],
        header=op._HEADER,
    )
    for b in op.BITS
}
_matvec_kernels = {
    b: mx.fast.metal_kernel(
        name=f"int8_int{b}_gemm_matvec",
        input_names=["xq", "x_scale", "w_packed", "w_scale"],
        output_names=["out"],
        source=op._MATVEC_SOURCES[b],
        header=op._HEADER,
    )
    for b in op.BITS
}
_fp4_full_k_kernel = mx.fast.metal_kernel(
    name="fp_int4_gemm_full_k",
    input_names=["x", "w_packed", "w_scale"],
    output_names=["out"],
    source=fp4._FULL_K_SOURCE,
    header=fp4._HEADER,
)
_fp4_split_k_kernel = mx.fast.metal_kernel(
    name="fp_int4_gemm_split_k",
    input_names=["x", "w_packed"],
    output_names=["partial"],
    source=fp4._SPLIT_K_SOURCE,
    header=fp4._HEADER,
)
_fp4_split_k_reduce_kernel = mx.fast.metal_kernel(
    name="fp_int4_gemm_split_k_reduce",
    input_names=["partial", "w_scale"],
    output_names=["out"],
    source=fp4._SPLIT_K_REDUCE_SOURCE,
    header=fp4._HEADER,
)
_fp4_matvec_kernel = mx.fast.metal_kernel(
    name="fp_int4_gemm_matvec",
    input_names=["x", "w_packed", "w_scale"],
    output_names=["out"],
    source=fp4._MATVEC_SOURCE,
    header=fp4._HEADER,
)
_fp4_wide_kernel = mx.fast.metal_kernel(
    name="fp_int4_gemm_wide",
    input_names=["x", "w_packed", "w_scale"],
    output_names=["out"],
    source=fp4._WIDE_SOURCE,
    header=fp4._HEADER,
)


def _affine_kernel_io(act: str, kind: str) -> Tuple[List[str], List[str]]:
    x_in = ["xq"] if act == "int8" else ["x"]
    epilogue = ["x_scale", "x_zero", "w_sum"] if act == "int8" else []
    w = ["w_packed", "w_scale", "w_bias"]
    return {
        "quantize": (["x"], ["xq", "x_scale", "x_zero", "x_sum"]),
        "x_sum": (["x"], ["x_sum"]),
        "matvec": (x_in + w + epilogue, ["out"]),
        "wide": (x_in + w + epilogue, ["out"]),
        "full_k": (x_in + ["x_sum"] + w + epilogue, ["out"]),
        "split_k": (x_in + ["x_sum"] + w + epilogue[:1], ["partial"]),
        "reduce": (["partial"] + epilogue, ["out"]),
        "reduce_raw": (["partial", "x_sum", "w_scale", "w_bias"] + epilogue, ["out"]),
    }[kind]


_affine_kernels = {
    (act, kind): mx.fast.metal_kernel(
        name=f"{act}_int4_affine_gemm_{kind}",
        input_names=_affine_kernel_io(act, kind)[0],
        output_names=_affine_kernel_io(act, kind)[1],
        source={"quantize": aff._INT8_QUANT_SOURCE, "x_sum": aff._FP_SUM_SOURCE}.get(
            kind, aff._SOURCES[act].get(kind)
        ),
        header=aff._HEADER,
    )
    for act in aff.ACTS
    for kind in (
        "matvec",
        "wide",
        "full_k",
        "split_k",
        "reduce",
        "reduce_raw",
        "quantize" if act == "int8" else "x_sum",
    )
}


def affine_spec(kernel: str) -> Tuple[str, Optional[int]]:
    """``{act}_affine_{gs<N>|pc}`` -> (act, group size or None)."""
    act, _, gs = kernel.split("_")
    return act, None if gs == "pc" else int(gs[2:])


def config_name(tile: Tile) -> str:
    bm, bn, splits = tile[:3]
    if bn == MATVEC:
        return f"matvec mr{bm}"
    if bn < 0:
        return f"wide nv{bm} kl{-bn}"
    sub = f" sg{tile[3]}x{tile[4]} b{tile[5]}" if len(tile) > 3 else ""
    if splits > 1:
        return f"split_k{splits} {bm}x{bn}{sub}"
    return f"full_k {bm}x{bn}{sub}"


def lowering(tile: Tile) -> str:
    return config_name(tile).split()[0].rstrip("0123456789")


def int8_intx_gemm(
    x: mx.array,
    w_packed: mx.array,
    w_scale: mx.array,
    tile: Tile,
    bits: int,
    quant_threads: int = op._QUANT_THREADS,
) -> mx.array:
    M, K = x.shape
    N = w_packed.shape[0]
    bm, bn, splits = tile
    if bn == MATVEC:
        xq, x_scale = _quant_kernel(
            inputs=[x],
            template=[("T", x.dtype), ("K", K), ("TPR", quant_threads)],
            grid=(quant_threads, M, 1),
            threadgroup=(quant_threads, 1, 1),
            output_shapes=[(M, K), (M,)],
            output_dtypes=[mx.int8, mx.float32],
        )
        rows_per_tg = op._MATVEC_SIMDGROUPS * op._MATVEC_ROWS_PER_SIMDGROUP
        return _matvec_kernels[bits](
            inputs=[xq, x_scale, w_packed, w_scale],
            template=[("T", x.dtype), ("N", N), ("K", K), ("MR", bm)],
            grid=(
                math.ceil(N / rows_per_tg) * op._MATVEC_SIMDGROUPS * 32,
                math.ceil(M / bm),
                1,
            ),
            threadgroup=(op._MATVEC_SIMDGROUPS * 32, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[x.dtype],
        )[0]
    launch = {
        "grid": (math.ceil(N / bn) * op._THREADS, math.ceil(M / bm), 1),
        "threadgroup": (op._THREADS, 1, 1),
        "output_shapes": [(M, N)],
        "output_dtypes": [x.dtype],
    }
    template = [("T", x.dtype), ("N", N), ("K", K), ("BM", bm), ("BN", bn)]
    # Zero-pad xq to whole BM tiles, as the op always does on the MPP path.
    rows = math.ceil(M / bm) * bm
    xq, x_scale = _quant_kernel(
        inputs=[x],
        template=[("T", x.dtype), ("K", K), ("TPR", quant_threads)],
        grid=(quant_threads, rows, 1),
        threadgroup=(quant_threads, 1, 1),
        output_shapes=[(rows, K), (M,)],
        output_dtypes=[mx.int8, mx.float32],
    )
    if splits > 1:
        partial = _split_k_kernels[bits](
            inputs=[xq, x_scale, w_packed],
            template=template + [("SPLITS", splits)],
            grid=(math.ceil(N / bn) * op._THREADS, math.ceil(M / bm), splits),
            threadgroup=(op._THREADS, 1, 1),
            output_shapes=[(splits, M, N)],
            output_dtypes=[mx.int32],
        )[0]
        return _split_k_reduce_kernel(
            inputs=[partial, x_scale, w_scale],
            template=[("T", x.dtype), ("N", N), ("SPLITS", splits)],
            grid=(math.ceil(M * N / 256) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[x.dtype],
        )[0]
    return _full_k_kernels[bits](
        inputs=[xq, x_scale, w_packed, w_scale], template=template, **launch
    )[0]


def fp_int4_gemm(
    x: mx.array, w_packed: mx.array, w_scale: mx.array, tile: Tile
) -> mx.array:
    M, K = x.shape
    N = w_packed.shape[0]
    bm, bn, splits = tile
    if bn == MATVEC:
        rows_per_tg = fp4._MATVEC_SIMDGROUPS * fp4._MATVEC_ROWS_PER_SIMDGROUP
        return _fp4_matvec_kernel(
            inputs=[x, w_packed, w_scale],
            template=[("T", x.dtype), ("N", N), ("K", K), ("MR", bm)],
            grid=(
                math.ceil(N / rows_per_tg) * fp4._MATVEC_SIMDGROUPS * 32,
                math.ceil(M / bm),
                1,
            ),
            threadgroup=(fp4._MATVEC_SIMDGROUPS * 32, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[x.dtype],
        )[0]
    if bn < 0:
        kl = -bn
        channels_per_tg = fp4._MATVEC_SIMDGROUPS * (32 // kl)
        return _fp4_wide_kernel(
            inputs=[x, w_packed, w_scale],
            template=[("T", x.dtype), ("N", N), ("K", K), ("NV", bm), ("KL", kl)],
            grid=(
                math.ceil(N / channels_per_tg) * fp4._MATVEC_SIMDGROUPS * 32,
                math.ceil(M / bm),
                1,
            ),
            threadgroup=(fp4._MATVEC_SIMDGROUPS * 32, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[x.dtype],
        )[0]
    template = [("T", x.dtype), ("N", N), ("K", K), ("BM", bm), ("BN", bn)]
    grid = (math.ceil(N / bn) * fp4._THREADS, math.ceil(M / bm), 1)
    if splits > 1:
        partial = _fp4_split_k_kernel(
            inputs=[x, w_packed],
            template=template + [("SPLITS", splits)],
            grid=grid[:2] + (splits,),
            threadgroup=(fp4._THREADS, 1, 1),
            output_shapes=[(splits, M, N)],
            output_dtypes=[mx.float32],
        )[0]
        return _fp4_split_k_reduce_kernel(
            inputs=[partial, w_scale],
            template=[("T", x.dtype), ("N", N), ("SPLITS", splits)],
            grid=(math.ceil(M * N / 256) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[x.dtype],
        )[0]
    return _fp4_full_k_kernel(
        inputs=[x, w_packed, w_scale],
        template=template,
        grid=grid,
        threadgroup=(fp4._THREADS, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]


def affine_gemm(  # noqa: C901
    x: mx.array,
    w_packed: mx.array,
    w_scale: mx.array,
    w_bias: mx.array,
    w_sum: mx.array,
    tile: Tile,
    act: str,
    group_size: Optional[int],
) -> mx.array:
    """The kernels ``mlx::{act}_int4_affine_gemm`` lowers ``tile`` to."""
    M, K = x.shape
    N = w_packed.shape[0]
    gs = group_size or K
    bm, bn, splits = tile[:3]
    T = ("T", x.dtype)
    out_spec = {"output_shapes": [(M, N)], "output_dtypes": [x.dtype]}

    def prologue(rows: int, xg: int) -> List[mx.array]:
        tmpl = [T, ("K", K), ("TPR", aff._PROLOGUE_THREADS), ("XG", xg)]
        if act == "int8":
            return _affine_kernels[("int8", "quantize")](
                inputs=[x],
                template=tmpl,
                grid=(aff._PROLOGUE_THREADS, rows, 1),
                threadgroup=(aff._PROLOGUE_THREADS, 1, 1),
                output_shapes=[(rows, K), (M,), (M,), (rows, K // xg)],
                output_dtypes=[mx.int8, mx.float32, mx.float32, mx.float32],
            )
        blocks_per_tg = aff._PROLOGUE_THREADS // 32
        return _affine_kernels[("fp", "x_sum")](
            inputs=[x],
            template=tmpl,
            grid=(
                math.ceil(K // xg / blocks_per_tg) * aff._PROLOGUE_THREADS,
                rows,
                1,
            ),
            threadgroup=(aff._PROLOGUE_THREADS, 1, 1),
            output_shapes=[(rows, K // xg)],
            output_dtypes=[mx.float32],
        )

    w = [w_packed, w_scale, w_bias]
    if bn <= 0:
        if act == "int8":
            xq, x_scale, x_zero, _ = prologue(M, K)
            x_in, epilogue = [xq], [x_scale, x_zero, w_sum]
        else:
            x_in, epilogue = [x], []
        if bn == MATVEC:
            rows_per_tg = aff._MATVEC_SIMDGROUPS * aff._MATVEC_ROWS_PER_SIMDGROUP
            return _affine_kernels[(act, "matvec")](
                inputs=x_in + w + epilogue,
                template=[T, ("N", N), ("K", K), ("MR", bm), ("GS", gs)],
                grid=(
                    math.ceil(N / rows_per_tg) * aff._MATVEC_SIMDGROUPS * 32,
                    math.ceil(M / bm),
                    1,
                ),
                threadgroup=(aff._MATVEC_SIMDGROUPS * 32, 1, 1),
                **out_spec,
            )[0]
        kl = -bn
        channels_per_tg = aff._MATVEC_SIMDGROUPS * (32 // kl)
        return _affine_kernels[(act, "wide")](
            inputs=x_in + w + epilogue,
            template=[T, ("N", N), ("K", K), ("NV", bm), ("KL", kl), ("GS", gs)],
            grid=(
                math.ceil(N / channels_per_tg) * aff._MATVEC_SIMDGROUPS * 32,
                math.ceil(M / bm),
                1,
            ),
            threadgroup=(aff._MATVEC_SIMDGROUPS * 32, 1, 1),
            **out_spec,
        )[0]

    sm, sn, bias = tile[3:]
    rows = math.ceil(M / bm) * bm
    xg = K // splits if gs == K else gs
    if act == "int8":
        xq, x_scale, x_zero, x_sum = prologue(rows, xg)
        x_in, epilogue = [xq], [x_scale, x_zero, w_sum]
    else:
        (x_sum,) = prologue(rows, xg)
        x_in, epilogue = [x], []
    threads = (bm // sm) * (bn // sn) * 32
    launch = {
        "grid": (math.ceil(N / bn) * threads, math.ceil(M / bm), splits),
        "threadgroup": (threads, 1, 1),
    }
    template = [
        T,
        ("ST", w_scale.dtype),
        ("N", N),
        ("K", K),
        ("BM", bm),
        ("BN", bn),
        ("SM", sm),
        ("SN", sn),
        ("NM", aff._SG_ROWS_PER_THREAD.get((sm, sn), 4)),
        ("GS", gs),
        ("SPLITS", splits),
        ("BIAS", bias),
    ]
    if splits == 1:
        return _affine_kernels[(act, "full_k")](
            inputs=x_in + [x_sum] + w + epilogue,
            template=template,
            **launch,
            **out_spec,
        )[0]
    raw = gs == K
    partial = _affine_kernels[(act, "split_k")](
        inputs=x_in + [x_sum] + w + epilogue[:1],
        template=template,
        output_shapes=[(splits, M, N)],
        output_dtypes=[mx.int32 if raw and act == "int8" else mx.float32],
        **launch,
    )[0]
    vw = 4 if N % 4 == 0 else 1
    reduce_in = [partial, x_sum, w_scale, w_bias] if raw else [partial]
    return _affine_kernels[(act, "reduce_raw" if raw else "reduce")](
        inputs=reduce_in + epilogue,
        template=[T, ("N", N), ("SPLITS", splits)],
        grid=(math.ceil(M * N // vw / 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        **out_spec,
    )[0]


def _unpack_wq(w_packed: mx.array, cols: int, bits: int) -> mx.array:
    w = w_packed[:cols].astype(mx.int32)
    mask = (1 << bits) - 1
    fields = mx.stack([(w >> s) & mask for s in range(0, 8, bits)], axis=-1)
    fields = fields.reshape(cols, -1)
    half = 1 << (bits - 1)
    return mx.where(fields >= half, fields - 2 * half, fields).astype(mx.float32)


def _rel_err(out: mx.array, ref: mx.array) -> float:
    err = mx.max(mx.abs(out.astype(mx.float32) - ref)) / mx.max(mx.abs(ref))
    return err.item()


def check_int8_intx(
    x: mx.array,
    w_packed: mx.array,
    w_scale: mx.array,
    tile: Tile,
    bits: int,
    cols: int = 256,
    **kwargs,
) -> float:
    """Max error of the first ``cols`` outputs, relative to max |reference|."""
    out = int8_intx_gemm(x, w_packed, w_scale, tile, bits, **kwargs)[:, :cols]
    xf = x.astype(mx.float32)
    x_scale = mx.maximum(mx.max(mx.abs(xf), axis=-1, keepdims=True) / 127, op._EPS)
    xq = mx.clip(mx.round(xf * mx.reciprocal(x_scale)), -128, 127)
    wq = _unpack_wq(w_packed, cols, bits)
    ref = (xq @ wq.T) * x_scale * w_scale[:cols].astype(mx.float32)
    return _rel_err(out, ref)


def check_fp_int4(
    x: mx.array, w_packed: mx.array, w_scale: mx.array, tile: Tile, cols: int = 256
) -> float:
    """Max error of the first ``cols`` outputs, relative to max |reference|."""
    out = fp_int4_gemm(x, w_packed, w_scale, tile)[:, :cols]
    wq = _unpack_wq(w_packed, cols, 4)
    ref = (x.astype(mx.float32) @ wq.T) * w_scale[:cols].astype(mx.float32)
    return _rel_err(out, ref)


def check_affine(
    x: mx.array,
    w_packed: mx.array,
    w_scale: mx.array,
    w_bias: mx.array,
    w_sum: mx.array,
    tile: Tile,
    act: str,
    group_size: Optional[int],
    cols: int = 256,
) -> float:
    """Max error of the first ``cols`` outputs against the op's formula,
    relative to max |reference| (``w_sum`` need not be the true row sums)."""
    out = affine_gemm(x, w_packed, w_scale, w_bias, w_sum, tile, act, group_size)
    K = x.shape[1]
    gs = group_size or K
    wq = _unpack_wq(w_packed, cols, 4).reshape(cols, K // gs, gs)
    s = w_scale[:, :cols].T.astype(mx.float32)[..., None]
    b = w_bias[:, :cols].T.astype(mx.float32)[..., None]
    w = (wq * s + b).reshape(cols, K)
    xf = x.astype(mx.float32)
    if act == "fp":
        return _rel_err(out[:, :cols], xf @ w.T)
    lo = mx.minimum(mx.min(xf, axis=-1, keepdims=True), 0)
    hi = mx.maximum(mx.max(xf, axis=-1, keepdims=True), 0)
    x_scale = mx.maximum((hi - lo) / 255, aff._EPS)
    x_zero = -128 - lo / x_scale
    xq = mx.clip(mx.round((xf - lo) * mx.reciprocal(x_scale)), 0, 255) - 128
    ref = x_scale * (xq @ w.T - x_zero * w_sum[:cols].astype(mx.float32))
    return _rel_err(out[:, :cols], ref)


def time_per_call(
    fn: Callable[[mx.array, int], mx.array],
    x0: mx.array,
    warmup: int,
    max_iters: int,
    budget_s: float,
    min_iters: int,
) -> float:
    """Average serialized seconds per call; ``fn(x, i)`` must build (not
    evaluate) run ``i`` on input ``x``. Runs about ``budget_s`` worth of calls
    (estimated from one evaluated call), clamped to [min_iters, max_iters]."""

    def chain(call: Optional[Callable[[mx.array, int], mx.array]], iters: int) -> float:
        x = x0
        for i in range(iters):
            y = call(x, i) if call is not None else x
            x = x0 + 0 * y[:, :1]
        start = time.perf_counter()
        mx.eval(x)
        return time.perf_counter() - start

    for i in range(warmup):
        mx.eval(fn(x0, i))
    estimate = chain(fn, 1)
    iters = max(min_iters, min(max_iters, math.ceil(budget_s / max(estimate, 1e-9))))
    chain(None, iters)
    return max(chain(fn, iters) - chain(None, iters), 0.0) / iters


def affine_tile_configs(
    K: int, M: int, args: argparse.Namespace, kernel: str
) -> List[Tile]:
    """Like tile_configs, for the affine kernels' per-SIMD-group MPP tiles:
    BM % 16 == 0 (16 x 64 / 16 x 32 sub-tiles below BM = 32, 32 x 32 above),
    at most 1024 threads, splits that cover whole groups, and for grouped
    weights each ``--affine-bias`` mode (b1 needs a BM x BN fp32 tile in
    threadgroup memory, so BM * BN <= 8192)."""
    _, group_size = affine_spec(kernel)
    gs = group_size or K
    min_bm = 32 if M >= 128 else 16
    tiles = []
    for bm in args.bm:
        if bm % 16 or not min_bm <= bm <= max(16, M):
            continue
        for bn in args.affine_bn:
            sm, sn = (32, 32) if bm >= 32 else (16, 64 if bn >= 64 else 32)
            if bn % sn or (bm // sm) * (bn // sn) * 32 > 1024:
                continue
            for splits in args.splits:
                if (K // gs) % splits if gs < K else K % (64 * splits):
                    continue
                for bias in args.affine_bias if gs < K else (0,):
                    if bias == 1 and bm * bn > 8192:
                        continue
                    tiles.append((bm, bn, splits, sm, sn, bias))
    return tiles


def tile_configs(
    K: int, M: int, args: argparse.Namespace, kernel: str = ""
) -> List[Tile]:
    """Matvec for M <= --matvec-max-m (needs K % 512 == 0), plus the
    qmv_wide-style kernel for fp_int4 and the affine kernels; MPP tiles for
    M >= --mpp-min-m or when there is no matvec, so M in between sweeps both.
    MPP needs BM % 8 == 0, BM > max(8, M) only adds padding rows, BM < 32
    never won at M >= 128, and split-K needs K % (64 * splits) == 0."""
    tiles: List[Tile] = []
    if M <= args.matvec_max_m and K % op._MATVEC_BLOCK == 0:
        # MR < M launches ceil(M / MR) threadgroup rows that share weights
        # through cache, like MLX's qmv (one row per threadgroup). MR = 3 was
        # 2-5x slower than MR = 2 or 4 on every Muse Glimmer shape.
        tiles = [(mr, MATVEC, 1) for mr in (1, 2, 4) if mr <= M]
        if kernel == "fp_int4" or kernel in AFFINE_KERNELS:
            tiles += [
                (nv, -kl, 1)
                for nv in sorted({M, (M + 1) // 2})
                for kl in args.wide_kl
                if K % (16 * kl) == 0
            ]
    if tiles and M < args.mpp_min_m:
        return tiles
    if kernel in AFFINE_KERNELS:
        return tiles + affine_tile_configs(K, M, args, kernel)
    min_bm = 32 if M >= 128 else 8
    return tiles + [
        (bm, bn, splits)
        for bm in args.bm
        if bm % 8 == 0 and min_bm <= bm <= max(8, M)
        for bn in args.bn
        for splits in args.splits
        if splits == 1 or K % (64 * splits) == 0
    ]


def passed(entries: List[Entry]) -> List[Entry]:
    """Configs that ran and passed the accuracy check."""
    return [e for e in entries if e[1] is not None and e[2] is not None and e[2] < 1e-2]


def refine_candidates(entries: List[Entry], top: int) -> List[Tile]:
    """The ``top`` fastest passing tiles of each lowering (full-K, split-K,
    matvec)."""
    ok = sorted(passed(entries), key=lambda e: e[1])
    by_lowering: Dict[str, List[Tile]] = {}
    for tile, _, _ in ok:
        group = by_lowering.setdefault(lowering(tile), [])
        if len(group) < top:
            group.append(tile)
    return [tile for group in by_lowering.values() for tile in group]


def _random_bytes(shape: Tuple[int, ...]) -> mx.array:
    return mx.random.randint(0, 256, shape).astype(mx.uint8)


def bench_shape(  # noqa: C901
    name: str, dtype_name: str, ms: List[int], args: argparse.Namespace
) -> Dict[int, dict]:
    """Returns {M: {"mlx": seconds or None, "best": {kernel: (Tile, seconds)
    or None}}}; ``mlx`` and ``best`` are the official medians."""
    N, K, _ = SHAPES[name]
    dtype = DTYPES[dtype_name]
    gs = args.group_size
    copies = max(1, math.ceil(args.rotate_mb * 2**20 / (N * K // 2)))
    bits_used = sorted({KERNEL_BITS[k] for k in args.kernels})
    # Same copy count for every bit width keeps the rotation simple; int2
    # copies are half the size, still well past cache.
    w_packed = {
        b: [_random_bytes((N, K * b // 8)) for _ in range(copies)] for b in bits_used
    }
    w_scale = [mx.random.uniform(0.005, 0.02, (N,)) for _ in range(copies)]
    mlx_w = [_random_bytes((N, K // 2)).view(mx.uint32) for _ in range(copies)]
    mlx_scales = [
        mx.random.uniform(0.005, 0.02, (N, K // gs)).astype(dtype)
        for _ in range(copies)
    ]
    mlx_biases = [
        mx.random.uniform(-0.1, 0.0, (N, K // gs)).astype(dtype) for _ in range(copies)
    ]
    # Affine scale / bias planes [G, N] per group size, and w_sum (random:
    # the check uses the op's formula, which does not need true row sums).
    qdtype = DTYPES[args.affine_qparam_dtype]
    affine_planes: Dict[Optional[int], List[Tuple[mx.array, mx.array]]] = {}
    for kernel in args.kernels:
        if kernel in AFFINE_KERNELS:
            g = affine_spec(kernel)[1]
            groups = K // (g or K)
            affine_planes.setdefault(
                g,
                [
                    (
                        mx.random.uniform(0.005, 0.02, (groups, N)).astype(qdtype),
                        mx.random.uniform(-0.1, 0.0, (groups, N)).astype(qdtype),
                    )
                    for _ in range(copies)
                ],
            )
    w_sum = [mx.random.normal((N,)) for _ in range(copies)]
    mx.eval(
        list(w_packed.values()),
        w_scale,
        mlx_w,
        mlx_scales,
        mlx_biases,
        list(affine_planes.values()),
        w_sum,
    )
    quant_opts = {"quant_threads": args.quant_threads}

    def timed(
        fn: Callable[[mx.array, int], mx.array], x0: mx.array, iters: int
    ) -> float:
        return time_per_call(
            fn, x0, args.warmup, iters, args.time_budget_ms / 1e3, args.min_iters
        )

    def call(kernel: str, xi: mx.array, i: int, tile: Tile) -> mx.array:
        bits = KERNEL_BITS[kernel]
        wp, ws = w_packed[bits][i % copies], w_scale[i % copies]
        if kernel in AFFINE_KERNELS:
            act, g = affine_spec(kernel)
            s, b = affine_planes[g][i % copies]
            return affine_gemm(xi, wp, s, b, w_sum[i % copies], tile, act, g)
        if kernel == "fp_int4":
            return fp_int4_gemm(xi, wp, ws, tile)
        return int8_intx_gemm(xi, wp, ws, tile, bits, **quant_opts)

    def check(kernel: str, x: mx.array, tile: Tile) -> float:
        bits = KERNEL_BITS[kernel]
        wp = w_packed[bits][0]
        if kernel in AFFINE_KERNELS:
            act, g = affine_spec(kernel)
            s, b = affine_planes[g][0]
            return check_affine(x, wp, s, b, w_sum[0], tile, act, g)
        if kernel == "fp_int4":
            return check_fp_int4(x, wp, w_scale[0], tile)
        return check_int8_intx(x, wp, w_scale[0], tile, bits, **quant_opts)

    def mlx_call(xi: mx.array, i: int) -> mx.array:
        return mx.quantized_matmul(
            xi,
            mlx_w[i % copies],
            mlx_scales[i % copies],
            mlx_biases[i % copies],
            transpose=True,
            group_size=gs,
            bits=4,
            mode="affine",
        )

    def select(kernel: str, x: mx.array, M: int) -> Optional[Tuple[Tile, float]]:
        """Sweep + refine; returns (tile, sweep or refine seconds) or None. A
        single config (matvec) is only checked; the official re-run times it."""
        tiles = tile_configs(K, M, args, kernel)
        if len(tiles) == 1:
            err = check(kernel, x, tiles[0])
            if err >= 1e-2:
                print(
                    f"  [{dtype_name}] {name:<8} M={M:<4} {kernel} "
                    f"{config_name(tiles[0])} failed check ({err:.1e})",
                    flush=True,
                )
                return None
            return tiles[0], 0.0
        entries: List[Entry] = []
        for tile in tiles:
            try:
                err = check(kernel, x, tile)
                t = timed(
                    lambda xi, i, tile=tile: call(kernel, xi, i, tile), x, args.iters
                )
            except Exception as e:  # compile or launch failure for this tile
                reason = str(e).strip().splitlines()[0][:100]
                print(f"    skip {kernel} {config_name(tile)}: {reason}", flush=True)
                err, t = None, None
            entries.append((tile, t, err))
            if args.verbose and t is not None:
                print(
                    f"    {kernel:<9} {config_name(tile):<20} {t * 1e6:>10.1f} us  "
                    f"check {err:.1e}",
                    flush=True,
                )
        candidates = refine_candidates(entries, args.refine_top)
        if len(candidates) == 1:
            return candidates[0], next(e[1] for e in entries if e[0] == candidates[0])
        if not candidates:
            print(
                f"  [{dtype_name}] {name:<8} M={M:<4} no {kernel} config passed",
                flush=True,
            )
            return None
        refined = {
            tile: timed(
                lambda xi, i, tile=tile: call(kernel, xi, i, tile),
                x,
                args.refine_iters,
            )
            for tile in candidates
        }
        if args.verbose:
            ranked = ", ".join(
                f"{config_name(c)} {t * 1e6:.1f}"
                for c, t in sorted(refined.items(), key=lambda kv: kv[1])
            )
            print(f"    {kernel} refine (us): {ranked}", flush=True)
        tile = min(refined, key=refined.get)
        return tile, refined[tile]

    results: Dict[int, dict] = {}
    for M in ms:
        x = mx.random.normal((M, K)).astype(dtype)
        mx.eval(x)
        winners = {k: select(k, x, M) for k in args.kernels}
        live = {k: w[0] for k, w in winners.items() if w is not None}

        times: Dict[str, List[float]] = {k: [] for k in live}
        t_mlxs: List[float] = []
        for _ in range(args.final_repeats):
            for kernel, tile in live.items():
                times[kernel].append(
                    timed(
                        lambda xi, i, kernel=kernel, tile=tile: call(
                            kernel, xi, i, tile
                        ),
                        x,
                        args.final_iters,
                    )
                )
            t_mlxs.append(timed(mlx_call, x, args.final_iters))
        t_mlx = statistics.median(t_mlxs)
        best = {
            k: (live[k], statistics.median(times[k])) if k in live else None
            for k in args.kernels
        }
        results[M] = {"mlx": t_mlx, "best": best}
        parts = [
            f"{k} {config_name(b[0])} {b[1] * 1e6:.1f} us ({t_mlx / b[1]:.2f}x)"
            for k, b in best.items()
            if b is not None
        ]
        print(
            f"  [{dtype_name}] {name:<8} M={M:<4} mlx {t_mlx * 1e6:.1f} us | "
            + " | ".join(parts),
            flush=True,
        )

    return results


def print_dtype_report(
    dtype_name: str,
    ms: List[int],
    res: Dict[str, Dict[int, dict]],
    gs: int,
    kernels: List[str],
) -> None:
    print()
    print(f"=== {dtype_name}: best config per kernel vs MLX 4-bit affine gs={gs} ===")
    print("(official: refined winners and MLX re-timed alternately, medians)")
    header = f"{'shape':<8} {'N':>7} {'K':>6} {'M':>4} | {'mlx us':>9} {'TOPS':>5}"
    for k in kernels:
        header += f" | {k + ' us':>12} {'TOPS':>5} {'config':<18} {'vs mlx':>6}"
    print(header)
    print("-" * len(header))
    for name, shape_res in res.items():
        N, K, _ = SHAPES[name]
        for M in ms:
            ops = 2 * M * N * K
            t_mlx = shape_res[M]["mlx"]
            row = (
                f"{name:<8} {N:>7} {K:>6} {M:>4} | "
                f"{t_mlx * 1e6:>9.1f} {ops / t_mlx / 1e12:>5.2f}"
            )
            for k in kernels:
                b = shape_res[M]["best"][k]
                if b is None:
                    row += f" | {'failed':>12} {'':>5} {'':<18} {'':>6}"
                    continue
                tile, t = b
                row += (
                    f" | {t * 1e6:>12.1f} {ops / t / 1e12:>5.2f} "
                    f"{config_name(tile):<18} {t_mlx / t:>5.2f}x"
                )
            print(row)

    print()
    counts = ", ".join(f"{n} x{SHAPES[n][2]}" for n in res)
    print(
        f"=== {dtype_name}: estimated Muse Glimmer GEMM time per forward ({counts}) ==="
    )
    header = f"{'M':>4} | {'mlx ms':>8}"
    for k in kernels:
        header += f" | {k + ' ms':>12} {'vs mlx':>6}"
    print(header)
    print("-" * len(header))
    for M in ms:
        t_mlx = sum(SHAPES[n][2] * res[n][M]["mlx"] for n in res)
        row = f"{M:>4} | {t_mlx * 1e3:>8.2f}"
        for k in kernels:
            bests = [res[n][M]["best"][k] for n in res]
            if any(b is None for b in bests):
                row += f" | {'n/a':>12} {'-':>6}"
                continue
            t = sum(SHAPES[n][2] * b[1] for n, b in zip(res, bests))
            row += f" | {t * 1e3:>12.2f} {t_mlx / t:>5.2f}x"
        print(row)


def preset_time(
    res: Dict[str, Dict[int, dict]], M: int, preset: Dict[str, str], act: str
) -> Optional[float]:
    """Seconds per forward of ``preset`` with ``act`` activations, or None if
    a shape or kernel it needs was not benchmarked."""
    total = 0.0
    for group, names in GROUPS.items():
        kernel = preset[group].format(act=act)
        for name in names:
            if name not in res:
                return None
            r = res[name][M]
            if kernel == "mlx":
                t = r["mlx"]
            else:
                best = r["best"].get(kernel)
                if best is None:
                    return None
                t = best[1]
            total += SHAPES[name][2] * t
    return total


def print_presets(
    dtype_name: str, ms: List[int], res: Dict[str, Dict[int, dict]], gs: int
) -> None:
    rows = []
    for name, preset in PRESETS.items():
        acts = ["int8", "fp"] if any("{act}" in k for k in preset.values()) else [""]
        for act in acts:
            times = [preset_time(res, M, preset, act) for M in ms]
            if any(t is not None for t in times):
                rows.append((name, act, times))
    print()
    print(
        f"=== {dtype_name}: presets, estimated Muse Glimmer GEMM time per forward "
        f"(ms; mlx = MLX 4-bit affine gs={gs}, lm_head follows FFN) ==="
    )
    if not rows:
        print("(no preset has every kernel it needs; see PRESETS)")
        return
    t_mlx = [preset_time(res, M, PRESETS["mlx FFN + mlx ATTN"], "") for M in ms]
    header = f"{'preset':<28} {'act':<4}" + "".join(f" | {f'M={M}':>16}" for M in ms)
    print(header)
    print("-" * len(header))
    for name, act, times in rows:
        row = f"{name:<28} {act:<4}"
        for t, tm in zip(times, t_mlx):
            if t is None:
                row += f" | {'n/a':>16}"
            else:
                vs = f" ({tm / t:.2f}x)" if tm else ""
                row += f" | {t * 1e3:>8.2f}{vs:>8}"
        print(row)


def save_results(
    path: str, args: argparse.Namespace, ms: List[int], all_res: Dict[str, dict]
) -> None:
    data = {
        "device": mx.device_info().get("device_name", "?"),
        "group_size": args.group_size,
        "kernels": args.kernels,
        "m": ms,
        "results": {
            dtype: {
                name: {
                    str(M): {
                        "mlx": r["mlx"],
                        "best": {
                            k: None if b is None else [list(b[0]), b[1]]
                            for k, b in r["best"].items()
                        },
                    }
                    for M, r in shape_res.items()
                }
                for name, shape_res in res.items()
            }
            for dtype, res in all_res.items()
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"saved results to {path}")


def load_results(path: str) -> Tuple[dict, Dict[str, dict]]:
    with open(path) as f:
        data = json.load(f)
    all_res = {
        dtype: {
            name: {
                int(M): {
                    "mlx": r["mlx"],
                    "best": {
                        k: None if b is None else (tuple(b[0]), b[1])
                        for k, b in r["best"].items()
                    },
                }
                for M, r in shape_res.items()
            }
            for name, shape_res in res.items()
        }
        for dtype, res in data["results"].items()
    }
    return data, all_res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 256],
        help="Token batch sizes (GEMM M). 1-2 use matvec (decode); 256 is the "
        "Muse Glimmer prefill chunk; larger M is GPU-saturated and scales "
        "linearly.",
    )
    parser.add_argument(
        "--shapes", nargs="+", choices=list(SHAPES), default=list(SHAPES)
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=list(DTYPES),
        default=["fp16"],
        help="Activation dtypes (default fp16; add bf16 to also run it).",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=KERNELS,
        default=list(DEFAULT_KERNELS),
        help="int8_int4 / int8_int2: int8 dynamic activations x int4 / int2; "
        "fp_int4: fp16/bf16 activations x int4; {int8,fp}_affine_{gs64,gs128,"
        "gs256,pc} (opt-in): asymmetric activations x int4 with a scale and "
        "bias per group (pc = per channel). Each is swept and compared against "
        "MLX 4-bit.",
    )
    parser.add_argument(
        "--affine-bn",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Affine kernels: MPP threadgroup tile columns.",
    )
    parser.add_argument(
        "--affine-bias",
        type=int,
        nargs="+",
        choices=[1, 3],
        default=[1, 3],
        help="Affine kernels, grouped weights: bias term as an MPP call (1) or "
        "after the group loop (3).",
    )
    parser.add_argument(
        "--affine-qparam-dtype",
        choices=list(DTYPES),
        default="fp16",
        help="Affine kernels: dtype of the scale / bias planes.",
    )
    parser.add_argument(
        "--bm",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="MPP output tile rows (multiple of 8; capped at max(8, M)).",
    )
    parser.add_argument(
        "--bn",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="MPP output tile columns. BN = 16 never won a full M4 sweep.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Full-K split-K factors (1 = no split); needs K %% (64 * splits) == 0.",
    )
    parser.add_argument(
        "--group-size", type=int, default=32, help="MLX baseline group size."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed runs per config / shape / M before timing (compiles kernels).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Max timed runs per tile config in the sweep.",
    )
    parser.add_argument(
        "--time-budget-ms",
        type=float,
        default=50,
        help="Target GPU time per timing; the iteration count is budget / "
        "(one-call estimate), clamped to [--min-iters, round cap]. Keeps short "
        "kernels precise and long ones (large M) cheap, limiting heat.",
    )
    parser.add_argument(
        "--min-iters",
        type=int,
        default=3,
        help="Minimum timed runs per timing.",
    )
    parser.add_argument(
        "--refine-top",
        type=int,
        default=3,
        help="Fastest sweep configs per lowering (full-K, split-K) re-timed in "
        "the refine round.",
    )
    parser.add_argument(
        "--refine-iters",
        type=int,
        default=20,
        help="Max timed runs per config in the refine round.",
    )
    parser.add_argument(
        "--final-iters",
        type=int,
        default=100,
        help="Max timed runs per official re-timing of the refined winner and " "MLX.",
    )
    parser.add_argument(
        "--final-repeats",
        type=int,
        default=5,
        help="Alternating re-timings of the refined winner and MLX; the medians "
        "are the official numbers.",
    )
    parser.add_argument(
        "--dtype-pause",
        type=float,
        default=60,
        help="Seconds to pause between activation dtypes.",
    )
    parser.add_argument(
        "--shape-pause",
        type=float,
        default=10,
        help="Seconds to pause between shapes within a dtype.",
    )
    parser.add_argument(
        "--rotate-mb",
        type=float,
        default=512,
        help="Rotate weight copies until their total exceeds this size.",
    )
    parser.add_argument(
        "--matvec-max-m",
        type=int,
        default=4,
        help="M up to this value sweeps the matvec kernel (MR in {1, 2, 4} up "
        "to M) and, for fp_int4 / affine, the qmv_wide-style kernel. 0 "
        "disables them.",
    )
    parser.add_argument(
        "--mpp-min-m",
        type=int,
        default=3,
        help="M from this value sweeps MPP tiles (also any M without matvec); "
        "M in [--mpp-min-m, --matvec-max-m] sweeps both and keeps the fastest.",
    )
    parser.add_argument(
        "--wide-kl",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="fp_int4 qmv_wide-style kernel (M <= --matvec-max-m): lanes per "
        "output channel to sweep. Rows per threadgroup sweeps M and ceil(M / 2).",
    )
    parser.add_argument(
        "--quant-threads",
        type=int,
        default=256,
        help="int8_int4: threads per row in the activation quantize kernel "
        "(multiple of 32, at most 1024).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print every tile config."
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Write the official per-shape results (every dtype) to this file.",
    )
    parser.add_argument(
        "--load-json",
        type=str,
        default=None,
        help="Skip benchmarking; print the reports and presets from a "
        "--save-json file.",
    )
    args = parser.parse_args()

    if args.load_json:
        data, all_res = load_results(args.load_json)
        print(f"loaded {args.load_json}: device {data['device']}", flush=True)
        for dtype_name, res in all_res.items():
            print_dtype_report(
                dtype_name, data["m"], res, data["group_size"], data["kernels"]
            )
            print_presets(dtype_name, data["m"], res, data["group_size"])
            print(flush=True)
        return

    ms = sorted(set(args.m))

    info = mx.device_info()
    print(
        f"device: {info.get('device_name', '?')}  mlx: {mx.__version__}  "
        f"M: {ms}  dtypes: {args.dtypes}  kernels: {args.kernels}  "
        f"mlx group_size: {args.group_size}  "
        f"warmup: {args.warmup}  time budget: {args.time_budget_ms:g} ms "
        f"(min iters {args.min_iters})  sweep iters: {args.iters}  "
        f"refine: top {args.refine_top}/lowering x {args.refine_iters} iters  "
        f"final iters: {args.final_iters}  final repeats: {args.final_repeats}  "
        f"quant threads: {args.quant_threads}  "
        f"bm: {args.bm}  bn: {args.bn}  splits: {args.splits}  "
        f"matvec max M: {args.matvec_max_m}  mpp min M: {args.mpp_min_m}  "
        f"wide kl: {args.wide_kl}  "
        f"affine bn: {args.affine_bn}  affine bias: {args.affine_bias}  "
        f"affine qparams: {args.affine_qparam_dtype}",
        flush=True,
    )

    all_res = {}
    for i, dtype_name in enumerate(args.dtypes):
        if i > 0 and args.dtype_pause > 0:
            print(f"pausing {args.dtype_pause:g}s before {dtype_name}...", flush=True)
            time.sleep(args.dtype_pause)
        res = {}
        for j, name in enumerate(args.shapes):
            if j > 0 and args.shape_pause > 0:
                print(f"pausing {args.shape_pause:g}s before {name}...", flush=True)
                time.sleep(args.shape_pause)
            res[name] = bench_shape(name, dtype_name, ms, args)
            mx.clear_cache()
        all_res[dtype_name] = res
        print_dtype_report(dtype_name, ms, res, args.group_size, args.kernels)
        print_presets(dtype_name, ms, res, args.group_size)
        print(flush=True)
    if args.save_json:
        save_results(args.save_json, args, ms, all_res)


if __name__ == "__main__":
    main()
