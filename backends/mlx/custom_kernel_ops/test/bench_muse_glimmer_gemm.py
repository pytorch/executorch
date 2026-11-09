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
    * M <= ``--matvec-max-m`` (4): only the matvec kernel, which streams packed
      weights like MLX's qmv. Rows per threadgroup MR in {1, 2, 4} up to M are
      tried; MR < M launches ceil(M / MR) threadgroup rows (MR = 3 compiles
      badly and is skipped). fp_int4 also sweeps the qmv_wide-style kernel:
      ``--wide-kl`` lanes per channel, M or ceil(M / 2) rows per threadgroup.
    * larger M: only MPP tiles (BM x BN, BM capped at max(8, M) since larger
      tiles only add padding rows, and BM >= 32 for M >= 128). Full-K
      (``--splits`` 1) runs MPP over all
      of K; split-K (S > 1) gives each of S threadgroups per output tile K / S
      and a reduce kernel adds the partials and applies the scales.
int8_int4 / int8_int2 quantize activations first (included in their times).
Tile selection has two rounds: every config is timed with ``--iters``, then
the ``--refine-top`` fastest configs of each lowering (full-K, split-K) are
re-timed with ``--refine-iters`` and the fastest is kept. Each kernel's winner
is then re-timed against MLX: they alternate ``--final-repeats`` times and the
median of each is the reported (official) number, so all are measured under
the same GPU conditions. The Muse Glimmer estimate is ``sum over shapes of
(official time x times the shape occurs in the model)``.

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
import math
import statistics
import time
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx
from executorch.backends.mlx.custom_kernel_ops import (
    fp_int4_gemm as fp4,
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
KERNELS = ("int8_int4", "int8_int2", "fp_int4")
# Weight bits per kernel.
KERNEL_BITS = {"int8_int4": 4, "int8_int2": 2, "fp_int4": 4}
MATVEC = 0

# (BM, BN, splits): an MPP tile, split over K when splits > 1. BN == MATVEC
# selects the matvec kernel with BM activation rows per threadgroup; BN < 0
# (fp_int4 only) the qmv_wide-style kernel with BM rows per threadgroup and -BN
# lanes per output channel.
Tile = Tuple[int, int, int]
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


def config_name(tile: Tile) -> str:
    bm, bn, splits = tile
    if bn == MATVEC:
        return f"matvec mr{bm}"
    if bn < 0:
        return f"wide nv{bm} kl{-bn}"
    if splits > 1:
        return f"split_k{splits} {bm}x{bn}"
    return f"full_k {bm}x{bn}"


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


def tile_configs(
    K: int, M: int, args: argparse.Namespace, kernel: str = ""
) -> List[Tile]:
    """Matvec only for M <= --matvec-max-m (needs K % 512 == 0), plus the
    qmv_wide-style kernel for fp_int4; otherwise MPP tiles only. MPP needs
    BM % 8 == 0, BM > max(8, M) only adds padding rows, BM < 32 never won at
    M >= 128, and split-K needs K % (64 * splits) == 0."""
    if M <= args.matvec_max_m and K % op._MATVEC_BLOCK == 0:
        # MR < M launches ceil(M / MR) threadgroup rows that share weights
        # through cache, like MLX's qmv (one row per threadgroup). MR = 3 was
        # 2-5x slower than MR = 2 or 4 on every Muse Glimmer shape.
        tiles = [(mr, MATVEC, 1) for mr in (1, 2, 4) if mr <= M]
        if kernel == "fp_int4":
            tiles += [
                (nv, -kl, 1)
                for nv in sorted({M, (M + 1) // 2})
                for kl in args.wide_kl
                if K % (16 * kl) == 0
            ]
        return tiles
    min_bm = 32 if M >= 128 else 8
    return [
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
    mx.eval(list(w_packed.values()), w_scale, mlx_w, mlx_scales, mlx_biases)
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
        if kernel == "fp_int4":
            return fp_int4_gemm(xi, wp, ws, tile)
        return int8_intx_gemm(xi, wp, ws, tile, bits, **quant_opts)

    def check(kernel: str, x: mx.array, tile: Tile) -> float:
        bits = KERNEL_BITS[kernel]
        wp = w_packed[bits][0]
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
        default=list(KERNELS),
        help="int8_int4 / int8_int2: int8 dynamic activations x int4 / int2; "
        "fp_int4: fp16/bf16 activations x int4. Each is swept and compared "
        "against MLX 4-bit.",
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
        help="M up to this value sweeps only the matvec kernel (MR in {1, 2, 4} "
        "up to M); larger M sweeps only MPP tiles. 0 disables matvec.",
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
    args = parser.parse_args()
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
        f"matvec max M: {args.matvec_max_m}  wide kl: {args.wide_kl}",
        flush=True,
    )

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
        print_dtype_report(dtype_name, ms, res, args.group_size, args.kernels)
        print(flush=True)


if __name__ == "__main__":
    main()
