#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark ``mlx::int8_int4_gemm`` (dynamic per-token int8 activations x
per-channel int4 weights, MPP INT32 accumulation) against MLX
``quantized_matmul`` (4-bit affine, group size 32) on every Muse Glimmer GEMM
shape, and estimate total Muse Glimmer GEMM time per forward pass.

For each activation dtype, shape, and M (token batch size), every int8_int4
tile config (BM x BN x BK) is checked against a reference and timed. BK picks
the lowering:
    * ``BK > 0``: one fused kernel that steps over K in BK tiles (BK a
      multiple of 32 dividing K).
    * ``BK = -1``: quantize kernel + full-K MPP ``int8 x int4b`` GEMM; MPP
      handles the whole K itself.
The fastest config is reported against MLX. The Muse Glimmer estimate is
``sum over shapes of (best time x times the shape occurs in the model)``.

Defaults cover speculative decoding / prefill (M = 4..512), fp16 and bf16, all
shapes, and the full tile sweep. Just run::

    python -m executorch.backends.mlx.custom_kernel_ops.test.bench_int8_int4_gemm

and send back the output. ``--verbose`` prints every tile config.

Timing: each config / shape / M runs ``--iters`` times, built lazily and timed
as a single ``mx.eval``; the reported time is the average per call. Each kernel
runs once untimed first (the accuracy check doubles as this warmup).
Calls rotate across enough weight copies to exceed ``--rotate-mb`` so weights
are not served from cache. int8_int4 times include activation quantization.

Requires MLX and an MPP-capable GPU (Metal 4).
"""

import argparse
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx
from executorch.backends.mlx.custom_kernel_ops import int8_int4_gemm as op

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
FULL_K = -1
_THREADGROUP_MEMORY = 32768

# (BM, BN, BK); BK == FULL_K selects the full-K lowering.
Tile = Tuple[int, int, int]
# (tile, seconds or None on failure, check error or None)
Entry = Tuple[Tile, Optional[float], Optional[float]]

_staged_kernel = mx.fast.metal_kernel(
    name="int8_int4_gemm",
    input_names=["x", "w_packed", "w_scale"],
    output_names=["out"],
    source=op._SOURCE,
    header=op._HEADER,
)
_quant_kernel = mx.fast.metal_kernel(
    name="int8_int4_gemm_quantize",
    input_names=["x"],
    output_names=["xq", "x_scale"],
    source=op._QUANT_SOURCE,
    header=op._HEADER,
)
_full_k_kernel = mx.fast.metal_kernel(
    name="int8_int4_gemm_full_k",
    input_names=["xq", "x_scale", "w_packed", "w_scale"],
    output_names=["out"],
    source=op._FULL_K_SOURCE,
    header=op._HEADER,
)


def config_name(tile: Tile) -> str:
    bm, bn, bk = tile
    if bk == FULL_K:
        return f"full_k {bm}x{bn}"
    return f"staged {bm}x{bn}x{bk}"


def int8_int4_gemm(
    x: mx.array, w_packed: mx.array, w_scale: mx.array, tile: Tile
) -> mx.array:
    M, K = x.shape
    N = w_packed.shape[0]
    bm, bn, bk = tile
    launch = dict(
        grid=(math.ceil(N / bn) * op._THREADS, math.ceil(M / bm), 1),
        threadgroup=(op._THREADS, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )
    template = [("T", x.dtype), ("N", N), ("K", K), ("BM", bm), ("BN", bn)]
    if bk != FULL_K:
        return _staged_kernel(
            inputs=[x, w_packed, w_scale], template=template + [("BK", bk)], **launch
        )[0]
    xq, x_scale = _quant_kernel(
        inputs=[x],
        template=[("T", x.dtype), ("K", K)],
        grid=(32, M, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(M, K), (M,)],
        output_dtypes=[mx.int8, mx.float32],
    )
    return _full_k_kernel(
        inputs=[xq, x_scale, w_packed, w_scale], template=template, **launch
    )[0]


def check_int8_int4(
    x: mx.array, w_packed: mx.array, w_scale: mx.array, tile: Tile, cols: int = 256
) -> float:
    """Max error of the first ``cols`` outputs, relative to max |reference|."""
    out = int8_int4_gemm(x, w_packed, w_scale, tile)[:, :cols]
    xf = x.astype(mx.float32)
    x_scale = mx.maximum(mx.max(mx.abs(xf), axis=-1, keepdims=True) / 127, op._EPS)
    xq = mx.clip(mx.round(xf * mx.reciprocal(x_scale)), -128, 127)
    w = w_packed[:cols].astype(mx.int32)
    nibbles = mx.stack([w & 0xF, w >> 4], axis=-1).reshape(cols, -1)
    wq = mx.where(nibbles >= 8, nibbles - 16, nibbles).astype(mx.float32)
    ref = (xq @ wq.T) * x_scale * w_scale[:cols].astype(mx.float32)
    err = mx.max(mx.abs(out.astype(mx.float32) - ref)) / mx.max(mx.abs(ref))
    return err.item()


def time_per_call(fn: Callable[[int], mx.array], iters: int) -> float:
    """Average seconds per call over ``iters`` runs; ``fn(i)`` must build (not
    evaluate) run ``i``."""
    outs = [fn(i) for i in range(iters)]
    start = time.perf_counter()
    mx.eval(outs)
    return (time.perf_counter() - start) / iters


def tile_configs(K: int, args: argparse.Namespace) -> List[Tile]:
    """Valid tiles; BK > 0 needs BK % 32 == 0, K % BK == 0, and tiles that fit
    in threadgroup memory."""
    tiles: List[Tile] = []
    for bm in args.bm:
        for bn in args.bn:
            for bk in args.bk:
                if bk == FULL_K:
                    tiles.append((bm, bn, bk))
                    continue
                tg_bytes = (bm + bn) * bk + 8 * bm
                if bk % 32 == 0 and K % bk == 0 and tg_bytes <= _THREADGROUP_MEMORY:
                    tiles.append((bm, bn, bk))
    return tiles


def best(entries: List[Entry]) -> Optional[Entry]:
    """Fastest config that ran and passed the accuracy check."""
    ok = [e for e in entries if e[1] is not None and e[2] is not None and e[2] < 1e-2]
    return min(ok, key=lambda e: e[1]) if ok else None


def _random_bytes(shape: Tuple[int, ...]) -> mx.array:
    return mx.random.randint(0, 256, shape).astype(mx.uint8)


def bench_shape(
    name: str, dtype_name: str, ms: List[int], args: argparse.Namespace
) -> Dict[int, dict]:
    """Returns {M: {"mlx": seconds, "int8": [Entry]}}."""
    N, K, _ = SHAPES[name]
    dtype = DTYPES[dtype_name]
    gs = args.group_size
    copies = max(1, math.ceil(args.rotate_mb * 2**20 / (N * K // 2)))
    w_packed = [_random_bytes((N, K // 2)) for _ in range(copies)]
    w_scale = [mx.random.uniform(0.005, 0.02, (N,)) for _ in range(copies)]
    mlx_w = [_random_bytes((N, K // 2)).view(mx.uint32) for _ in range(copies)]
    mlx_scales = [
        mx.random.uniform(0.005, 0.02, (N, K // gs)).astype(dtype)
        for _ in range(copies)
    ]
    mlx_biases = [
        mx.random.uniform(-0.1, 0.0, (N, K // gs)).astype(dtype) for _ in range(copies)
    ]
    mx.eval(w_packed, w_scale, mlx_w, mlx_scales, mlx_biases)

    def timed(fn: Callable[[int], mx.array]) -> float:
        return time_per_call(fn, args.iters)

    results: Dict[int, dict] = {}
    for M in ms:
        x = mx.random.normal((M, K)).astype(dtype)
        mx.eval(x)
        entries: List[Entry] = []
        for tile in tile_configs(K, args):
            try:
                err = check_int8_int4(x, w_packed[0], w_scale[0], tile)
                t = timed(
                    lambda i: int8_int4_gemm(
                        x, w_packed[i % copies], w_scale[i % copies], tile
                    )
                )
            except Exception as e:  # compile or launch failure for this tile
                reason = str(e).strip().splitlines()[0][:100]
                print(f"    skip {config_name(tile)}: {reason}", flush=True)
                err, t = None, None
            entries.append((tile, t, err))
            if args.verbose and t is not None:
                print(
                    f"    {config_name(tile):<20} {t * 1e6:>10.1f} us  "
                    f"check {err:.1e}",
                    flush=True,
                )

        def mlx_call(i: int) -> mx.array:
            return mx.quantized_matmul(
                x,
                mlx_w[i % copies],
                mlx_scales[i % copies],
                mlx_biases[i % copies],
                transpose=True,
                group_size=gs,
                bits=4,
                mode="affine",
            )

        mx.eval(mlx_call(0))
        t_mlx = timed(mlx_call)
        results[M] = {"mlx": t_mlx, "int8": entries}

        b = best(entries)
        summary = (
            f"best {config_name(b[0])} {b[1] * 1e6:.1f} us, "
            f"{t_mlx / b[1]:.2f}x vs mlx"
            if b is not None
            else "no int8_int4 config passed"
        )
        print(
            f"  [{dtype_name}] {name:<8} M={M:<4} mlx {t_mlx * 1e6:>9.1f} us | {summary}",
            flush=True,
        )

    del w_packed, w_scale, mlx_w, mlx_scales, mlx_biases
    mx.clear_cache()
    return results


def print_dtype_report(
    dtype_name: str, ms: List[int], res: Dict[str, Dict[int, dict]], gs: int
) -> None:
    print()
    print(f"=== {dtype_name}: best int8_int4 config vs MLX 4-bit affine gs={gs} ===")
    header = (
        f"{'shape':<8} {'N':>7} {'K':>6} {'M':>4} | {'int8_int4 us':>12} "
        f"{'TOPS':>6} {'config':<20} | {'mlx us':>10} {'TOPS':>6} | {'speedup':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, shape_res in res.items():
        N, K, _ = SHAPES[name]
        for M in ms:
            ops = 2 * M * N * K
            t_mlx = shape_res[M]["mlx"]
            b = best(shape_res[M]["int8"])
            mlx_cols = f"{t_mlx * 1e6:>10.1f} {ops / t_mlx / 1e12:>6.2f}"
            if b is None:
                int8_cols = f"{'failed':>12} {'':>6} {'':<20}"
                speedup = f"{'-':>7}"
            else:
                tile, t, _ = b
                int8_cols = (
                    f"{t * 1e6:>12.1f} {ops / t / 1e12:>6.2f} {config_name(tile):<20}"
                )
                speedup = f"{t_mlx / t:>6.2f}x"
            print(
                f"{name:<8} {N:>7} {K:>6} {M:>4} | {int8_cols} | {mlx_cols} | {speedup}"
            )

    print()
    counts = ", ".join(f"{n} x{SHAPES[n][2]}" for n in res)
    print(
        f"=== {dtype_name}: estimated Muse Glimmer GEMM time per forward ({counts}) ==="
    )
    header = f"{'M':>4} | {'int8_int4 ms':>12} | {'mlx ms':>10} | {'speedup':>7}"
    print(header)
    print("-" * len(header))
    for M in ms:
        bests = {n: best(res[n][M]["int8"]) for n in res}
        t_mlx = sum(SHAPES[n][2] * res[n][M]["mlx"] for n in res)
        if any(b is None for b in bests.values()):
            print(f"{M:>4} | {'n/a':>12} | {t_mlx * 1e3:>10.2f} | {'-':>7}")
            continue
        t_int8 = sum(SHAPES[n][2] * bests[n][1] for n in res)
        print(
            f"{M:>4} | {t_int8 * 1e3:>12.2f} | {t_mlx * 1e3:>10.2f} | "
            f"{t_mlx / t_int8:>6.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[4, 8, 16, 256, 512],
        help="Token batch sizes (GEMM M).",
    )
    parser.add_argument(
        "--shapes", nargs="+", choices=list(SHAPES), default=list(SHAPES)
    )
    parser.add_argument(
        "--dtypes", nargs="+", choices=list(DTYPES), default=list(DTYPES)
    )
    parser.add_argument("--bm", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--bn", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument(
        "--bk",
        type=int,
        nargs="+",
        default=[32, 64, 128, FULL_K],
        help=f"K step (multiple of 32 dividing K), or {FULL_K} for full K.",
    )
    parser.add_argument(
        "--group-size", type=int, default=32, help="MLX baseline group size."
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="Timed runs per config / shape / M."
    )
    parser.add_argument(
        "--rotate-mb",
        type=float,
        default=512,
        help="Rotate weight copies until their total exceeds this size.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print every tile config."
    )
    args = parser.parse_args()
    ms = sorted(set(args.m))

    info = mx.device_info()
    print(
        f"device: {info.get('device_name', '?')}  mlx: {mx.__version__}  "
        f"M: {ms}  dtypes: {args.dtypes}  mlx group_size: {args.group_size}  "
        f"iters: {args.iters}",
        flush=True,
    )

    for dtype_name in args.dtypes:
        res = {name: bench_shape(name, dtype_name, ms, args) for name in args.shapes}
        print_dtype_report(dtype_name, ms, res, args.group_size)
        print(flush=True)


if __name__ == "__main__":
    main()
