# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in the root LICENSE.
"""Experimental BF16 dense kernels for a staged-dequant prefill comparison.

Grouped tile ordering and persistent scheduling follow Triton's official
09-persistent-matmul tutorial, https://github.com/triton-lang/triton (MIT).
This independent adaptation uses ET's [N,K] dense weight layout and explicitly
BF16 outputs/FP32 accumulation. It does not register or change model dispatch.
No TMA, warp specialization, activation quantization or SM90-only primitives.
"""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class Config:
    bm: int = 128
    bn: int = 128
    bk: int = 32
    warps: int = 4
    stages: int = 3
    group_m: int = 8
    persistent: bool = False


@triton.jit
def _pid(tile, nm, nn, GROUP: tl.constexpr):
    group = tile // (GROUP * nn)
    first_m = group * GROUP
    group_m = tl.minimum(nm - first_m, GROUP)
    return first_m + tile % group_m, (tile % (GROUP * nn)) // group_m


@triton.jit
def _dense(X, W, OUT, M, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           GROUP: tl.constexpr, STRIDE: tl.constexpr, STAGES: tl.constexpr):
    start = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn: tl.constexpr = tl.cdiv(N, BN)
    total = nm * nn
    # Distinct epilogue counter follows the tutorial's Blackwell pipelining
    # workaround. For a regular grid STRIDE is total, so there is one tile.
    tile_c = start - STRIDE
    for tile in tl.range(start, total, STRIDE, flatten=True):
        pm, pn = _pid(tile, nm, nn, GROUP)
        ms = pm * BM + tl.arange(0, BM)
        ns = pn * BN + tl.arange(0, BN)
        km = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in tl.range(0, K // BK, num_stages=STAGES):
            ks = ki * BK + km
            a = tl.load(X + ms[:, None] * K + ks[None, :],
                        mask=ms[:, None] < M, other=0)
            b = tl.load(W + ns[None, :] * K + ks[:, None],
                        mask=ns[None, :] < N, other=0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        tile_c += STRIDE
        cm, cn = _pid(tile_c, nm, nn, GROUP)
        rows = cm * BM + tl.arange(0, BM)
        cols = cn * BN + tl.arange(0, BN)
        tl.store(OUT + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16),
                 mask=(rows[:, None] < M) & (cols[None, :] < N))


def launch(x, weight, out, config):
    assert x.ndim == weight.ndim == out.ndim == 2
    assert x.is_cuda and x.device == weight.device == out.device
    assert x.dtype == weight.dtype == out.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous() and out.is_contiguous()
    m, k = x.shape
    n, wk = weight.shape
    assert k == wk and k % config.bk == 0 and out.shape == (m, n)
    assert config.bm in [64, 128] and config.bn in [64, 128]
    assert config.bk in [32, 64, 128] and config.warps in [4, 8]
    assert config.stages in [1, 2, 3, 4] and config.group_m in [1, 8]
    total = triton.cdiv(m, config.bm) * triton.cdiv(n, config.bn)
    stride = min(total, torch.cuda.get_device_properties(x.device).multi_processor_count) if config.persistent else total
    _dense[(stride,)](x, weight, out, m, N=n, K=k, BM=config.bm,
        BN=config.bn, BK=config.bk, GROUP=config.group_m, STRIDE=stride,
        STAGES=config.stages, num_warps=config.warps,
        num_stages=config.stages, enable_fp_fusion=False)
    return out


def configs(resource_followup=False):
    if resource_followup:
        # Specific first-screen configurations exceed SM120's 101376-byte
        # shared-memory limit. Reduce pipeline stages without changing math.
        shapes = [(128, 128, 128, 8, 1), (128, 128, 128, 8, 2),
                  (128, 128, 64, 8, 2), (128, 64, 128, 4, 2),
                  (128, 64, 128, 4, 1), (64, 128, 128, 4, 2),
                  (128, 128, 32, 4, 3)]
        return [Config(bm, bn, bk, warps, stages, 8, persistent)
                for bm, bn, bk, warps, stages in shapes for persistent in [False, True]]
    shapes = [(128, 128, 32, 4, 3), (128, 128, 64, 4, 3),
              (128, 128, 128, 8, 3), (128, 128, 64, 8, 4),
              (64, 128, 64, 4, 3), (64, 128, 128, 4, 3),
              (128, 64, 64, 4, 3), (128, 64, 128, 4, 3),
              (64, 64, 128, 4, 3), (128, 128, 128, 8, 4)]
    return [Config(bm, bn, bk, warps, stages, 8, persistent)
            for bm, bn, bk, warps, stages in shapes for persistent in [False, True]]
