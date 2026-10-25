# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in the root LICENSE.
"""Experimental uint32 reads of unchanged ET INT4 bytes, with BF16 GEMM.

GemLite's packed-word Triton decode is the scheduling/dataflow reference.
No GemLite weight conversion is used: each ET uint32 view holds its original
eight consecutive 4-bit values. Quantization codes and metadata are unchanged.
Both independently verified ET decode arithmetic variants remain explicit.
This module registers no runtime/model dispatch and allocates no permanent
dense weights. Materialization is a diagnostic oracle check only.
"""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from public_grouped_w4a16 import _validate, _reduce


@dataclass(frozen=True)
class Config:
    bm: int = 64
    bn: int = 64
    bk: int = 128
    split: int = 1
    warps: int = 4
    stages: int = 2
    group_m: int = 8


@triton.jit
def _weight(Q32, SC, SS, Z, ZS, ns, start, N: tl.constexpr, K: tl.constexpr,
            BN: tl.constexpr, BK: tl.constexpr, ZERO_FMA: tl.constexpr):
    words = tl.load(Q32 + ns[:, None] * (K // 8) + start // 8 + tl.arange(0, BK // 8)[None, :],
                    mask=ns[:, None] < N, other=0).to(tl.uint32)
    shifts = tl.arange(0, 8) * 4
    codes = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 15, (BN, BK)).to(tl.bfloat16)
    groups = start // 32 + tl.arange(0, BK // 32)
    sc = tl.load(SC + ns[:, None] * (K // 32) + groups[None, :], mask=ns[:, None] < N, other=0).to(tl.bfloat16)
    z = tl.load(Z + ns[:, None] * (K // 32) + groups[None, :], mask=ns[:, None] < N, other=0).to(tl.bfloat16)
    ss = tl.load(SS + ns * (K // 256) + start // 256, mask=ns < N, other=0).to(tl.bfloat16)
    zs = tl.load(ZS + ns * (K // 256) + start // 256, mask=ns < N, other=0).to(tl.bfloat16)
    sg = (sc.to(tl.float32) * ss[:, None].to(tl.float32)).to(tl.bfloat16)
    se = tl.reshape(tl.broadcast_to(sg[:, :, None], (BN, BK // 32, 32)), (BN, BK))
    if ZERO_FMA:
        ze = tl.reshape(tl.broadcast_to(z[:, :, None], (BN, BK // 32, 32)), (BN, BK))
        shifted = tl.fma(-ze.to(tl.float32), zs[:, None].to(tl.float32), codes.to(tl.float32)).to(tl.bfloat16)
    else:
        zg = (z.to(tl.float32) * zs[:, None].to(tl.float32)).to(tl.bfloat16)
        ze = tl.reshape(tl.broadcast_to(zg[:, :, None], (BN, BK // 32, 32)), (BN, BK))
        shifted = (codes.to(tl.float32) - ze.to(tl.float32)).to(tl.bfloat16)
    return (shifted.to(tl.float32) * se.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _gemm(X, Q32, SC, SS, Z, ZS, OUT, M, N: tl.constexpr, K: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          SPLIT: tl.constexpr, GROUP: tl.constexpr, STAGES: tl.constexpr,
          ZERO_FMA: tl.constexpr):
    pid = tl.program_id(0); split = tl.program_id(1)
    nm = tl.cdiv(M, BM); nn: tl.constexpr = tl.cdiv(N, BN)
    group = pid // (GROUP * nn)
    first_m = group * GROUP; size_m = tl.minimum(nm - first_m, GROUP)
    pm = first_m + pid % size_m; pn = (pid % (GROUP * nn)) // size_m
    ms = pm * BM + tl.arange(0, BM); ns = pn * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    tiles: tl.constexpr = K // BK
    per: tl.constexpr = tiles // SPLIT; extra: tl.constexpr = tiles % SPLIT
    first = split * per + tl.minimum(split, extra); count = per + (split < extra)
    acc = tl.zeros((BM, BN), tl.float32)
    for tile in tl.range(first, first + count, num_stages=STAGES):
        start = tile * BK
        a = tl.load(X + ms[:, None] * K + start + ks[None, :], mask=ms[:, None] < M, other=0)
        b = _weight(Q32, SC, SS, Z, ZS, ns, start, N, K, BN, BK, ZERO_FMA)
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
    offset = ms[:, None] * N + ns[None, :]
    if SPLIT == 1:
        tl.store(OUT + offset, acc.to(tl.bfloat16), mask=(ms[:, None] < M) & (ns[None, :] < N))
    else:
        tl.store(OUT + split * M * N + offset, acc, mask=(ms[:, None] < M) & (ns[None, :] < N))


@triton.jit
def _materialize(Q32, SC, SS, Z, ZS, OUT, N: tl.constexpr, K: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, ZERO_FMA: tl.constexpr):
    ns = tl.program_id(0) * BN + tl.arange(0, BN); start = tl.program_id(1) * BK
    w = _weight(Q32, SC, SS, Z, ZS, ns, start, N, K, BN, BK, ZERO_FMA)
    tl.store(OUT + ns[:, None] * K + start + tl.arange(0, BK)[None, :], w, mask=ns[:, None] < N)


def launch(x, weights, *, config=Config(), arithmetic, out=None, partial=None):
    m, n, k = _validate(x, weights, arithmetic)
    assert config.bm in [32, 64, 128] and config.bn in [64, 128] and config.bk in [32, 64, 128, 256]
    assert config.split in [1, 4, 8] and config.warps in [4, 8] and config.stages in [1, 2, 3]
    if out is None: out = torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
    assert tuple(out.shape) == (m,n) and out.dtype == torch.bfloat16 and out.is_contiguous() and out.device == x.device
    if config.split > 1:
        if partial is None: partial = torch.empty((config.split,m,n),device=x.device,dtype=torch.float32)
        assert tuple(partial.shape) == (config.split,m,n) and partial.dtype == torch.float32 and partial.is_contiguous() and partial.device == x.device
    q, *metadata = weights
    q32 = q.view(torch.int32)
    assert q32.data_ptr() == q.data_ptr() and tuple(q32.shape) == (n,k//8)
    target = out if config.split == 1 else partial
    _gemm[(triton.cdiv(m,config.bm)*triton.cdiv(n,config.bn),config.split)](
        x,q32,*metadata,target,m,N=n,K=k,BM=config.bm,BN=config.bn,BK=config.bk,
        SPLIT=config.split,GROUP=config.group_m,STAGES=config.stages,
        ZERO_FMA=arithmetic=='native-zero-fma-bf16',num_warps=config.warps,
        num_stages=config.stages,enable_fp_fusion=False)
    if config.split > 1:
        _reduce[(triton.cdiv(m*n,256),)](partial,out,m*n,SPLIT=config.split,BLOCK=256,num_warps=4,enable_fp_fusion=False)
    return out


def materialize(weights, *, arithmetic, bk=64):
    q = weights[0]; n, k = q.shape[0], 2*q.shape[1]
    dummy = torch.empty((1,k),dtype=torch.bfloat16,device=q.device)
    _validate(dummy,weights,arithmetic)
    assert bk in [32,64,128,256]
    out = torch.empty((n,k),dtype=torch.bfloat16,device=q.device)
    _materialize[(triton.cdiv(n,32),k//bk)](q.view(torch.int32),*weights[1:],out,
        N=n,K=k,BN=32,BK=bk,ZERO_FMA=arithmetic=='native-zero-fma-bf16',
        num_warps=4,num_stages=1,enable_fp_fusion=False)
    return out


def configs(m):
    return [Config(32,64,128,stages=2), Config(32,64,256,stages=2),
            Config(64,64,128,stages=2), Config(64,64,128,stages=3),
            Config(64,64,256,stages=1), Config(64,128,128,stages=2),
            Config(64,128,128,stages=1), Config(128,64,128,warps=8,stages=1)]
