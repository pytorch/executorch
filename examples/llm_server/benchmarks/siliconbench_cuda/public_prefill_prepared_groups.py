"""Experimental A16 GEMM with temporary exact BF16 group metadata.

Reference: GemLite f649e80e3014bc7ca8195573683c2c440f3bd94e (Apache-2.0),
https://github.com/mobiusml/gemlite. Prepare the two-level ET scale and zero
once per forward, then reconstruct weights inside each GEMM tile. Original
INT4 bytes remain inputs; no dense weight or persistent extra model copy.
This module registers no model route and changes no SM80 dispatch.
"""
import torch
import triton
import triton.language as tl
from public_grouped_w4a16 import Config, _validate


@triton.jit
def _prepare(SC, SS, Z, ZS, SG, ZG, SIZE: tl.constexpr,
             GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // GROUPS
    group = i % GROUPS
    step_i = row * (GROUPS // 8) + group // 8
    sc = tl.load(SC + i, i < SIZE, other=0).to(tl.bfloat16)
    z = tl.load(Z + i, i < SIZE, other=0).to(tl.bfloat16)
    ss = tl.load(SS + step_i, i < SIZE, other=0).to(tl.bfloat16)
    zs = tl.load(ZS + step_i, i < SIZE, other=0).to(tl.bfloat16)
    sg = (sc.to(tl.float32) * ss.to(tl.float32)).to(tl.bfloat16)
    zg = (z.to(tl.float32) * zs.to(tl.float32)).to(tl.bfloat16)
    tl.store(SG + i, sg, i < SIZE)
    tl.store(ZG + i, zg, i < SIZE)


@triton.jit
def _tile(Q, SG, ZG, ns, k, N: tl.constexpr, K: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr):
    packed_k = k // 2 + tl.arange(0, BK // 2)
    q = tl.load(Q + ns[:, None] * (K // 2) + packed_k[None, :],
                ns[:, None] < N, other=0).to(tl.uint32)
    code = tl.interleave(q & 15, (q >> 4) & 15).to(tl.bfloat16)
    groups = k // 32 + tl.arange(0, BK // 32)
    sg = tl.load(SG + ns[:, None] * (K // 32) + groups[None, :],
                 ns[:, None] < N, other=0)
    zg = tl.load(ZG + ns[:, None] * (K // 32) + groups[None, :],
                 ns[:, None] < N, other=0)
    se = tl.reshape(tl.broadcast_to(sg[:, :, None], (BN, BK // 32, 32)), (BN, BK))
    ze = tl.reshape(tl.broadcast_to(zg[:, :, None], (BN, BK // 32, 32)), (BN, BK))
    shifted = (code.to(tl.float32) - ze.to(tl.float32)).to(tl.bfloat16)
    return (shifted.to(tl.float32) * se.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _gemm(X, Q, SG, ZG, Y, M, N: tl.constexpr, K: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          GROUP_M: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn: tl.constexpr = tl.cdiv(N, BN)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    size_m = tl.minimum(nm - first_m, GROUP_M)
    pid_m = first_m + pid % size_m
    pid_n = (pid % (GROUP_M * nn)) // size_m
    ms = pid_m * BM + tl.arange(0, BM)
    ns = pid_n * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for tile in tl.range(0, K // BK, num_stages=STAGES):
        k = tile * BK
        a = tl.load(X + ms[:, None] * K + k + ks[None, :],
                    ms[:, None] < M, other=0).to(tl.bfloat16)
        b = _tile(Q, SG, ZG, ns, k, N, K, BN, BK)
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)
    tl.store(Y + ms[:, None] * N + ns[None, :], acc.to(tl.bfloat16),
             (ms[:, None] < M) & (ns[None, :] < N))


@triton.jit
def _materialize(Q, SG, ZG, Y, N: tl.constexpr, K: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr):
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.program_id(1) * BK
    b = _tile(Q, SG, ZG, ns, k, N, K, BN, BK)
    tl.store(Y + ns[:, None] * K + k + tl.arange(0, BK)[None, :], b,
             ns[:, None] < N)


def prepare(weights, sg, zg):
    q, sc, ss, z, zs = weights
    n, k = q.shape[0], q.shape[1] * 2
    assert sg.shape == zg.shape == (n, k // 32)
    assert sg.dtype == zg.dtype == torch.bfloat16
    assert sg.device == zg.device == q.device
    _prepare[(triton.cdiv(n * (k // 32), 256),)](sc, ss, z, zs, sg, zg,
        SIZE=n * (k // 32), GROUPS=k // 32, BLOCK=256,
        num_warps=4, num_stages=1, enable_fp_fusion=False)


def launch(x, weights, sg, zg, out, config):
    m, n, k = _validate(x, weights, 'bulk-staged-bf16')
    config.validate()
    assert config.split == 1 and out.shape == (m, n)
    assert out.dtype == torch.bfloat16 and out.device == x.device
    prepare(weights, sg, zg)
    _gemm[(triton.cdiv(m, config.bm) * triton.cdiv(n, config.bn),)](
        x, weights[0], sg, zg, out, m, N=n, K=k,
        BM=config.bm, BN=config.bn, BK=config.bk, GROUP_M=config.group_m,
        STAGES=config.stages, num_warps=config.warps,
        num_stages=config.stages, enable_fp_fusion=False)
    return out


def materialize(weights, sg, zg):
    n, k = weights[0].shape[0], weights[0].shape[1] * 2
    prepare(weights, sg, zg)
    out = torch.empty((n, k), device=weights[0].device, dtype=torch.bfloat16)
    _materialize[(triton.cdiv(n, 32), k // 128)](weights[0], sg, zg, out,
        N=n, K=k, BN=32, BK=128, num_warps=4, num_stages=1,
        enable_fp_fusion=False)
    return out


def configs():
    return [Config(64, 64, 64), Config(64, 64, 128),
            Config(64, 128, 64), Config(64, 128, 128, stages=2)]
