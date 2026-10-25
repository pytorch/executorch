"""Experimental grouped Triton W4A16 for the existing ET coalesced packing.

Reference: GemLite f649e80e3014bc7ca8195573683c2c440f3bd94e,
https://github.com/mobiusml/gemlite (Apache-2.0). This implementation adapts
the tiled grouped GEMM and grouped-M scheduling ideas to ET's different
row-major packed codes and two-level scale/zero metadata. No GemLite
weight conversion or activation quantization is used.

This module does not register a model route or alter native ET dispatch.
The caller must explicitly choose the arithmetic of its existing entry:
staged BF16 for compiled bulk dequant, or the independently verified BF16
zero-shift FMA for the current native M8/M16 decoder. M alone is insufficient.
"""
from dataclasses import asdict, dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class Config:
    bm: int = 32
    bn: int = 64
    bk: int = 64
    split: int = 1
    warps: int = 4
    stages: int = 3
    group_m: int = 8

    def validate(self):
        assert self.bm in (16, 32, 64, 128)
        assert self.bn in (32, 64, 128)
        assert self.bk in (32, 64, 128, 256)
        assert self.split in (1, 2, 4, 8, 16)
        assert self.warps in (4, 8) and self.stages in (1, 2, 3, 4)
        assert self.group_m in (1, 4, 8)
        return self


@triton.jit
def _weight_tile(Q, SC, SS, Z, ZS, ns, k_start, N: tl.constexpr,
                 K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 ZERO_FMA: tl.constexpr):
    # BK divides K256, so every tile uses one superblock step per N.
    packed_k = k_start // 2 + tl.arange(0, BK // 2)
    packed = tl.load(Q + ns[:, None] * (K // 2) + packed_k[None, :],
                     mask=ns[:, None] < N, other=0).to(tl.uint32)
    codes = tl.interleave(packed & 15, (packed >> 4) & 15).to(tl.bfloat16)
    groups = k_start // 32 + tl.arange(0, BK // 32)
    scale = tl.load(SC + ns[:, None] * (K // 32) + groups[None, :],
                    mask=ns[:, None] < N, other=0).to(tl.bfloat16)
    zero = tl.load(Z + ns[:, None] * (K // 32) + groups[None, :],
                   mask=ns[:, None] < N, other=0).to(tl.bfloat16)
    step = tl.load(SS + ns * (K // 256) + k_start // 256,
                   mask=ns < N, other=0).to(tl.bfloat16)
    zero_step = tl.load(ZS + ns * (K // 256) + k_start // 256,
                        mask=ns < N, other=0).to(tl.bfloat16)
    sg = (scale.to(tl.float32) * step[:, None].to(tl.float32)).to(tl.bfloat16)
    se = tl.reshape(tl.broadcast_to(sg[:, :, None], (BN, BK // 32, 32)), (BN, BK))
    if ZERO_FMA:
        # Exact FP32 product/subtract of BF16 operands, then one BF16 round.
        ze = tl.reshape(tl.broadcast_to(zero[:, :, None], (BN, BK // 32, 32)), (BN, BK))
        shifted = tl.fma(-ze.to(tl.float32), zero_step[:, None].to(tl.float32),
                         codes.to(tl.float32)).to(tl.bfloat16)
    else:
        zg = (zero.to(tl.float32) * zero_step[:, None].to(tl.float32)).to(tl.bfloat16)
        ze = tl.reshape(tl.broadcast_to(zg[:, :, None], (BN, BK // 32, 32)), (BN, BK))
        shifted = (codes.to(tl.float32) - ze.to(tl.float32)).to(tl.bfloat16)
    return (shifted.to(tl.float32) * se.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _gemm(X, Q, SC, SS, Z, ZS, OUT, M, N: tl.constexpr, K: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          SPLIT: tl.constexpr, GROUP_M: tl.constexpr, STAGES: tl.constexpr,
          ZERO_FMA: tl.constexpr):
    pid = tl.program_id(0)
    split = tl.program_id(1)
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
    tiles: tl.constexpr = K // BK
    per_split: tl.constexpr = tiles // SPLIT
    extra: tl.constexpr = tiles % SPLIT
    first = split * per_split + tl.minimum(split, extra)
    count = per_split + (split < extra)
    acc = tl.zeros((BM, BN), tl.float32)
    for tile in tl.range(first, first + count, num_stages=STAGES):
        k = tile * BK
        a = tl.load(X + ms[:, None] * K + k + ks[None, :],
                    mask=ms[:, None] < M, other=0).to(tl.bfloat16)
        b = _weight_tile(Q, SC, SS, Z, ZS, ns, k, N, K, BN, BK, ZERO_FMA)
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
    offset = ms[:, None] * N + ns[None, :]
    if SPLIT == 1:
        tl.store(OUT + offset, acc.to(tl.bfloat16),
                 mask=(ms[:, None] < M) & (ns[None, :] < N))
    else:
        tl.store(OUT + split * M * N + offset, acc,
                 mask=(ms[:, None] < M) & (ns[None, :] < N))


@triton.jit
def _reduce(PART, OUT, SIZE, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for split in tl.static_range(SPLIT):
        acc += tl.load(PART + split * SIZE + offsets, mask=offsets < SIZE, other=0)
    tl.store(OUT + offsets, acc.to(tl.bfloat16), mask=offsets < SIZE)


@triton.jit
def _materialize(Q, SC, SS, Z, ZS, OUT, N: tl.constexpr, K: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, ZERO_FMA: tl.constexpr):
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.program_id(1) * BK
    b = _weight_tile(Q, SC, SS, Z, ZS, ns, k, N, K, BN, BK, ZERO_FMA)
    ks = k + tl.arange(0, BK)
    tl.store(OUT + ns[:, None] * K + ks[None, :], b, mask=ns[:, None] < N)


def _validate(x, weights, arithmetic):
    assert arithmetic in ('bulk-staged-bf16', 'native-zero-fma-bf16')
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous() and x.ndim == 2
    q, sc, ss, z, zs = weights
    n, half_k = q.shape
    k = x.shape[1]
    assert x.shape[0] > 0 and k == 2 * half_k and k % 256 == 0
    for tensor, shape, dtype in [(q, (n, k // 2), torch.uint8),
                                 (sc, (n, k // 32), torch.uint8),
                                 (z, (n, k // 32), torch.uint8),
                                 (ss, (n, k // 256), torch.float16),
                                 (zs, (n, k // 256), torch.float16)]:
        assert tuple(tensor.shape) == shape and tensor.dtype == dtype
        assert tensor.is_contiguous() and tensor.device == x.device
    return x.shape[0], n, k


def launch(x, weights, *, config=Config(), arithmetic, out=None, partial=None):
    """Explicit experimental launch; no auto selection, model registration or cast."""
    config.validate()
    m, n, k = _validate(x, weights, arithmetic)
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    assert tuple(out.shape) == (m, n) and out.dtype == torch.bfloat16
    assert out.device == x.device and out.is_contiguous()
    if config.split > 1:
        if partial is None:
            partial = torch.empty((config.split, m, n), device=x.device, dtype=torch.float32)
        assert tuple(partial.shape) == (config.split, m, n) and partial.dtype == torch.float32
        assert partial.device == x.device and partial.is_contiguous()
    target = out if config.split == 1 else partial
    grid = (triton.cdiv(m, config.bm) * triton.cdiv(n, config.bn), config.split)
    _gemm[grid](x, *weights, target, m, N=n, K=k, BM=config.bm,
                BN=config.bn, BK=config.bk, SPLIT=config.split,
                GROUP_M=config.group_m, STAGES=config.stages,
                ZERO_FMA=arithmetic == 'native-zero-fma-bf16',
                num_warps=config.warps, num_stages=config.stages,
                enable_fp_fusion=False)
    if config.split > 1:
        _reduce[(triton.cdiv(m * n, 256),)](partial, out, m * n, SPLIT=config.split,
                                          BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return out


def materialize(weights, *, arithmetic, bk=64):
    """Diagnostic only: temporary dense weights must not enter a model route."""
    q = weights[0]
    n, k = q.shape[0], 2 * q.shape[1]
    dummy = torch.empty((1, k), dtype=torch.bfloat16, device=q.device)
    _validate(dummy, weights, arithmetic)
    assert bk in (32, 64, 128, 256)
    out = torch.empty((n, k), dtype=torch.bfloat16, device=q.device)
    _materialize[(triton.cdiv(n, 32), k // bk)](*weights, out, N=n, K=k, BN=32,
        BK=bk, ZERO_FMA=arithmetic == 'native-zero-fma-bf16', num_warps=4,
        num_stages=1, enable_fp_fusion=False)
    return out


def configs(m):
    """Bounded screening space; all candidates still require measured acceptance."""
    values = [Config(32, 64, 64), Config(32, 64, 128), Config(32, 128, 64),
              Config(64, 64, 64), Config(64, 128, 64), Config(64, 64, 128),
              Config(64, 128, 128, stages=2), Config(128, 64, 64, warps=8)]
    if m <= 128:
        values += [Config(16, 64, 64, split=4, stages=2),
                   Config(16, 64, 128, split=4, stages=2),
                   Config(32, 64, 64, split=4, stages=2),
                   Config(32, 64, 128, split=4, stages=2),
                   Config(32, 64, 128, split=8, stages=2)]
    return values
