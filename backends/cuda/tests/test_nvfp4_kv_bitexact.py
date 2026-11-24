# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bit-exactness of the NVFP4 KV encode/decode paths on every architecture.

Pre-sm_89 GPUs take software E4M3 paths and pre-sm_120 GPUs software E2M1
paths; newer GPUs use the native cvt instructions. Every check runs each
``ARCH`` the device supports out of {80, 89, 120} (software only, native E4M3
with software E2M1, all native) against an architecture-independent
reference, so one run on a new GPU checks every path it can execute.

* E4M3 encode: every fp32 bit pattern in [0, 448] vs torch float8_e4m3fn (RN-even).
* E2M1 encode: every non-NaN fp32 bit pattern vs an fp64 nearest-even reference
  with saturation and the sign taken from the sign bit (cvt.rn.satfinite).
* Tile decode (``_load_kv_tile_f16``): all 256 code bytes x all finite E4M3
  scales except -0 (never written; the software decode returns +0 for it on
  Triton < 3.8), a distinct scale per K16 group, both scale layouts, vs exact
  fp64 products (exact in fp16).
* Fused write: V cache bytes vs a PyTorch emulation of the write kernel (V has
  no norm/RoPE, and the kernel divides with IEEE round-to-nearest, so every
  fp32 step is reproducible).
* Cross-architecture write: the four cache tensors after a model-shaped write
  (K norm + RoPE, ring wrap) are identical for every ``ARCH``.
"""

import unittest
from unittest import mock

import executorch.backends.cuda.triton.kernels.nvfp4_kv as K

import torch
import triton
import triton.language as tl

dev = "cuda"


def arches():
    """The ARCH values the current device can execute."""
    return [a for a in (80, 89, 120) if a <= K.cuda_arch()]


@triton.jit
def _e4m3_enc_kernel(Y, base, N, BLOCK: tl.constexpr, ARCH: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = (base + i).to(tl.uint32).to(tl.float32, bitcast=True)
    tl.store(Y + i, K._e4m3_encode_nonneg(x, ARCH), mask=m)


@triton.jit
def _e2m1_enc_kernel(Y, base, N, BLOCK: tl.constexpr, ARCH: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = (base + i).to(tl.uint32).to(tl.float32, bitcast=True)
    # pair (x, -x): low nibble = code(x), high nibble = code(-x).  Negate by
    # flipping the sign bit: Triton < 3.8 lowers -x to 0 - x, so -0.0 -> +0.0.
    neg = ((base + i).to(tl.uint32) ^ 0x80000000).to(tl.float32, bitcast=True)
    tl.store(Y + i, K._e2m1x2(x, neg, ARCH), mask=m)


def _f32_from_bits(base, n):
    b = torch.arange(base, base + n, device=dev, dtype=torch.int64)
    return (
        b.to(torch.int32).view(torch.float32)
        if base + n <= 2**31
        else (b - 2**32).to(torch.int32).view(torch.float32)
    )


def _check_e4m3_encode(tc, arch):
    top = 0x43E00000  # 448.0
    chunk = 1 << 24
    bad = 0
    for base in range(0, top + 1, chunk):
        n = min(chunk, top + 1 - base)
        y = torch.empty(n, dtype=torch.uint8, device=dev)
        _e4m3_enc_kernel[(triton.cdiv(n, 1024),)](y, base, n, BLOCK=1024, ARCH=arch)
        ref = _f32_from_bits(base, n).to(torch.float8_e4m3fn).view(torch.uint8)
        bad += int((y != ref).sum())
    tc.assertEqual(bad, 0, "E4M3 encode, all fp32 in [0, 448]")


GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def _e2m1_ref(x):
    a = x.double().abs()
    d = (a[:, None] - GRID.to(x.device)[None, :]).abs()
    cand = d == d.min(1, keepdim=True).values
    parity = (torch.arange(8, device=x.device) & 1)[None, :].expand_as(d)
    code = torch.where(cand, parity, 2).argmin(1)
    code = torch.where(a > 6.0, 7, code)
    sign = (x.view(torch.int32) < 0).long() * 8
    return (code | sign).to(torch.uint8)


def _check_e2m1_encode(tc, arch):
    chunk = 1 << 22
    bad = n_tot = 0
    for base in range(0, 1 << 31, chunk):  # non-negative patterns; -x covers the sign
        n = chunk
        x = _f32_from_bits(base, n)
        keep = ~torch.isnan(x)
        y = torch.empty(n, dtype=torch.uint8, device=dev)
        _e2m1_enc_kernel[(triton.cdiv(n, 1024),)](y, base, n, BLOCK=1024, ARCH=arch)
        lo, hi = y & 15, y >> 4
        r_lo, r_hi = _e2m1_ref(x), _e2m1_ref(-x)
        bad += int(((lo != r_lo) & keep).sum() + ((hi != r_hi) & keep).sum())
        n_tot += 2 * int(keep.sum())
    tc.assertEqual(bad, 0, "E2M1 encode, all non-NaN fp32 (both signs)")


@triton.jit
def _tile_kernel(
    KQ,
    KS,
    LO,
    HI,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SF_GATHER: tl.constexpr,
    ARCH: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_valid = offs_n >= 0
    lo, hi = K._load_kv_tile_f16(
        KQ, KS, 0, 0, offs_n, n_valid, BLOCK_N, D, ARCH, SF_GATHER
    )
    half = tl.arange(0, D // 2)
    tl.store(LO + offs_n[:, None] * (D // 2) + half[None, :], lo)
    tl.store(HI + offs_n[:, None] * (D // 2) + half[None, :], hi)


def _check_tile_decode(tc, arch):
    D = 128
    scales = torch.tensor(
        [s for s in range(256) if s & 0x7F != 0x7F and s != 0x80], dtype=torch.uint8
    )
    rows = []
    for s in scales.tolist():
        for r in range(4):
            rows.append((s, torch.arange(r * 64, r * 64 + 64, dtype=torch.uint8)))
    L = len(rows)
    q = torch.stack([b for _, b in rows]).to(dev)
    # Row i's group g takes the scale after its own by g: every group of a row
    # differs, and every code byte still meets every scale across the rows.
    order = {sv: i for i, sv in enumerate(scales.tolist())}
    s = torch.tensor(
        [
            [scales[(order[sv] + g) % len(scales)] for g in range(D // 16)]
            for sv, _ in rows
        ],
        dtype=torch.uint8,
        device=dev,
    )
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float64,
        device=dev,
    )
    sf_b = s.view(torch.float8_e4m3fn).float().repeat_interleave(8, dim=1)
    r_lo = (lut[(q & 15).long()] * sf_b).half()
    r_hi = (lut[(q >> 4).long()] * sf_b).half()
    for gather in (False, True):
        lo = torch.empty(L, D // 2, dtype=torch.float16, device=dev)
        hi = torch.empty_like(lo)
        _tile_kernel[(L // 4,)](
            q,
            s,
            lo,
            hi,
            D=D,
            BLOCK_N=4,
            SF_GATHER=gather,
            ARCH=arch,
            num_warps=1,
        )
        bad = int(
            (lo.view(torch.int16) != r_lo.view(torch.int16)).sum()
            + (hi.view(torch.int16) != r_hi.view(torch.int16)).sum()
        )
        tc.assertEqual(
            bad, 0, f"tile decode ARCH={arch} SF_GATHER={gather} (bitwise fp16)"
        )


def _emulate_quant_row(x):
    """PyTorch emulation of ``_quantize_row_store`` (IEEE division, as ``div_rn``)."""
    D = x.shape[-1]
    xb = x.reshape(*x.shape[:-1], D // 16, 16)
    sf = (xb.abs().amax(-1) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    sff = sf.float()
    den = torch.where(sff > 0, sff, torch.ones_like(sff))
    v = xb / den[..., None]
    codes = _e2m1_ref(v.reshape(-1)).reshape(v.shape).reshape(*x.shape[:-1], D // 2, 2)
    qb = codes[..., 0] | (codes[..., 1] << 4)
    return qb, sf.view(torch.uint8)


def _check_write_v_bytes(tc):
    torch.manual_seed(0)
    D, H, T, L = 128, 2, 4096, 4096
    k = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
    v = (torch.randn(1, T, H, D, device=dev) * torch.logspace(-3, 2, D, device=dev)).to(
        torch.bfloat16
    )
    v[0, :7, 0, :] = 0  # all-zero rows
    pos = torch.arange(T, device=dev)
    cache = [
        torch.zeros(1, H, L, D // 2, dtype=torch.uint8, device=dev),
        torch.zeros(1, H, L, D // 16, dtype=torch.uint8, device=dev),
    ]
    vcache = [torch.zeros_like(c) for c in cache]
    torch.ops.triton.nvfp4_kv_write(k, v, pos, *cache, *vcache, None, False, None)
    qb, sb = _emulate_quant_row(v.float().transpose(1, 2))
    tc.assertEqual(int((vcache[0] != qb).sum()), 0, "write V codes vs emulation")
    tc.assertEqual(int((vcache[1] != sb).sum()), 0, "write V E4M3 scales vs emulation")


def _write_fixture():
    """K/V as strided slices of a qko-like projection, K norm + interleaved
    RoPE, a 4096-slot ring written in two chunks (4096 + 300, so it wraps)."""
    D, H, L = 128, 2, 4096
    g = torch.Generator(device=dev).manual_seed(1234)
    scale = torch.logspace(-2, 1.5, 8448, device=dev)
    proj = (torch.randn(1, 4396, 8448, device=dev, generator=g) * scale).bfloat16()
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=dev) / D))
    kc = [
        torch.zeros(1, H, L, D // 2, dtype=torch.uint8, device=dev),
        torch.zeros(1, H, L, D // 16, dtype=torch.uint8, device=dev),
    ]
    vc = [torch.zeros_like(c) for c in kc]
    for s, e in ((0, 4096), (4096, 4396)):
        x = proj[:, s:e]
        k = x[..., 4096:4352].view(1, e - s, H, D)
        v = x[..., 4352:4608].view(1, e - s, H, D)
        pos = torch.arange(s, e, device=dev)
        torch.ops.triton.nvfp4_kv_write(k, v, pos, *kc, *vc, inv, True, 1e-5)
    return dict(zip(("k_q", "k_s", "v_q", "v_s"), kc + vc))


class TestNVFP4KVBitExact(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")

    def test_tile_decode(self):
        for arch in arches():
            _check_tile_decode(self, arch)

    def test_write_v_bytes(self):
        for arch in arches():
            with mock.patch.object(K, "cuda_arch", lambda arch=arch: arch):
                _check_write_v_bytes(self)

    def test_e4m3_encode(self):
        for arch in arches():
            _check_e4m3_encode(self, arch)

    def test_e2m1_encode(self):
        for arch in arches():
            _check_e2m1_encode(self, arch)

    def test_cross_arch_write(self):
        """Every ARCH writes the same cache bytes (K norm, RoPE, ring wrap)."""
        if len(arches()) < 2:
            self.skipTest("needs an SM89+ GPU to compare against the software path")
        caches = {}
        for arch in arches():
            with mock.patch.object(K, "cuda_arch", lambda arch=arch: arch):
                caches[arch] = _write_fixture()
        base, software = next(iter(caches.items()))
        for arch, cache in caches.items():
            for name, t in cache.items():
                rows = (t != software[name]).flatten(0, 2).any(-1).nonzero()
                self.assertEqual(
                    rows.numel(),
                    0,
                    f"{name}: ARCH {arch} differs from ARCH {base} in rows "
                    f"{rows[:8].flatten().tolist()}",
                )


if __name__ == "__main__":
    unittest.main()
