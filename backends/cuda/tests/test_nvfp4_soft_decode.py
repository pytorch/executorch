# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 scale and code decoding on GPUs without native FP8/FP4 conversion."""

import pytest
import torch
import triton
import triton.language as tl
from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm as kernels
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    _e4m3_to_f32,
    nvfp4_embedding,
    nvfp4_gemv,
    nvfp4_w4a16,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = E2M1 + [-v for v in E2M1]


def finite_e4m3_bytes():
    b = torch.arange(256, dtype=torch.int32)
    return b[(b & 0x7F) != 0x7F].to(torch.uint8).cuda()


def relative_error_per_row(actual, expected, x, w):
    """Error of each output column (one weight row) relative to that row's
    magnitude, so a wrong small or subnormal scale cannot hide behind larger
    rows; |x| @ |w| does not shrink when a column's terms cancel."""
    err = (actual.float() - expected).abs().amax(0)
    size = (x.float().abs() @ w.abs().T).amax(0)
    return torch.where(size > 0, err / size, err)


def assert_bits_equal(actual, expected):
    assert torch.equal(
        actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32)
    )


def full_range_weight(k, seed=0):
    """Random codes; row r has the r-th finite E4M3 byte as its only scale, so
    all 254 rows cover every finite byte, both signs and every subnormal."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = finite_e4m3_bytes()[:, None].expand(-1, k // 16).contiguous()
    q = torch.randint(
        0, 256, (s.shape[0], k // 2), device="cuda", dtype=torch.uint8, generator=g
    )
    return q, s


def reference_weight(q, s, pts):
    """FP32 E2M1 * E4M3 * global scale, the product every kernel rounds."""
    codes = torch.stack((q & 15, q >> 4), -1).reshape(q.shape[0], -1).long()
    lut = torch.tensor(E2M1, device=q.device)
    scale = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    return lut[codes] * scale * pts.reshape(-1, 1)


def relative_error(actual, expected):
    return float((actual.float() - expected).abs().max() / expected.abs().max())


@triton.jit
def _decode_e4m3_kernel(B, Y, N, BLOCK: tl.constexpr, NATIVE_E4M3: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, _e4m3_to_f32(tl.load(B + i, i < N, 0), NATIVE_E4M3), i < N)


@pytest.mark.parametrize("native", [False, True])
def test_e4m3_decode_is_exact(native):
    # The software path is selectable on any GPU; the native one needs SM89+.
    if native and not kernels._has_native_e4m3():
        pytest.skip("native E4M3 (SM89+) required")
    b = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
    y = torch.empty(b.shape, device="cuda", dtype=torch.float32)
    _decode_e4m3_kernel[(1,)](b, y, b.numel(), BLOCK=256, NATIVE_E4M3=native)
    expected = b.view(torch.float8_e4m3fn).float()
    finite = (b & 0x7F) != 0x7F
    assert_bits_equal(y[finite], expected[finite])
    assert y[~finite].isnan().all()


@pytest.mark.parametrize("row_pts", [False, True])
def test_kernels_decode_every_scale(row_pts):
    k = 512
    q, s = full_range_weight(k)
    n = q.shape[0]
    pts = (
        torch.rand(n, device="cuda") + 0.5
        if row_pts
        else torch.tensor([0.37], device="cuda")
    )
    w = reference_weight(q, s, pts).to(torch.bfloat16).float()
    for m in (1, 3, 37):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        expected = x.float() @ w.T
        for split_k in (1, 4):
            actual = (
                nvfp4_gemv(x, q, s, pts, split_k=split_k)
                if m <= 4
                else nvfp4_w4a16(x, q, s, pts, split_k=split_k)
            )
            error = relative_error_per_row(actual, expected, x, w)
            assert bool((error < 1e-2).all()), (m, split_k)


def test_embedding_decodes_every_scale_exactly():
    k = 128
    q, s = full_range_weight(k, seed=1)
    n = q.shape[0]
    pts = torch.rand(n, device="cuda") + 0.5
    ids = torch.arange(n, device="cuda")
    codes = torch.stack((q & 15, q >> 4), -1).reshape(n, -1).long()
    lut = torch.tensor(E2M1, device="cuda")
    scale = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    expected = (lut[codes] * (scale * pts[:, None])).to(torch.bfloat16)[ids]
    actual = nvfp4_embedding(ids, q, s, pts, torch.bfloat16)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
