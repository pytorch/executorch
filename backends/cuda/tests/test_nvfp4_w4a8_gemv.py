# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 W4A8 dp4a decode GEMV (int8 activations per 16 elements)."""

from unittest import mock

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from executorch.backends.cuda.autotune.launch_params import InvalidLaunchParam
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.triton.kernels import nvfp4_w4a8_gemv as w4a8
from executorch.backends.cuda.triton.kernels.nvfp4_w4a8_gemv import (
    _act_q8,
    _e2m1x8_to_s8,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = E2M1 + [-v for v in E2M1]


def weight(n, k, seed=0):
    """Random codes; row r has the (r % 254)-th finite E4M3 byte as its only
    scale, so 254 rows cover both signs and every subnormal."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(
        0, 256, (n, k // 2), device="cuda", dtype=torch.uint8, generator=g
    )
    b = torch.arange(256, device="cuda")
    finite = b[(b & 0x7F) != 0x7F].to(torch.uint8)
    s = finite[torch.arange(n, device="cuda") % finite.numel(), None]
    return q, s.expand(n, k // 16).contiguous()


def codes(q):
    return torch.stack((q & 15, q >> 4), -1).reshape(q.shape[0], -1).long()


def act_q8_reference(x):
    """Per-16 symmetric int8: q = rint(x * 127 / amax), scale = amax / 127."""
    m, k = x.shape
    blocks = x.float().reshape(m, k // 16, 16)
    amax = blocks.abs().amax(-1)
    inv = torch.where(amax > 0, 127.0 / amax, 0.0)
    q = torch.round(blocks * inv[..., None]).clamp(-127, 127).to(torch.int8)
    return q.reshape(m, k), amax / 127.0


def w4a8_reference(xq, xs, q, s, pts):
    """Exact integer dot products per K16 block, then FP32 block scales."""
    lut2 = torch.tensor([int(2 * v) for v in E2M1], device="cuda")
    wq = lut2[codes(q)]
    n, k = wq.shape
    m = xq.shape[0]
    dots = (xq.long().reshape(m, 1, k // 16, 16) * wq.reshape(1, n, k // 16, 16)).sum(
        -1
    )
    sw = s.view(torch.float8_e4m3fn).float()
    out = (dots.double() * sw[None].double() * xs[:, None].double()).sum(-1) * 0.5
    return (out * pts.double().reshape(1, -1)).float()


def dense_reference(x, q, s, pts):
    lut = torch.tensor(E2M1, device="cuda")
    scale = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    w = (lut[codes(q)] * scale * pts.reshape(-1, 1)).to(x.dtype).float()
    return x.float() @ w.T


def relative_error(actual, expected):
    return float((actual.float() - expected).abs().max() / expected.abs().max())


def relative_error_per_row(actual, expected, x, q, s, pts):
    """Error of each output column relative to that weight row's magnitude, so
    a wrong small or subnormal scale cannot hide behind larger rows."""
    lut = torch.tensor(E2M1, device="cuda")
    scale = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    w = lut[codes(q)] * scale * pts.reshape(-1, 1)
    err = (actual.float() - expected).abs().amax(0)
    size = (x.float().abs() @ w.abs().T).amax(0)
    return torch.where(size > 0, err / size, err)


def test_activation_quantization_matches_reference():
    torch.manual_seed(0)
    m, k = 3, 1040
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    x[1, 16:32] = 0
    xq = torch.empty((m, k), device="cuda", dtype=torch.int8)
    xs = torch.empty((m, k // 16), device="cuda", dtype=torch.float32)
    _act_q8[(m, triton.cdiv(k // 16, 64))](x, xq, xs, k, 64)
    rq, rs = act_q8_reference(x)
    torch.testing.assert_close(xs, rs, rtol=1e-6, atol=0)
    # Only exact rounding ties may resolve differently.
    assert (xq.int() - rq.int()).abs().max() <= 1
    assert (xq != rq).float().mean() < 1e-3
    assert bool((xq[1, 16:32] == 0).all()) and bool((xs[1, 1] == 0).all())


@triton.jit
def _unpack_kernel(Q, LO, HI, N: tl.constexpr):
    i = tl.arange(0, N)
    lo, hi = _e2m1x8_to_s8(tl.load(Q + i))
    tl.store(LO + i, lo)
    tl.store(HI + i, hi)


def test_e2m1_to_int8_permute_is_exact():
    """Every code at every nibble position becomes int8 2 * E2M1, in K order."""
    nibbles = (torch.arange(16)[:, None] + torch.arange(8)[None, :]) % 16
    words = (nibbles << (4 * torch.arange(8))[None, :]).sum(1)
    q = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32).cuda()
    lo = torch.empty(16, device="cuda", dtype=torch.int32)
    hi = torch.empty_like(lo)
    _unpack_kernel[(1,)](q, lo, hi, 16)
    actual = torch.cat(
        (lo.view(torch.int8).reshape(16, 4), hi.view(torch.int8).reshape(16, 4)), 1
    )
    lut2 = torch.tensor([int(2 * v) for v in E2M1], dtype=torch.int8)
    assert torch.equal(actual.cpu(), lut2[nibbles])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("row_pts", [False, True])
@pytest.mark.parametrize("split_k", [1, 2, 4, 8])
def test_w4a8_gemv_matches_integer_reference(dtype, row_pts, split_k):
    # K = 2880 is not a multiple of any BK: the tail masks run.
    n, k = 254, 2880
    q, s = weight(n, k)
    pts = (
        torch.rand(n, device="cuda") + 0.5
        if row_pts
        else torch.tensor([0.37], device="cuda")
    )
    for m in (1, 2, 4):
        torch.manual_seed(m)
        # Scaled so FP16 outputs of the 448-scale rows stay finite.
        x = (torch.randn(m, k, device="cuda") * 0.0625).to(dtype)
        xq, xs = act_q8_reference(x)
        y = w4a8.launch_w4a8_gemv(x, q, s, pts, split_k=split_k)
        assert y.shape == (m, n) and y.dtype == x.dtype
        # Only FP32 accumulation order and the output rounding differ.
        exact = relative_error_per_row(
            y, w4a8_reference(xq, xs, q, s, pts), x, q, s, pts
        )
        assert bool((exact < 8e-3).all())
        dense = relative_error_per_row(y, dense_reference(x, q, s, pts), x, q, s, pts)
        assert bool((dense < 3e-2).all())


def test_w4a8_rejects_illegal_split():
    q, s = weight(64, 256)
    x = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16)
    pts = torch.ones(1, device="cuda")
    with pytest.raises(InvalidLaunchParam):
        w4a8.launch_w4a8_gemv(x, q, s, pts, split_k=3)
    with pytest.raises(InvalidLaunchParam):
        w4a8.launch_w4a8_gemv(x, q, s, pts, split_k=2)


def test_w4a8_op_validates_and_matches_reference():
    n, k = 160, 2048
    q, s = weight(n, k, seed=5)
    pts = torch.tensor([0.37], device="cuda")
    x = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
    xq, xs = act_q8_reference(x)
    y = w4a8.nvfp4_w4a8_gemv(x, q, s, pts)
    assert relative_error(y, w4a8_reference(xq, xs, q, s, pts)) < 8e-3
    with pytest.raises(ValueError):
        w4a8.nvfp4_w4a8_gemv(x[:, :-16], q, s, pts)


@pytest.mark.parametrize("split", [1, 4])
def test_auto_decode_runs_w4a8_mode(split, monkeypatch):
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch

    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: False)
    n, k = 160, 2048
    q, s = weight(n, k, seed=7)
    pts = torch.tensor([0.37], device="cuda")
    x = torch.randn(2, k, device="cuda", dtype=torch.bfloat16)
    xq, xs = act_q8_reference(x)
    y = nvfp4_dispatch._decode(
        x, q, s, pts, None, mode_split=2 * nvfp4_dispatch._MODE_STRIDE + split
    )
    assert relative_error(y, w4a8_reference(xq, xs, q, s, pts)) < 8e-3


@pytest.mark.parametrize("native", [False, True])
def test_auto_decode_times_w4a8_only_without_native_fp4(native, monkeypatch):
    from executorch.backends.cuda.autotune.launch_params import (
        autotune_launch_params,
        clear_launch_param_cache,
    )
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch

    if native and not nvfp4_dispatch._has_native_fp4():
        pytest.skip("native FP4 required")
    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: native)
    n, k = 160, 2048
    q, s = weight(n, k, seed=8)
    pts = torch.tensor([0.37], device="cuda")
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    clear_launch_param_cache()
    with autotune_launch_params() as stats:
        out = nvfp4_dispatch.nvfp4_auto_decode(x, q, s, pts)
    (pick,) = [p for p in stats.picks if p.param == "mode_split"]
    modes = {c // nvfp4_dispatch._MODE_STRIDE for c in pick.times_us}
    assert modes == ({0, 1} if native else {0, 2})
    assert relative_error(out, dense_reference(x, q, s, pts)) < 3e-2


@pytest.mark.parametrize("mode", ["auto", "w4a8", "w4a16"])
def test_linear_dispatch_without_native_fp4(mode, monkeypatch):
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch

    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    n, k = 96, 512
    q, s = weight(n, k, seed=6)
    w = CudaNVFP4Tensor(q, s, torch.tensor(0.5, device="cuda"))
    x = torch.randn(2, 2, k, device="cuda", dtype=torch.bfloat16)
    forced = mock.Mock(wraps=w4a8.nvfp4_w4a8_gemv)
    with mock.patch.object(
        nvfp4_dispatch, "_has_native_fp4", return_value=False
    ), mock.patch.object(w4a8, "nvfp4_w4a8_gemv", forced):
        y = F.linear(x, w)
    assert forced.call_count == (mode == "w4a8")
    expected = dense_reference(x.reshape(4, k), q, s, w.per_tensor_scale)
    assert relative_error(y.reshape(4, n), expected) < 3e-2
