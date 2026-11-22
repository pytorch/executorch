# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 linears for small static M (2..16): rows / W4A4 / W4A16 per bucket."""

from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from executorch.backends.cuda.autotune.launch_params import InvalidLaunchParam
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.triton.kernels import nvfp4_small_m_gemm as small_m

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = E2M1 + [-v for v in E2M1]


def native_fp4():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12


def weight(n, k, seed=0, scales=(0, 0x7F)):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(
        0, 256, (n, k // 2), device="cuda", dtype=torch.uint8, generator=g
    )
    s = torch.randint(
        *scales, (n, k // 16), device="cuda", dtype=torch.uint8, generator=g
    )
    return q, s


def reference(x, q, s, pts):
    codes = torch.stack((q & 15, q >> 4), -1).reshape(q.shape[0], -1).long()
    lut = torch.tensor(E2M1, device="cuda")
    scale = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    w = (lut[codes] * scale * pts.reshape(-1, 1)).to(x.dtype).float()
    return x.float() @ w.T


def relative_error(actual, expected):
    return float((actual.float() - expected).abs().max() / expected.abs().max())


def make_pts(kind, n):
    return {
        "unit": torch.ones(1, device="cuda"),
        "scalar": torch.tensor([0.37], device="cuda"),
        "row": torch.rand(n, device="cuda") + 0.5,
    }[kind]


@pytest.mark.parametrize("pts_kind", ["unit", "scalar", "row"])
@pytest.mark.parametrize("m", [2, 3, 4, 5, 8, 13, 16])
def test_rows_and_w4a16_match_reference(m, pts_kind):
    n, k = 200, 1024
    q, s = weight(n, k)
    pts = make_pts(pts_kind, n)
    unit = pts_kind == "unit"
    bucket = next(b for b in small_m.SMALL_M_BUCKETS if m <= b)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    expected = reference(x, q, s, pts)
    for split in (1, 2, 8):
        rows = small_m.launch_rows(bucket, x, q, s, pts, unit, split)
        w4a16 = small_m.launch_w4a16(x, q, s, pts, split)
        for name, y in (("rows", rows), ("w4a16", w4a16)):
            assert y.shape == (m, n) and y.dtype == x.dtype
            assert relative_error(y, expected) < 1e-2, (name, split)


def test_rows_without_native_fp4_matches_native_decode():
    """The software E2M1 decode feeds the rows kernel the same weights."""
    n, k, m = 128, 512, 8
    q, s = weight(n, k, seed=1)
    pts = torch.tensor([0.37], device="cuda")
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    with mock.patch.object(small_m, "_has_native_fp4_conversion", return_value=False):
        y = small_m.launch_rows(8, x, q, s, pts, False, 1)
    assert relative_error(y, reference(x, q, s, pts)) < 1e-2


@pytest.mark.skipif(not native_fp4(), reason="SM120 required")
@pytest.mark.parametrize("m", [2, 8, 16])
def test_w4a4_matches_reference(m):
    n, k = 200, 1024
    q, s = weight(n, k, scales=(0x30, 0x40))
    pts = torch.tensor([0.37], device="cuda")
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    expected = reference(x, q, s, pts)
    for split in (1, 2):
        y = small_m.launch_w4a4(x, q, s, pts, None, split)
        # NVFP4 activations: quantization error dominates.
        assert relative_error(y, expected) < 0.2


@pytest.mark.skipif(native_fp4(), reason="needs a GPU without native FP4")
def test_w4a4_rejected_without_native_fp4():
    q, s = weight(64, 256)
    x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(InvalidLaunchParam):
        small_m.launch_w4a4(x, q, s, torch.ones(1, device="cuda"), None, 1)


def test_rows_rejects_illegal_split():
    q, s = weight(64, 256)
    x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(InvalidLaunchParam):
        small_m.launch_rows(2, x, q, s, torch.ones(1, device="cuda"), False, 4)


@pytest.mark.parametrize("m", [2, 7, 16])
def test_bucket_op_matches_reference(m):
    n, k = 160, 2048
    q, s = weight(n, k, seed=2, scales=(0x30, 0x40))
    pts = torch.tensor([0.37], device="cuda")
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bucket = small_m.small_m_bucket(x, q)
    assert bucket == next(b for b in small_m.SMALL_M_BUCKETS if m <= b)
    y = small_m.NVFP4_SMALL_M_GEMM.op(bucket)(x, q, s, pts, None, False)
    tolerance = 0.2 if native_fp4() else 1e-2
    assert relative_error(y, reference(x, q, s, pts)) < tolerance


def test_bucket_selection(monkeypatch):
    q, _ = weight(64, 256)
    x = lambda m, k=256, dtype=torch.bfloat16: torch.empty(  # noqa: E731
        m, k, device="cuda", dtype=dtype
    )
    assert small_m.small_m_bucket(x(1), q) is None
    assert [small_m.small_m_bucket(x(m), q) for m in (2, 3, 4, 5, 8, 9, 16)] == [
        2,
        4,
        4,
        8,
        8,
        16,
        16,
    ]
    assert small_m.small_m_bucket(x(17), q) is None
    assert small_m.small_m_bucket(x(4, dtype=torch.float32), q) is None
    assert small_m.small_m_bucket(x(4, k=128), q) is None
    monkeypatch.setenv("ET_CUDA_NVFP4_SMALL_M", "0")
    assert small_m.small_m_bucket(x(4), q) is None


def test_mode_list_validation(monkeypatch):
    monkeypatch.setenv("ET_CUDA_NVFP4_SMALL_M_MODES", "w4a16,rows")
    assert small_m._enabled_modes() == [0, 2]
    monkeypatch.setenv("ET_CUDA_NVFP4_SMALL_M_MODES", "rows,cublas")
    with pytest.raises(ValueError):
        small_m._enabled_modes()


@pytest.mark.parametrize("m", [1, 2, 8, 16, 17])
def test_linear_dispatch_routes_small_m(m, monkeypatch):
    from executorch.backends.cuda.quantize_op_dispatch import (  # noqa: F401
        nvfp4_dispatch,
    )

    monkeypatch.delenv("ET_CUDA_NVFP4_ACT", raising=False)
    monkeypatch.delenv("ET_CUDA_NVFP4_SMALL_M", raising=False)
    n, k = 96, 512
    q, s = weight(n, k, seed=3, scales=(0x30, 0x40))
    w = CudaNVFP4Tensor(q, s, torch.tensor(0.5, device="cuda"))
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    calls = []
    real_op = small_m.NVFP4_SMALL_M_GEMM.op

    def op(bucket):
        calls.append(bucket)
        return real_op(bucket)

    with mock.patch.object(small_m.NVFP4_SMALL_M_GEMM, "op", op):
        y = F.linear(x, w)
    assert calls == ([] if m in (1, 17) else [small_m.small_m_bucket(x, q)])
    tolerance = 0.2 if native_fp4() else 3e-2
    assert relative_error(y, reference(x, q, s, w.per_tensor_scale)) < tolerance
