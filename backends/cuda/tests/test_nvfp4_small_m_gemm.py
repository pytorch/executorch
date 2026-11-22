# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 linears for small M (2..16): rows / W4A4 / W4A16 per bucket."""

from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from executorch.backends.cuda.autotune.launch_params import InvalidLaunchParam
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import (
    select_bucket,
)
from executorch.backends.cuda.triton.kernels import nvfp4_small_m_gemm as small_m

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = E2M1 + [-v for v in E2M1]


@pytest.fixture(autouse=True)
def default_policy(monkeypatch):
    for name in (
        "ET_CUDA_NVFP4_ACT",
        "ET_CUDA_NVFP4_PREFILL_OUTPUT",
        "ET_CUDA_NVFP4_PREFILL_IMPL",
        "ET_CUDA_NVFP4_SMALL_M",
    ):
        monkeypatch.delenv(name, raising=False)


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("pts_kind", ["unit", "scalar", "row"])
@pytest.mark.parametrize("m", [2, 3, 4, 5, 8, 13, 16])
def test_rows_and_w4a16_match_reference(m, pts_kind, dtype):
    # K = 1040 is not a multiple of any BK: the tail masks run.
    n, k = 200, 1040
    q, s = weight(n, k)
    pts = make_pts(pts_kind, n)
    unit = pts_kind == "unit"
    bucket = next(b for b in small_m.SMALL_M_BUCKETS if m <= b)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
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
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("pts_kind", ["scalar", "row"])
@pytest.mark.parametrize("m", [2, 8, 16])
def test_w4a4_matches_reference(m, pts_kind, static):
    # Row scales take the ROW_PTS paths of the kernel and the reduction; a
    # static activation scale replaces the per-row one.
    n, k = 200, 1024
    q, s = weight(n, k, scales=(0x30, 0x40))
    pts = make_pts(pts_kind, n)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    act = x.float().abs().amax().reshape(1, 1) / (6 * 448) if static else None
    expected = reference(x, q, s, pts)
    for split in (1, 2):
        y = small_m.launch_w4a4(x, q, s, pts, act, split)
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
    bucket = bucket_of(x, q, s, pts)
    assert bucket == next(b for b in small_m.SMALL_M_BUCKETS if m <= b)
    # Outside autotuning the op runs its first candidate: rows, split 1.
    y = small_m.NVFP4_SMALL_M_GEMM.op(bucket)(x, q, s, pts, None, False, False)
    assert relative_error(y, reference(x, q, s, pts)) < 1e-2


def bucket_of(x, q, s, pts):
    return select_bucket(small_m.NVFP4_SMALL_M_GEMM, x, q, s, pts, None, False, False)


def test_bucket_selection():
    q, s = weight(64, 256)
    pts = torch.ones(1, device="cuda")
    x = lambda m, k=256, dtype=torch.bfloat16: torch.empty(  # noqa: E731
        m, k, device="cuda", dtype=dtype
    )
    assert [bucket_of(x(m), q, s, pts) for m in (2, 3, 4, 5, 8, 9, 16)] == [
        2,
        4,
        4,
        8,
        8,
        16,
        16,
    ]
    assert bucket_of(x(17), q, s, pts) is None
    assert bucket_of(x(4, dtype=torch.float32), q, s, pts) is None
    assert bucket_of(x(4, k=128), q, s, pts) is None


@pytest.mark.parametrize("weight_only", [False, True])
def test_bucket_op_autotunes_mode_and_split(weight_only):
    from executorch.backends.cuda.autotune.launch_params import (
        autotune_launch_params,
        clear_launch_param_cache,
    )

    n, k = 160, 2048
    q, s = weight(n, k, seed=4, scales=(0x30, 0x40))
    pts = torch.tensor([0.37], device="cuda")
    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    clear_launch_param_cache()
    with autotune_launch_params() as stats:
        y = small_m.NVFP4_SMALL_M_GEMM.op(8)(x, q, s, pts, None, False, weight_only)
    (pick,) = [p for p in stats.picks if p.param == "mode_split"]
    modes = {c // small_m._MODE_STRIDE for c in pick.times_us}
    fp4 = native_fp4() and not weight_only
    ids = small_m._MODE_IDS
    expected = {ids["rows"], ids["w4a16"]} | ({ids["w4a4"]} if fp4 else set())
    assert modes == expected
    tolerance = 0.2 if pick.choice // small_m._MODE_STRIDE == ids["w4a4"] else 1e-2
    assert relative_error(y, reference(x, q, s, pts)) < tolerance


@pytest.mark.parametrize("m", [1, 2, 8, 16, 17])
def test_linear_dispatch_routes_small_m(m, monkeypatch):
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch

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
    # With native FP4, decode-size M keeps nvfp4_auto_decode.
    decode = m == 1 or (m <= 4 and nvfp4_dispatch._has_native_fp4())
    expected = [] if decode or m == 17 else [bucket_of(x, q, s, w.per_tensor_scale)]
    assert calls == expected
    # Eager runs each op's first candidate, which keeps BF16 activations.
    assert relative_error(y, reference(x, q, s, w.per_tensor_scale)) < 3e-2
