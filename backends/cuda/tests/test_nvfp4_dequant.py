# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Whole-weight NVFP4 dequantization and the dequant + matmul prefill path."""

from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch
from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm as kernels
from executorch.backends.cuda.triton.kernels.nvfp4_dequant import nvfp4_dequant

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = E2M1 + [-v for v in E2M1]


def weight(n, k, seed=0, scales=(0, 0x7F)):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(
        0, 256, (n, k // 2), device="cuda", dtype=torch.uint8, generator=g
    )
    s = torch.randint(
        *scales, (n, k // 16), device="cuda", dtype=torch.uint8, generator=g
    )
    return q, s


def kernel_order_reference(q, s, pts, dtype):
    """(e2m1 * e4m3) * global scale in FP32, rounded once: the fused kernels' weights."""
    codes = torch.stack((q & 15, q >> 4), -1).reshape(q.shape[0], -1).long()
    lut = torch.tensor(E2M1, device=q.device)
    scale = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    return ((lut[codes] * scale) * pts.reshape(-1, 1)).to(dtype)


def relative_error(actual, expected):
    return float((actual.float() - expected).abs().max() / expected.abs().max())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("pts_kind", ["unit", "scalar", "row"])
@pytest.mark.parametrize("n,k", [(1, 32), (37, 272), (520, 1024), (300, 6656)])
def test_dequant_is_bit_exact(n, k, pts_kind, dtype):
    q, s = weight(n, k)
    if dtype == torch.float16:
        # Keep products inside the FP16 range.
        s = s.clamp(max=0x60)
    pts = {
        "unit": torch.ones(1, device="cuda"),
        "scalar": torch.tensor([0.37], device="cuda"),
        "row": torch.rand(n, device="cuda") + 0.5,
    }[pts_kind]
    actual = nvfp4_dequant(q, s, pts, dtype)
    expected = kernel_order_reference(q, s, pts, dtype)
    assert actual.shape == (n, k) and actual.dtype == dtype
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_dequant_matches_fused_w4a16_weights():
    """Dense matmul of the dequantized weight == fused W4A16 up to summation order."""
    m, n, k = 64, 384, 2048
    q, s = weight(n, k, seed=1, scales=(0x30, 0x40))
    pts = torch.tensor([0.37], device="cuda")
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    dense = x.float() @ nvfp4_dequant(q, s, pts).float().T
    fused = kernels.nvfp4_w4a16(x, q, s, pts, split_k=1)
    assert relative_error(fused, dense) < 1e-2


@pytest.mark.parametrize(
    "env,native,expected",
    [
        (None, False, "dequant"),
        (None, True, "kernel"),
        ("auto", False, "dequant"),
        ("kernel", False, "kernel"),
        ("dequant", True, "dequant"),
    ],
)
def test_prefill_impl_selection(env, native, expected, monkeypatch):
    if env is None:
        monkeypatch.delenv("ET_CUDA_NVFP4_PREFILL_IMPL", raising=False)
    else:
        monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_IMPL", env)
    with mock.patch.object(nvfp4_dispatch, "_has_native_fp4", return_value=native):
        assert nvfp4_dispatch._nvfp4_prefill_impl() == expected


def test_prefill_impl_rejects_unknown(monkeypatch):
    monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_IMPL", "cublas")
    with pytest.raises(ValueError):
        nvfp4_dispatch._nvfp4_prefill_impl()


@pytest.mark.parametrize("impl", ["dequant", "kernel"])
@pytest.mark.parametrize("row_pts", [False, True])
def test_linear_prefill_dispatch(impl, row_pts, monkeypatch):
    monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_IMPL", impl)
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", "w4a16")
    m, n, k = 37, 192, 512
    q, s = weight(n, k, seed=2, scales=(0x30, 0x40))
    pts = (
        torch.rand(n, device="cuda") + 0.5
        if row_pts
        else torch.tensor(0.37, device="cuda")
    )
    w = CudaNVFP4Tensor(q, s, pts)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    spy = mock.Mock(wraps=nvfp4_dispatch._dequant_linear)
    with mock.patch.object(nvfp4_dispatch, "_dequant_linear", spy):
        y = F.linear(x, w)
    assert spy.call_count == (1 if impl == "dequant" else 0)
    expected = x.float() @ kernel_order_reference(q, s, pts, x.dtype).float().T
    assert relative_error(y, expected) < 1e-2


def test_dequant_linear_chunks_huge_weights():
    """N above the chunk size dequantizes and multiplies slice by slice."""
    from executorch.backends.cuda.quantize_op_dispatch import _gemm_family_dispatch
    from executorch.backends.cuda.triton.kernels import nvfp4_dequant as dequant

    m, n, k = 9, _gemm_family_dispatch._DEQUANT_N_THRESHOLD + 1000, 256
    q, s = weight(n, k, seed=3, scales=(0x30, 0x40))
    pts = torch.rand(n, device="cuda") + 0.5
    w = CudaNVFP4Tensor(q, s, pts)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    spy = mock.Mock(wraps=dequant.nvfp4_dequant)
    with mock.patch.object(dequant, "nvfp4_dequant", spy):
        y = nvfp4_dispatch._dequant_linear(x, w)
    chunk = _gemm_family_dispatch._DEQUANT_N_CHUNK
    assert spy.call_count == -(-n // chunk)
    expected = x.float() @ kernel_order_reference(q, s, pts, x.dtype).float().T
    assert relative_error(y, expected) < 1e-2
