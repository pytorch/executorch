# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    nvfp4_embedding,
    nvfp4_gemv,
    nvfp4_quantize_activation,
    nvfp4_w4a16,
    nvfp4_w4a4,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="CUDA with native E4M3 (SM89+) required",
)


def weight(n, k):
    torch.manual_seed(42)
    q = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    s = (
        (0.05 + torch.rand(n, k // 16, device="cuda") * 0.1)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    return CudaNVFP4Tensor(q, s, torch.tensor(0.25, device="cuda"))


@pytest.mark.parametrize(
    "m,n,k", [(1, 71, 192), (2, 97, 80), (4, 65, 256), (16, 71, 192), (37, 97, 80)]
)
def test_w4a16(m, n, k):
    w = weight(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    expected = (x.float() @ w.dequantize().float().T).to(x.dtype)
    actual = (
        nvfp4_gemv(x, w.qdata, w.scale, w.per_tensor_scale, split_k=4)
        if m <= 4
        else nvfp4_w4a16(x, w.qdata, w.scale, w.per_tensor_scale)
    )
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.015)


@pytest.mark.parametrize("static", [0.0, 0.001])
def test_activation_reference(static):
    torch.manual_seed(123)
    x = torch.randn(7, 192, device="cuda", dtype=torch.bfloat16)
    x[0] = 0
    q, s, g = nvfp4_quantize_activation(x, static)
    # Independently choose global/block scales and nearest E2M1 codes.
    x32 = x.float()
    rg = (
        torch.full((7,), static, device="cuda")
        if static > 0
        else x32.abs().amax(1) / (448 * 6)
    )
    rg = torch.where(rg > 0, rg, 1.0)
    rs = (x32.reshape(7, -1, 16).abs().amax(2) / (6 * rg[:, None])).to(
        torch.float8_e4m3fn
    )
    torch.testing.assert_close(g, rg, rtol=1e-6, atol=1e-9)
    assert torch.equal(s, rs.view(torch.uint8))
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    normalized = x32.reshape(7, -1, 16) / torch.where(
        rs.float()[:, :, None] > 0, rs.float()[:, :, None] * rg[:, None, None], 1.0
    )
    errs = (normalized.abs()[:, :, :, None] - lut).abs()
    # Resolve ties to even code.
    best = errs.amin(-1, keepdim=True)
    even = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device="cuda")
    ref = even[(errs[..., even] == best).to(torch.int64).argmax(-1)]
    ref = (ref | ((normalized < 0).long() << 3)).reshape(7, 192).to(torch.uint8)
    unpack = torch.stack([q & 15, q >> 4], dim=-1).reshape(7, 192)
    assert torch.equal(unpack, ref)


def test_activation_rounding_boundaries():
    # Every E2M1 midpoint, both signs; exact global/block scales make the
    # expected ties independent of floating-point scale calculation.
    positive = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
    x = torch.tensor(
        [positive + [-v for v in positive], [0.0, -0.0] * 8],
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, s, g = nvfp4_quantize_activation(x, static_scale=1.0)
    actual = torch.stack((q & 15, q >> 4), dim=-1).reshape(2, 16)
    expected = torch.tensor(
        [[0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15], [0] * 16],
        device="cuda",
        dtype=torch.uint8,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(s, torch.tensor([[56], [0]], device="cuda", dtype=torch.uint8))
    torch.testing.assert_close(g, torch.ones_like(g), rtol=0, atol=0)


@pytest.mark.parametrize("m,n,k", [(1, 71, 192), (16, 97, 80), (37, 65, 256)])
def test_native_w4a4(m, n, k):
    if torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("SM120 required")
    w = weight(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    q, s, g = nvfp4_quantize_activation(x)
    aw = CudaNVFP4Tensor(q, s, torch.ones((), device="cuda"), orig_dtype=torch.float32)
    expected = ((aw.dequantize() * g[:, None]) @ w.dequantize(torch.float32).T).to(
        x.dtype
    )
    actual = nvfp4_w4a4(x, w.qdata, w.scale, w.per_tensor_scale)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("split_k", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("mode", ["w4a4", "w4a16"])
def test_gemm_split_k(mode, split_k):
    if mode == "w4a4" and torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("SM120 required")
    k = 4224 if split_k == 32 else 2176
    w = weight(97, k)
    x = torch.randn(17, k, device="cuda", dtype=torch.bfloat16)
    fn = nvfp4_w4a4 if mode == "w4a4" else nvfp4_w4a16
    expected = fn(x, w.qdata, w.scale, w.per_tensor_scale, split_k=1)
    actual = fn(x, w.qdata, w.scale, w.per_tensor_scale, split_k=split_k)
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.02)


@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("split_k", [1, 2, 4, 8, 16, 32])
def test_gemv_split_k(m, split_k):
    # A non-power-of-two global scale checks rounding of the BF16 weights.
    w = weight(71, 2176)
    w.per_tensor_scale = torch.tensor([0.37], device="cuda")
    x = torch.randn(m, 2176, device="cuda", dtype=torch.bfloat16)
    expected = (x.float() @ w.dequantize().float().T).to(x.dtype)
    actual = nvfp4_gemv(x, w.qdata, w.scale, w.per_tensor_scale, split_k=split_k)
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.015)


@pytest.mark.parametrize("mode", ["w4a4", "w4a16"])
def test_split_workspace_exceeds_int32(mode):
    """Exercise a real-model workspace whose element offsets exceed 2**31."""
    import triton
    from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm as kernels
    from torch._library.triton import set_wrap_triton_enabled

    if mode == "w4a4" and torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("SM120 required")
    if torch.cuda.mem_get_info()[0] < 13 * 1024**3:
        pytest.skip("Large split workspace needs 13 GiB of free VRAM")
    m, n, k, split = 4096, 39936, 2048, 16
    x = torch.full((m, k), 0.5, device="cuda", dtype=torch.bfloat16)
    q = torch.full((n, k // 2), 0x11, device="cuda", dtype=torch.uint8)
    s = torch.full((n, k // 16), 56, device="cuda", dtype=torch.uint8)
    pts = torch.ones(1, device="cuda")
    partial = torch.empty((split, m, n), device="cuda", dtype=torch.float32)
    y = torch.empty((m, n), device="cuda", dtype=x.dtype)
    with set_wrap_triton_enabled(False):
        if mode == "w4a4":
            a, sa, g = kernels.nvfp4_quantize_activation(x)
            kernels._w4a4.fn[(triton.cdiv(m, 64) * triton.cdiv(n, 128), split)](
                a,
                sa,
                g,
                q,
                s,
                pts,
                partial,
                m,
                n,
                k,
                split,
                BM=64,
                BN=128,
                BK=128,
                GROUP_M=1,
                num_warps=4,
                num_stages=3,
            )
        else:
            g = pts
            kernels._w4a16.fn[(triton.cdiv(m, 64) * triton.cdiv(n, 64), split)](
                x,
                q,
                s,
                pts,
                partial,
                m,
                n,
                k,
                split,
                BM=64,
                BN=64,
                BK=64,
                GROUP_M=1,
                num_warps=4,
                num_stages=3,
            )
        kernels._reduce_partials[(triton.cdiv(m * n, 256),)](
            partial, g, pts, y, m, n, split, mode == "w4a4", 256
        )
        torch.cuda.synchronize()
        assert bool((y == 512).all())


def test_embedding():
    w = weight(97, 192)
    ids = torch.tensor([[1, 3, 2, 1], [96, 0, 51, 72]], device="cuda")
    torch.testing.assert_close(
        nvfp4_embedding(ids, w.qdata, w.scale, w.per_tensor_scale, w.dtype),
        w.dequantize()[ids], rtol=0, atol=0,
    )


