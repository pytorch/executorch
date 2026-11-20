# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Output-channel fusion with distinct original NVFP4 global scales."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    nvfp4_embedding,
    nvfp4_gemv,
    nvfp4_quantize_activation,
    nvfp4_w4a16,
    nvfp4_w4a4,
)

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="CUDA with native E4M3 (SM89+) required",
)


from executorch.backends.cuda.tests.nvfp4_test_utils import shards


@pytest.mark.parametrize("positional", [False, True])
def test_embedding_rejects_max_norm_without_mutating_weights(positional):
    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401

    w = shards(192)[0]
    before = (w.qdata.clone(), w.scale.clone(), w.per_tensor_scale.clone())
    ids = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="max_norm"):
        if positional:
            F.embedding(ids, w, None, 1.0)
        else:
            F.embedding(ids, w, max_norm=1.0)
    for actual, original in zip((w.qdata, w.scale, w.per_tensor_scale), before):
        assert torch.equal(actual, original)


@CUDA
@pytest.mark.parametrize("row_scales", [False, True])
def test_embedding_bf16_scale_rounding_boundary(row_scales):
    q = torch.full((3, 8), 0x33, dtype=torch.uint8, device="cuda")
    s = torch.full((3, 1), 0x0D, dtype=torch.uint8, device="cuda")
    pts = torch.tensor(
        [0x3E341A41] * (3 if row_scales else 1), dtype=torch.int32, device="cuda"
    ).view(torch.float32)
    w = CudaNVFP4Tensor(q, s, pts)
    ids = torch.tensor([2, 0, 1], device="cuda")
    actual = nvfp4_embedding(ids, q, s, pts, torch.bfloat16)
    expected = w.dequantize()[ids]
    assert torch.equal(
        expected.view(torch.int16), torch.full_like(expected.view(torch.int16), 0x3BDC)
    )
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@CUDA
@pytest.mark.parametrize("split", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("m", [1, 4])
def test_row_scale_gemv(m, split):
    w = CudaNVFP4Tensor.fuse_along_output(shards(2176, "cuda"))
    x = torch.randn(m, 2176, device="cuda", dtype=torch.bfloat16)
    expected = (x.float() @ w.dequantize().float().T).to(x.dtype)
    actual = nvfp4_gemv(x, w.qdata, w.scale, w.per_tensor_scale, split_k=split)
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.015)


@CUDA
@pytest.mark.parametrize("split", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("mode", ["w4a4", "w4a16"])
def test_row_scale_gemm(mode, split):
    if mode == "w4a4" and torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("SM120 required")
    k = 4224 if split == 32 else 2176
    w = CudaNVFP4Tensor.fuse_along_output(shards(k, "cuda"))
    x = torch.randn(17, k, device="cuda", dtype=torch.bfloat16)
    if mode == "w4a4":
        q, s, g = nvfp4_quantize_activation(x)
        a = (
            CudaNVFP4Tensor(
                q, s, torch.ones(1, device="cuda"), orig_dtype=torch.float32
            ).dequantize()
            * g[:, None]
        )
        expected = (a @ w.dequantize(torch.float32).T).to(x.dtype)
        actual = nvfp4_w4a4(x, w.qdata, w.scale, w.per_tensor_scale, split_k=split)
    else:
        expected = (x.float() @ w.dequantize().float().T).to(x.dtype)
        actual = nvfp4_w4a16(x, w.qdata, w.scale, w.per_tensor_scale, split_k=split)
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.02)


@CUDA
def test_row_scale_embedding():
    w = CudaNVFP4Tensor.fuse_along_output(shards(192, "cuda"))
    w.per_tensor_scale[31] = 0.0
    ids = torch.tensor([[0, 30, 31, 49, 50, 96]], device="cuda")
    actual = nvfp4_embedding(ids, w.qdata, w.scale, w.per_tensor_scale, w.dtype)
    torch.testing.assert_close(actual, w.dequantize()[ids], rtol=0, atol=0)


class FusedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", CudaNVFP4Tensor.fuse_along_output(shards(192)))
        self.register_buffer("bias", torch.randn(97, dtype=torch.bfloat16))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


@CUDA
@pytest.mark.parametrize("mode,m,dynamic", [("w4a16", 1, False), ("w4a4", 16, True)])
def test_fused_row_scale_pte(mode, m, dynamic, monkeypatch):
    if mode == "w4a4" and torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("SM120 required")
    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.runtime import Runtime

    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    module = FusedLinear().eval()
    x = torch.randn(m, 192, dtype=torch.bfloat16)
    shapes = ({0: torch.export.Dim("tokens", min=1, max=64)},) if dynamic else None
    exported = torch.export.export(module, (x,), dynamic_shapes=shapes, strict=True)
    lowered = to_edge_transform_and_lower(
        exported,
        partitioner=[
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "fused.pte"
        with path.open("wb") as output:
            lowered.write_to_file(output)
        lowered.write_tensor_data_to_file(folder)
        data = next(Path(folder).glob("*.ptd"))
        method = (
            Runtime.get()
            .load_program(str(path), data_path=str(data))
            .load_method("forward")
        )
        for rows in (1, 5, 37, 64) if dynamic else (m,):
            value = torch.randn(rows, 192, dtype=x.dtype)
            actual = method.execute([value])[0].cpu()
            if mode == "w4a4":
                q, s, g = nvfp4_quantize_activation(value.cuda())
                a = (
                    CudaNVFP4Tensor(
                        q, s, torch.ones(1, device="cuda"), orig_dtype=torch.float32
                    ).dequantize()
                    * g[:, None]
                )
                expected = (a.cpu() @ module.weight.dequantize(torch.float32).T).to(
                    x.dtype
                ) + module.bias
            else:
                expected = F.linear(value, module.weight.dequantize(), module.bias)
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
