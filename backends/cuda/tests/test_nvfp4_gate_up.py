# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Fused NVFP4 gate/up: the gated-linear op, its epilogue, and serialized
programs for every split and row scales."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch.nvfp4_dispatch import (
    _has_native_fp4,
    nvfp4_gated_linear,
)
from executorch.backends.cuda.tests.nvfp4_test_utils import shards
from executorch.backends.cuda.triton.kernels.nvfp4_gated_gemm import (
    nvfp4_gate_up_epilogue,
    nvfp4_w4a4_gate_up,
)
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import nvfp4_w4a4
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch._subclasses.fake_tensor import FakeTensorMode

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
NATIVE_FP4 = pytest.mark.skipif(not _has_native_fp4(), reason="native FP4 required")


def gated_reference(x, weight):
    projected = x.float() @ weight.dequantize(torch.float32).T
    gate, up = projected.chunk(2, dim=-1)
    return (F.silu(gate) * up).to(x.dtype)


def test_fake_gated_linear_shape_dtype_and_errors():
    with FakeTensorMode():
        x = torch.empty(3, 5, 64, dtype=torch.bfloat16)
        out = nvfp4_gated_linear(x, torch.empty(14, 64, dtype=torch.bfloat16))
        assert out.shape == (3, 5, 7) and out.dtype == torch.bfloat16
        for bad in (torch.empty(13, 64), torch.empty(14, 32)):
            with pytest.raises(ValueError, match="even N and matching K"):
                nvfp4_gated_linear(x, bad.to(torch.bfloat16))


def test_gated_linear_rejects_odd_n_and_k_mismatch():
    odd = shards(192)[0]
    x = torch.randn(2, 192, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="even N and matching K"):
        nvfp4_gated_linear(x, odd)
    fused = CudaNVFP4Tensor.fuse_along_output(shards(192)[:2])
    with pytest.raises(ValueError, match="even N and matching K"):
        nvfp4_gated_linear(x[:, :64], fused)
    with pytest.raises(ValueError, match="even"):
        nvfp4_gate_up_epilogue(torch.empty(2, 7))


def test_gated_linear_cpu_fallback():
    weight = CudaNVFP4Tensor.fuse_along_output(shards(192)[:2])
    x = torch.randn(5, 192, dtype=torch.bfloat16)
    torch.testing.assert_close(
        nvfp4_gated_linear(x, weight), gated_reference(x, weight), rtol=0.02, atol=0.02
    )


@CUDA
def test_gate_up_epilogue():
    x = torch.randn(5, 2 * 70, device="cuda", dtype=torch.bfloat16)
    gate, up = x.float().chunk(2, dim=1)
    torch.testing.assert_close(
        nvfp4_gate_up_epilogue(x), (F.silu(gate) * up).to(x.dtype)
    )


@CUDA
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9),
    reason="native E4M3 (SM89+) required",
)
@pytest.mark.parametrize("m", [1, 37])
def test_gated_linear_weight_only(m, monkeypatch):
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", "w4a16")
    weight = CudaNVFP4Tensor.fuse_along_output(shards(192, device="cuda")[:2])
    x = torch.randn(m, 192, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        nvfp4_gated_linear(x, weight), gated_reference(x, weight), rtol=0.02, atol=0.02
    )


class GateUp(torch.nn.Module):
    def __init__(self, rows, split):
        super().__init__()
        torch.manual_seed(42)
        self.split = split
        self.register_buffer("q", torch.randint(0, 256, (70, 2112), dtype=torch.uint8))
        self.register_buffer(
            "s", torch.full((70, 264), 0.0625).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        self.register_buffer(
            "pts", torch.linspace(0.1, 0.3, 70) if rows else torch.tensor([0.25])
        )

    def forward(self, x):
        return nvfp4_w4a4_gate_up(x, self.q, self.s, self.pts, split_k=self.split)


@CUDA
@NATIVE_FP4
@pytest.mark.parametrize("row_scales", [False, True])
@pytest.mark.parametrize("split", [1, 2, 4, 8, 16, 32])
def test_gate_up_split_k_pte(row_scales, split):
    module = GateUp(row_scales, split).eval()
    x = torch.randn(17, 4224, dtype=torch.bfloat16)
    exported = torch.export.export(
        module,
        (x,),
        dynamic_shapes=({0: torch.export.Dim("tokens", min=1, max=64)},),
        strict=True,
    )
    lowered = to_edge_transform_and_lower(
        exported,
        partitioner=[
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "gated.pte"
        with path.open("wb") as output:
            lowered.write_to_file(output)
        lowered.write_tensor_data_to_file(folder)
        data = next(Path(folder).glob("*.ptd"))
        from executorch.export.utils import is_fbcode

        if is_fbcode():
            # The in-process runtime in fbcode has no CUDA backend.
            return
        from executorch.runtime import Runtime

        method = (
            Runtime.get()
            .load_program(str(path), data_path=str(data))
            .load_method("forward")
        )
        for rows in (1, 5, 17, 37, 64):
            value = torch.randn(rows, 4224, dtype=x.dtype)
            linear = nvfp4_w4a4(
                value.cuda(),
                module.q.cuda(),
                module.s.cuda(),
                module.pts.cuda(),
                split_k=split,
            )
            gate, up = linear.chunk(2, dim=1)
            expected = torch.nn.functional.silu(gate) * up
            actual = method.execute([value])[0].cpu()
            # Both GEMMs round to BF16 before SiLU; K accumulation may differ
            # between independently selected tiles.
            torch.testing.assert_close(actual, expected.cpu(), rtol=0.03, atol=0.04)
