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

from executorch.backends.cuda.autotune.launch_params import autotune_launch_params
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch.nvfp4_dispatch import (
    _has_native_fp4,
    _MODE_STRIDE,
    nvfp4_auto_gate_up_prefill,
    nvfp4_gated_linear,
)
from executorch.backends.cuda.tests.nvfp4_test_utils import shards
from executorch.backends.cuda.triton.kernels.nvfp4_gated_gemm import (
    nvfp4_gate_up_epilogue,
    nvfp4_w4a4_gate_up,
)
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    nvfp4_quantize_activation,
    nvfp4_w4a4,
)
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


def w4a4_gated_reference(x, weight):
    """W4A4 numerics: activations quantized like the kernel, FP32 output."""
    flat = x.reshape(-1, x.shape[-1])
    q, s, g = nvfp4_quantize_activation(flat, weight.act_per_tensor_scale or 0.0)
    one = torch.ones(1, device=x.device)
    a = CudaNVFP4Tensor(q, s, one, orig_dtype=torch.float32).dequantize()
    gate, up = ((a * g[:, None]) @ weight.dequantize(torch.float32).T).chunk(2, -1)
    return (F.silu(gate) * up).reshape(*x.shape[:-1], -1)


def gate_up_weight(device, act_scale=None):
    # Distinct random block scales and per-row global scales.
    w = CudaNVFP4Tensor.fuse_along_output(shards(192, device=device)[:2])
    return CudaNVFP4Tensor(
        w.qdata, w.scale, w.per_tensor_scale, act_per_tensor_scale=act_scale
    )


def test_plain_weights_are_rejected():
    x = torch.empty(3, 5, 64, dtype=torch.bfloat16)
    weight = torch.empty(14, 64, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="CudaNVFP4Tensor"):
        nvfp4_gated_linear(x, weight)
    with FakeTensorMode() as fake, pytest.raises(TypeError, match="CudaNVFP4Tensor"):
        nvfp4_gated_linear(fake.from_tensor(x), fake.from_tensor(weight))


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
    weight = gate_up_weight("cpu")
    x = torch.randn(2, 5, 192, dtype=torch.bfloat16)
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
    weight = gate_up_weight("cuda")
    x = torch.randn(m, 192, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        nvfp4_gated_linear(x, weight), gated_reference(x, weight), rtol=0.02, atol=0.02
    )


@CUDA
@NATIVE_FP4
@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("shape", [(37, 192), (2, 9, 192)])
def test_gated_linear_fused_w4a4(shape, fp32, monkeypatch):
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", "w4a4")
    monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_OUTPUT", "fp32" if fp32 else "input")
    weight = gate_up_weight("cuda", act_scale=0.002)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    actual = nvfp4_gated_linear(x, weight)
    assert actual.dtype == (torch.float32 if fp32 else torch.bfloat16)
    expected = w4a4_gated_reference(x, weight)
    torch.testing.assert_close(actual.float(), expected, rtol=0.03, atol=0.04)


@CUDA
@NATIVE_FP4
def test_gated_linear_auto_and_autotuned_prefill(monkeypatch):
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", "auto")
    weight = gate_up_weight("cuda")
    x = torch.randn(37, 192, device="cuda", dtype=torch.bfloat16)
    # Outside AOTI tuning, auto runs its first candidate: W4A16 + epilogue.
    torch.testing.assert_close(
        nvfp4_gated_linear(x, weight), gated_reference(x, weight), rtol=0.02, atol=0.02
    )
    args = (x, weight.qdata, weight.scale, weight.per_tensor_scale)
    with autotune_launch_params() as stats:
        tuned = nvfp4_auto_gate_up_prefill(*args)
        split = nvfp4_w4a4_gate_up(*args)
    (mode_pick,) = [p for p in stats.picks if p.param == "mode_split"]
    assert len(mode_pick.times_us) > 1
    expected = (
        w4a4_gated_reference(x, weight)
        if mode_pick.choice // _MODE_STRIDE
        else gated_reference(x, weight)
    )
    torch.testing.assert_close(tuned.float(), expected.float(), rtol=0.03, atol=0.04)
    assert any(p.param == "split_k" for p in stats.picks)
    torch.testing.assert_close(
        split.float(), w4a4_gated_reference(x, weight), rtol=0.03, atol=0.04
    )


class GatedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(gate_up_weight("cpu"), requires_grad=False)

    def forward(self, x):
        return nvfp4_gated_linear(x, self.weight)


@CUDA
@NATIVE_FP4
def test_gated_linear_parameter_pte(monkeypatch):
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", "w4a4")
    module = GatedLinear().eval()
    x = torch.randn(17, 192, dtype=torch.bfloat16)
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
        path = Path(folder) / "gated_linear.pte"
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
        w = module.weight
        weight = CudaNVFP4Tensor(
            w.qdata.cuda(), w.scale.cuda(), w.per_tensor_scale.cuda()
        )
        for rows in (1, 5, 17, 64):
            value = torch.randn(rows, 192, dtype=x.dtype)
            expected = w4a4_gated_reference(value.cuda(), weight).to(x.dtype)
            actual = method.execute([value])[0].cpu()
            torch.testing.assert_close(actual, expected.cpu(), rtol=0.03, atol=0.04)


class GateUp(torch.nn.Module):
    def __init__(self, rows, split):
        super().__init__()
        torch.manual_seed(42)
        self.split = split
        self.register_buffer("q", torch.randint(0, 256, (70, 2112), dtype=torch.uint8))
        # Random block scales catch a gate/up row remap error.
        self.register_buffer(
            "s",
            (0.03 + torch.rand(70, 264) * 0.12)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8),
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
