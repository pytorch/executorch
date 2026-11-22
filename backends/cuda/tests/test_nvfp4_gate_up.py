# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Fused NVFP4 gate/up: the gated-linear op, its epilogue, and serialized
programs for every split and row scales."""

import contextlib
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_params,
    clear_launch_param_cache,
    InvalidLaunchParam,
)
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch
from executorch.backends.cuda.quantize_op_dispatch.nvfp4_dispatch import (
    _has_native_fp4,
    _MODE_STRIDE,
    nvfp4_auto_gate_up_prefill,
    nvfp4_gated_linear,
)
from executorch.backends.cuda.tests.nvfp4_test_utils import shards
from executorch.backends.cuda.triton.kernels import (
    nvfp4_gated_gemm,
    nvfp4_quantized_gemm,
)
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


@pytest.fixture(autouse=True)
def default_policy(monkeypatch):
    for name in ("ET_CUDA_NVFP4_ACT", "ET_CUDA_NVFP4_PREFILL_OUTPUT"):
        monkeypatch.delenv(name, raising=False)


def gated_reference(x, weight):
    projected = x.float() @ weight.dequantize(torch.float32).T
    gate, up = projected.chunk(2, dim=-1)
    return (F.silu(gate) * up).to(x.dtype)


def w4a4_gated_reference(x, weight):
    """W4A4 numerics: activations quantized like the kernel, FP32 output."""
    flat = x.reshape(-1, x.shape[-1])
    q, s, g = nvfp4_quantize_activation(flat, weight.act_per_tensor_scale)
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


def recorder(name, calls, original=None):
    """Records calls as ``name``; calls ``original``, or returns an empty
    result of the right shape when there is none."""

    def op(x, qdata, *args, output_dtype=None, **kwargs):
        calls.append((name, output_dtype))
        if original is not None:
            return original(x, qdata, *args, output_dtype=output_dtype, **kwargs)
        n = qdata.shape[0] // 2 if "gate_up" in name else qdata.shape[0]
        return x.new_empty((x.shape[0], n), dtype=output_dtype or x.dtype)

    return op


@pytest.mark.parametrize(
    "mode,dtype,rows,fp32,native,expected",
    [
        (
            "auto",
            torch.bfloat16,
            37,
            False,
            True,
            ("nvfp4_auto_gate_up_prefill", None),
        ),
        # Without native FP4, auto never reaches the fused W4A4 candidates.
        ("auto", torch.bfloat16, 37, False, False, ("nvfp4_w4a16", None)),
        ("w4a4", torch.bfloat16, 37, False, True, ("nvfp4_w4a4_gate_up", None)),
        ("w4a16", torch.bfloat16, 37, False, True, ("nvfp4_w4a16", None)),
        ("auto", torch.float16, 37, False, True, ("nvfp4_auto_prefill", None)),
        (
            "auto",
            torch.float32,
            37,
            True,
            True,
            ("nvfp4_auto_gate_up_prefill", torch.float32),
        ),
        # Decode-size FP32 under the FP32 policy falls back like _linear,
        # which dequantizes (no kernel call).
        ("auto", torch.float32, 2, True, True, None),
    ],
)
def test_gated_linear_routing(mode, dtype, rows, fp32, native, expected, monkeypatch):
    # Export traces with fake tensors; record which path the op takes.
    calls = []
    for module, names in (
        (nvfp4_gated_gemm, ("nvfp4_w4a4_gate_up",)),
        (nvfp4_quantized_gemm, ("nvfp4_gemv", "nvfp4_w4a16", "nvfp4_w4a4")),
        (
            nvfp4_dispatch,
            ("nvfp4_auto_gate_up_prefill", "nvfp4_auto_prefill", "nvfp4_auto_decode"),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, recorder(name, calls))
    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: native)
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_OUTPUT", "fp32" if fp32 else "input")
    fake = FakeTensorMode()
    x = fake.from_tensor(torch.randn(rows, 192, dtype=dtype))
    weight = fake.from_tensor(gate_up_weight("cpu"))
    with fake:
        out = nvfp4_gated_linear(x, weight)
    assert calls == ([expected] if expected else [])
    assert out.shape == (rows, weight.shape[0] // 2)


@pytest.mark.parametrize(
    "mode,module,op",
    [
        ("w4a4", nvfp4_gated_gemm, "nvfp4_w4a4_gate_up"),
        ("auto", nvfp4_dispatch, "nvfp4_auto_gate_up_prefill"),
    ],
)
def test_calibrated_activation_scale_reaches_gated_ops(mode, module, op, monkeypatch):
    seen = []

    def fake_op(x, qdata, scale, per_tensor_scale, act_scale, output_dtype=None):
        seen.append(act_scale)
        n = qdata.shape[0] // 2
        return x.new_empty((x.shape[0], n), dtype=output_dtype or x.dtype)

    monkeypatch.setattr(module, op, fake_op)
    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: True)
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    monkeypatch.delenv("ET_CUDA_NVFP4_PREFILL_OUTPUT", raising=False)
    fake = FakeTensorMode()
    x = fake.from_tensor(torch.randn(37, 192, dtype=torch.bfloat16))
    weight = fake.from_tensor(gate_up_weight("cpu", act_scale=0.002))
    with fake:
        nvfp4_gated_linear(x, weight)
    assert len(seen) == 1 and seen[0] is weight.act_per_tensor_scale


@pytest.mark.parametrize("fp4", [False, True])
def test_auto_gate_up_body_forwards_mode_split(fp4, monkeypatch):
    # The real _prefill_gate_up body runs; only the kernels it calls record.
    calls, epilogues = [], []

    def recorder(name):
        def op(x, qdata, *args, **kwargs):
            calls.append((name, args[2:], kwargs))
            n = qdata.shape[0] // (2 if fp4 else 1)
            return x.new_empty((x.shape[0], n), dtype=kwargs["output_dtype"])

        return op

    def epilogue(y):
        epilogues.append(y.shape)
        return y[:, : y.shape[1] // 2]

    monkeypatch.setattr(
        nvfp4_gated_gemm, "nvfp4_w4a4_gate_up", recorder("nvfp4_w4a4_gate_up")
    )
    monkeypatch.setattr(nvfp4_gated_gemm, "nvfp4_gate_up_epilogue", epilogue)
    monkeypatch.setattr(nvfp4_quantized_gemm, "nvfp4_w4a16", recorder("nvfp4_w4a16"))
    monkeypatch.setattr(nvfp4_quantized_gemm, "_split_workspace_limit", lambda: 2**40)
    w = gate_up_weight("cpu", act_scale=0.002)
    weight = (w.qdata, w.scale, w.per_tensor_scale, w.act_per_tensor_scale)
    x = torch.randn(37, 192, dtype=torch.bfloat16)
    mode_split = int(fp4) * _MODE_STRIDE + 2
    out = nvfp4_dispatch._prefill_gate_up(
        x, *weight, 64, mode_split=mode_split, output_dtype=torch.float32
    )
    name = "nvfp4_w4a4_gate_up" if fp4 else "nvfp4_w4a16"
    act = (w.act_per_tensor_scale,) if fp4 else ()
    assert calls == [(name, act, {"split_k": 2, "output_dtype": torch.float32})]
    assert epilogues == ([] if fp4 else [(37, w.shape[0])])
    assert out.shape == (37, w.shape[0] // 2) and out.dtype == torch.float32
    calls.clear()
    with pytest.raises(InvalidLaunchParam, match="bounded M"):
        nvfp4_dispatch._prefill_gate_up(x, *weight, None, mode_split=mode_split)
    assert not calls


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_w4a4_gate_up_rejects_invalid_static_scale(value):
    w = gate_up_weight("cpu")
    x = torch.empty(3, 192, dtype=torch.bfloat16)
    static = torch.full((1, 1), value)
    with pytest.raises(ValueError, match="static activation scale"):
        nvfp4_w4a4_gate_up(x, w.qdata, w.scale, w.per_tensor_scale, static)


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
    calls = []
    monkeypatch.setattr(
        nvfp4_gated_gemm,
        "nvfp4_w4a4_gate_up",
        recorder("nvfp4_w4a4_gate_up", calls, nvfp4_w4a4_gate_up),
    )
    actual = nvfp4_gated_linear(x, weight)
    assert [name for name, _ in calls] == ["nvfp4_w4a4_gate_up"]
    assert actual.dtype == (torch.float32 if fp32 else torch.bfloat16)
    expected = w4a4_gated_reference(x, weight)
    torch.testing.assert_close(actual.float(), expected, rtol=0.03, atol=0.04)


@CUDA
@NATIVE_FP4
def test_gated_linear_auto_and_autotuned_prefill(monkeypatch):
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", "auto")
    weight = gate_up_weight("cuda")
    x = torch.randn(37, 192, device="cuda", dtype=torch.bfloat16)
    calls = []
    monkeypatch.setattr(
        nvfp4_dispatch,
        "nvfp4_auto_gate_up_prefill",
        recorder("nvfp4_auto_gate_up_prefill", calls, nvfp4_auto_gate_up_prefill),
    )
    # Outside AOTI tuning, auto runs its first candidate: W4A16 + epilogue.
    torch.testing.assert_close(
        nvfp4_gated_linear(x, weight), gated_reference(x, weight), rtol=0.02, atol=0.02
    )
    assert [name for name, _ in calls] == ["nvfp4_auto_gate_up_prefill"]
    args = (x, weight.qdata, weight.scale, weight.per_tensor_scale)
    clear_launch_param_cache()
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


def test_gated_linear_auto_export_traces_the_auto_op(monkeypatch):
    # CPU: export in the default auto mode as on a native-FP4 GPU.
    calls = []
    monkeypatch.setattr(
        nvfp4_dispatch,
        "nvfp4_auto_gate_up_prefill",
        recorder("nvfp4_auto_gate_up_prefill", calls),
    )
    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: True)
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", "auto")
    torch.export.export(
        GatedLinear().eval(),
        (torch.randn(17, 192, dtype=torch.bfloat16),),
        dynamic_shapes=({0: torch.export.Dim("tokens", min=1, max=64)},),
        strict=True,
    ).run_decompositions()
    assert calls and set(calls) == {("nvfp4_auto_gate_up_prefill", None)}


@CUDA
@NATIVE_FP4
@pytest.mark.parametrize("mode", ["w4a4", "auto"])
def test_gated_linear_parameter_pte(mode, monkeypatch):
    import executorch.backends.cuda.cuda_backend as cuda_backend

    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    calls = []
    if mode == "w4a4":
        monkeypatch.setattr(
            nvfp4_gated_gemm,
            "nvfp4_w4a4_gate_up",
            recorder("nvfp4_w4a4_gate_up", calls, nvfp4_w4a4_gate_up),
        )
    else:
        monkeypatch.setattr(
            nvfp4_dispatch,
            "nvfp4_auto_gate_up_prefill",
            recorder("nvfp4_auto_gate_up_prefill", calls, nvfp4_auto_gate_up_prefill),
        )
    picks = []
    original_context = cuda_backend.autotune_launch_params

    @contextlib.contextmanager
    def capture_tuning(*args, **kwargs):
        with original_context(*args, **kwargs) as stats:
            yield stats
        picks.extend(stats.picks)

    monkeypatch.setattr(cuda_backend, "autotune_launch_params", capture_tuning)
    clear_launch_param_cache()
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
    assert calls, "export must trace the fused gate/up op"
    fp4 = True
    if mode == "auto":
        (pick,) = [p for p in picks if p.param == "mode_split"]
        fp4 = bool(pick.choice // _MODE_STRIDE)
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
        reference = w4a4_gated_reference if fp4 else gated_reference
        for rows in (1, 5, 17, 64):
            value = torch.randn(rows, 192, dtype=x.dtype)
            expected = reference(value.cuda(), weight).to(x.dtype)
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
