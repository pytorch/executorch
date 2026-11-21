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
from executorch.backends.cuda.autotune.launch_params import InvalidLaunchParam
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch.nvfp4_dispatch import _has_native_fp4
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    nvfp4_embedding,
    nvfp4_gemv,
    nvfp4_quantize_activation,
    nvfp4_w4a16,
    nvfp4_w4a4,
)
from torch._subclasses.fake_tensor import FakeTensorMode

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


from executorch.backends.cuda.tests.nvfp4_test_utils import shards


def test_cpu_fallback_matches_dequantized_weight(monkeypatch):
    from executorch.backends.cuda.quantize_op_dispatch import _gemm_family_dispatch

    # Dequantize the 97-row weight in 32-row chunks, as for a huge lm_head.
    monkeypatch.setattr(_gemm_family_dispatch, "_DEQUANT_N_THRESHOLD", 16)
    monkeypatch.setattr(_gemm_family_dispatch, "_DEQUANT_N_CHUNK", 32)

    w = CudaNVFP4Tensor.fuse_along_output(shards(192))
    ids = torch.tensor([[0, 40], [96, 5]])
    assert torch.equal(F.embedding(ids, w), F.embedding(ids, w.dequantize()))
    x = torch.randn(5, 192, dtype=torch.bfloat16)
    assert torch.equal(F.linear(x, w), F.linear(x, w.dequantize()))
    monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_OUTPUT", "fp32")
    out = F.linear(x, w)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, F.linear(x.float(), w.dequantize(torch.float32)))


def record_calls(monkeypatch, module, names, calls):
    def recorder(name):
        def op(x, qdata, *args, output_dtype=None, **kwargs):
            calls.append((name, output_dtype))
            dtype = output_dtype or x.dtype
            return x.new_empty((x.shape[0], qdata.shape[0]), dtype=dtype)

        return op

    for name in names:
        monkeypatch.setattr(module, name, recorder(name))


@pytest.mark.parametrize(
    "mode,native,rows,fp32,expected",
    [
        ("w4a16", False, 37, False, "nvfp4_w4a16"),
        ("w4a16", False, 1, False, "nvfp4_gemv"),
        ("auto", False, 37, False, "nvfp4_w4a16"),
        ("auto", True, 37, False, "nvfp4_auto_prefill"),
        ("auto", True, 37, True, "nvfp4_auto_prefill"),
        ("auto", True, 1, True, "nvfp4_auto_decode"),
        ("w4a4", True, 37, False, "nvfp4_w4a4"),
        # A dynamic M (bounded by 64) is never statically small: prefill.
        ("auto", True, None, True, "nvfp4_auto_prefill"),
        ("w4a16", False, None, False, "nvfp4_w4a16"),
    ],
)
def test_linear_routing(mode, native, rows, fp32, expected, monkeypatch):
    # Export traces with fake tensors; record which op each policy emits.
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch
    from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm

    calls = []
    record_calls(
        monkeypatch,
        nvfp4_quantized_gemm,
        ("nvfp4_gemv", "nvfp4_w4a16", "nvfp4_w4a4"),
        calls,
    )
    record_calls(
        monkeypatch,
        nvfp4_dispatch,
        ("nvfp4_auto_decode", "nvfp4_auto_prefill"),
        calls,
    )
    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: native)
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_OUTPUT", "fp32" if fp32 else "input")
    from executorch.backends.cuda.tests.test_nvfp4_launch_validation import dynamic_rows
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    w = shards(192)[0]
    fake = FakeTensorMode(shape_env=ShapeEnv())
    x = torch.randn(rows or 17, 192, dtype=torch.bfloat16)
    x = fake.from_tensor(x) if rows else dynamic_rows(fake, x, 64)
    weight = fake.from_tensor(w)
    with fake:
        out = F.linear(x, weight)
    # Prefill (M not statically <= 4) takes the FP32 output policy; decode
    # keeps the activation dtype.
    output_dtype = torch.float32 if fp32 and (rows is None or rows > 4) else None
    assert calls == [(expected, output_dtype)]
    assert out.shape == (x.shape[0], w.shape[0])
    assert out.dtype == (output_dtype or torch.bfloat16)


@pytest.mark.parametrize("fp4", [False, True])
def test_auto_bodies_forward_mode_split(fp4, monkeypatch):
    # The real _prefill/_decode bodies run; only the kernels they call record.
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch
    from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm

    calls = []

    def recorder(name):
        def op(x, qdata, *args, **kwargs):
            calls.append((name, kwargs))
            dtype = kwargs.get("output_dtype") or x.dtype
            return x.new_empty((x.shape[0], qdata.shape[0]), dtype=dtype)

        return op

    for name in ("nvfp4_gemv", "nvfp4_w4a16", "nvfp4_w4a4"):
        monkeypatch.setattr(nvfp4_quantized_gemm, name, recorder(name))
    monkeypatch.setattr(nvfp4_quantized_gemm, "_split_workspace_limit", lambda: 2**40)
    w = shards(512)[0]
    weight = (w.qdata, w.scale, w.per_tensor_scale)
    split = 4
    mode_split = int(fp4) * nvfp4_dispatch._MODE_STRIDE + split
    prefill_op = "nvfp4_w4a4" if fp4 else "nvfp4_w4a16"
    decode_op = "nvfp4_w4a4" if fp4 else "nvfp4_gemv"

    x = torch.randn(37, 512, dtype=torch.bfloat16)
    out = nvfp4_dispatch._prefill(
        x, *weight, 0.5, 64, mode_split=mode_split, output_dtype=torch.float32
    )
    assert calls.pop() == (
        prefill_op,
        {"split_k": split, "output_dtype": torch.float32},
    )
    assert out.dtype == torch.float32
    nvfp4_dispatch._decode(x[:2], *weight, 0.5, True, mode_split=mode_split)
    expected = {"split_k": split} if fp4 else {"split_k": split, "unit_scale": True}
    assert calls.pop() == (decode_op, expected)
    # An unbounded M (None) rejects split-K before any kernel is called.
    with pytest.raises(InvalidLaunchParam, match="bounded M"):
        nvfp4_dispatch._prefill(x, *weight, 0.5, None, mode_split=mode_split)
    assert not calls


def test_auto_ops_run_their_first_candidate_outside_tuning(monkeypatch):
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch
    from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm

    calls = []
    record_calls(
        monkeypatch, nvfp4_quantized_gemm, ("nvfp4_gemv", "nvfp4_w4a16"), calls
    )
    w = shards(512)[0]
    weight = (w.qdata, w.scale, w.per_tensor_scale)
    x = torch.randn(37, 512, dtype=torch.bfloat16)
    out = nvfp4_dispatch.nvfp4_auto_prefill(x, *weight, 0.0, output_dtype=torch.float32)
    assert out.dtype == torch.float32
    nvfp4_dispatch.nvfp4_auto_decode(x[:2], *weight, 0.0, False)
    assert calls == [("nvfp4_w4a16", torch.float32), ("nvfp4_gemv", None)]


def test_embedding_routing(monkeypatch):
    from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm

    calls = []

    def embedding(indices, qdata, scale, per_tensor_scale, output_dtype):
        calls.append(tuple(indices.shape))
        return indices.new_empty((*indices.shape, qdata.shape[1] * 2)).to(output_dtype)

    monkeypatch.setattr(nvfp4_quantized_gemm, "nvfp4_embedding", embedding)
    w = shards(192)[0]
    with FakeTensorMode() as fake:
        ids = fake.from_tensor(torch.tensor([[0, 3]]))
        weight = fake.from_tensor(w)
        outputs = (
            F.embedding(ids, weight),
            torch.ops.aten.embedding.default(weight, ids),
        )
    assert calls == [(1, 2), (1, 2)]
    assert all(o.shape == (1, 2, 192) and o.dtype == w.dtype for o in outputs)


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
    if mode == "w4a4" and not _has_native_fp4():
        pytest.skip("native FP4 required")
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
    if mode == "w4a4" and not _has_native_fp4():
        pytest.skip("native FP4 required")
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
        from executorch.export.utils import is_fbcode

        if is_fbcode():
            # The in-process runtime in fbcode has no CUDA backend.
            return
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
