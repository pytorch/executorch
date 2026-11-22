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


from executorch.backends.cuda.tests.nvfp4_test_utils import dynamic_rows, shards


@pytest.fixture(autouse=True)
def default_policy(monkeypatch):
    for name in (
        "ET_CUDA_NVFP4_ACT",
        "ET_CUDA_NVFP4_PREFILL_OUTPUT",
        "ET_CUDA_NVFP4_PREFILL_IMPL",
        "ET_CUDA_NVFP4_SMALL_M",
    ):
        monkeypatch.delenv(name, raising=False)


def test_cpu_fallback_matches_dequantized_weight(monkeypatch):
    from executorch.backends.cuda.quantize_op_dispatch import (
        _gemm_family_dispatch,
        nvfp4_dispatch,
    )

    # Dequantize the 97-row weight in 32-row chunks, as for a huge lm_head.
    monkeypatch.setattr(_gemm_family_dispatch, "_DEQUANT_N_THRESHOLD", 16)
    monkeypatch.setattr(_gemm_family_dispatch, "_DEQUANT_N_CHUNK", 32)
    chunks = []
    weight_rows = nvfp4_dispatch._weight_rows

    def record_chunk(w, rows):
        chunks.append((rows.start, rows.stop))
        return weight_rows(w, rows)

    w = CudaNVFP4Tensor.fuse_along_output(shards(192))
    ids = torch.tensor([[0, 40], [96, 5]])
    assert torch.equal(F.embedding(ids, w), F.embedding(ids, w.dequantize()))
    for invalid in (-1, 97):
        with pytest.raises(IndexError, match="out of range for 97 rows"):
            F.embedding(torch.tensor([0, invalid]), w)
    x = torch.randn(5, 192, dtype=torch.bfloat16)
    with monkeypatch.context() as patch:
        patch.setattr(nvfp4_dispatch, "_weight_rows", record_chunk)
        assert torch.equal(F.linear(x, w), F.linear(x, w.dequantize()))
    assert chunks == [(0, 32), (32, 64), (64, 96), (96, 97)]
    monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_OUTPUT", "fp32")
    out = F.linear(x, w)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, F.linear(x.float(), w.dequantize(torch.float32)))
    # Decode (M <= 4) keeps the activation dtype under the FP32 policy.
    assert F.linear(x[:2], w).dtype == torch.bfloat16


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
    "mode,native,rows,fp32,impl,expected",
    [
        ("w4a16", False, 37, False, "auto", "nvfp4_w4a16"),
        ("w4a16", False, 1, False, "auto", "nvfp4_gemv"),
        # Without native FP4, auto dequantizes only above the M <= 64 bucket.
        ("auto", False, 37, False, "auto", "nvfp4_w4a16"),
        ("auto", False, 65, False, "auto", "nvfp4_dequant"),
        ("auto", False, 65, False, "kernel", "nvfp4_w4a16"),
        ("w4a16", False, 37, False, "dequant", "nvfp4_dequant"),
        # The FP32 prefill policy and w4a4 keep the fused kernels.
        ("auto", False, 65, True, "auto", "nvfp4_w4a16"),
        ("w4a4", True, 65, False, "dequant", "nvfp4_w4a4"),
        # Without native FP4, auto decode times W4A16 against W4A8.
        ("auto", False, 1, False, "dequant", "nvfp4_auto_decode"),
        ("w4a8", False, 1, False, "auto", "nvfp4_w4a8_gemv"),
        ("w4a8", True, 1, False, "auto", "nvfp4_w4a8_gemv"),
        ("w4a8", True, 37, True, "auto", "nvfp4_w4a16"),
        ("w4a8", False, 65, False, "auto", "nvfp4_dequant"),
        ("auto", True, 37, False, "auto", "nvfp4_auto_prefill"),
        ("auto", True, 65, False, "auto", "nvfp4_auto_prefill"),
        ("auto", True, 37, True, "auto", "nvfp4_auto_prefill"),
        ("auto", True, 1, True, "auto", "nvfp4_auto_decode"),
        ("w4a4", True, 37, False, "auto", "nvfp4_w4a4"),
        ("w4a4", True, 1, True, "auto", "nvfp4_w4a4"),
        # A dynamic M is never statically small: prefill. Bounded by 64 it
        # keeps the fused kernels without native FP4.
        ("auto", True, ("dynamic", 64), True, "auto", "nvfp4_auto_prefill"),
        ("w4a16", False, ("dynamic", 64), False, "auto", "nvfp4_w4a16"),
        ("auto", False, ("dynamic", 4096), False, "auto", "nvfp4_dequant"),
    ],
)
def test_linear_routing(mode, native, rows, fp32, impl, expected, monkeypatch):
    # Export traces with fake tensors; record which op each policy emits.
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch
    from executorch.backends.cuda.triton.kernels import (
        nvfp4_dequant,
        nvfp4_quantized_gemm,
        nvfp4_w4a8_gemv,
    )

    def dequant(qdata, scale, per_tensor_scale, dtype):
        calls.append(("nvfp4_dequant", None))
        return qdata.new_empty((qdata.shape[0], qdata.shape[1] * 2), dtype=dtype)

    monkeypatch.setattr(nvfp4_dequant, "nvfp4_dequant", dequant)
    monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_IMPL", impl)

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
    record_calls(monkeypatch, nvfp4_w4a8_gemv, ("nvfp4_w4a8_gemv",), calls)
    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: native)
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_OUTPUT", "fp32" if fp32 else "input")
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    w = shards(192)[0]
    fake = FakeTensorMode(shape_env=ShapeEnv())
    static = isinstance(rows, int)
    x = torch.randn(rows if static else 17, 192, dtype=torch.bfloat16)
    x = (
        fake.from_tensor(x, static_shapes=True)
        if static
        else dynamic_rows(fake, x, rows[1])
    )
    weight = fake.from_tensor(w)
    with fake:
        out = F.linear(x, weight)
    # Prefill (M not statically <= 4) takes the FP32 output policy; decode
    # keeps the activation dtype.
    output_dtype = torch.float32 if fp32 and (not static or rows > 4) else None
    assert calls == [(expected, output_dtype)]
    assert out.shape == (x.shape[0], w.shape[0])
    assert out.dtype == (output_dtype or torch.bfloat16)


@pytest.mark.parametrize(
    "mode,native,rows,env,expected",
    [
        ("auto", True, 8, {}, (8, False)),
        ("auto", False, 2, {}, (2, False)),
        # A forced weight-only mode never times FP4 activations.
        ("w4a16", True, 8, {}, (8, True)),
        ("w4a16", True, 16, {}, (16, True)),
        ("w4a8", False, 2, {}, None),
        ("auto", True, 1, {}, None),
        ("auto", True, 17, {}, None),
        ("auto", True, 8, {"ET_CUDA_NVFP4_SMALL_M": "0"}, None),
        # The FP32 prefill policy and an explicit dequant keep M 5..16 off the
        # buckets; decode-size M keeps them.
        ("auto", True, 8, {"ET_CUDA_NVFP4_PREFILL_OUTPUT": "fp32"}, None),
        ("auto", True, 8, {"ET_CUDA_NVFP4_PREFILL_IMPL": "dequant"}, None),
        ("auto", False, 4, {"ET_CUDA_NVFP4_PREFILL_IMPL": "dequant"}, (4, False)),
        # With native FP4, decode-size auto keeps nvfp4_auto_decode.
        ("auto", True, 4, {}, None),
        ("auto", True, 2, {}, None),
        # A dynamic M bounded by a bucket takes it; a wider bound does not.
        ("auto", True, ("dynamic", 16), {}, (16, False)),
        ("auto", True, ("dynamic", 64), {}, None),
    ],
)
def test_small_m_routing(mode, native, rows, env, expected, monkeypatch):
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch
    from executorch.backends.cuda.triton.kernels import nvfp4_small_m_gemm

    calls = []

    def op(bucket):
        def run(x, qdata, *args):
            calls.append((bucket, args[-1]))
            return x.new_empty((x.shape[0], qdata.shape[0]))

        return run

    monkeypatch.setattr(nvfp4_small_m_gemm.NVFP4_SMALL_M_GEMM, "op", op)
    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: native)
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    fake = FakeTensorMode(shape_env=ShapeEnv())
    static = isinstance(rows, int)
    x = torch.randn(rows if static else 3, 192, dtype=torch.bfloat16)
    x = (
        fake.from_tensor(x, static_shapes=True)
        if static
        else dynamic_rows(fake, x, rows[1])
    )
    weight = fake.from_tensor(shards(192)[0], static_shapes=True)
    with fake:
        F.linear(x, weight)
    assert calls == ([expected] if expected else [])


@pytest.mark.parametrize(
    "act,output,native,error,match",
    [
        ("w4a4", "input", False, RuntimeError, "SM120"),
        ("w8a8", "input", True, ValueError, "ET_CUDA_NVFP4_ACT"),
        ("auto", "fp16", True, ValueError, "ET_CUDA_NVFP4_PREFILL_OUTPUT"),
        # The W4A8 GEMV is NVIDIA PTX: forced w4a8 fails clearly on ROCm.
        ("w4a8", "rocm", False, RuntimeError, "NVIDIA"),
        # An invalid PREFILL_IMPL raises even where it would not be read.
        ("w4a4", "bad_impl", True, ValueError, "ET_CUDA_NVFP4_PREFILL_IMPL"),
    ],
)
def test_linear_policy_errors(act, output, native, error, match, monkeypatch):
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch

    if output == "rocm":
        monkeypatch.setattr(torch.version, "hip", "6.2")
        output = "input"
    if output == "bad_impl":
        monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_IMPL", "dequnat")
        output = "input"
    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: native)
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", act)
    monkeypatch.setenv("ET_CUDA_NVFP4_PREFILL_OUTPUT", output)
    fake = FakeTensorMode()
    x = fake.from_tensor(torch.randn(37, 192, dtype=torch.bfloat16))
    weight = fake.from_tensor(shards(192)[0])
    with fake, pytest.raises(error, match=match):
        F.linear(x, weight)


@pytest.mark.parametrize("fp4", [False, True])
def test_auto_bodies_forward_mode_split(fp4, monkeypatch):
    # The real _prefill/_decode bodies run; only the kernels they call record.
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch
    from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm

    calls = []

    def recorder(name):
        def op(x, qdata, *args, **kwargs):
            # Positional args after the weight scales: W4A4's act_scale, M bound.
            calls.append((name, args[2:], kwargs))
            dtype = kwargs.get("output_dtype") or x.dtype
            return x.new_empty((x.shape[0], qdata.shape[0]), dtype=dtype)

        return op

    for name in ("nvfp4_gemv", "nvfp4_w4a16", "_launch_w4a4_impl"):
        monkeypatch.setattr(nvfp4_quantized_gemm, name, recorder(name))
    monkeypatch.setattr(nvfp4_quantized_gemm, "_split_workspace_limit", lambda: 2**40)
    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: True)
    w = shards(512)[0]
    weight = (w.qdata, w.scale, w.per_tensor_scale)
    split = 4
    mode_split = int(fp4) * nvfp4_dispatch._MODE_STRIDE + split
    prefill_op = "_launch_w4a4_impl" if fp4 else "nvfp4_w4a16"
    decode_op = "_launch_w4a4_impl" if fp4 else "nvfp4_gemv"

    x = torch.randn(37, 512, dtype=torch.bfloat16)
    act_scale = torch.full((1, 1), 0.5)
    out = nvfp4_dispatch._prefill(
        x, *weight, act_scale, 64, mode_split=mode_split, output_dtype=torch.float32
    )
    assert calls.pop() == (
        prefill_op,
        (act_scale, 64) if fp4 else (),
        {"split_k": split, "output_dtype": torch.float32},
    )
    assert out.dtype == torch.float32
    nvfp4_dispatch._decode(x[:2], *weight, act_scale, True, mode_split=mode_split)
    expected = {"split_k": split} if fp4 else {"split_k": split, "unit_scale": True}
    assert calls.pop() == (decode_op, (act_scale, 2) if fp4 else (), expected)
    # An unbounded M (None) rejects split-K before any kernel is called.
    with pytest.raises(InvalidLaunchParam, match="bounded M"):
        nvfp4_dispatch._prefill(x, *weight, act_scale, None, mode_split=mode_split)
    assert not calls
    if fp4:
        # Without native FP4, timing skips the W4A4 candidates.
        monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: False)
        with pytest.raises(InvalidLaunchParam, match="native FP4"):
            nvfp4_dispatch._prefill(x, *weight, act_scale, 64, mode_split=mode_split)
        with pytest.raises(InvalidLaunchParam, match="native FP4"):
            nvfp4_dispatch._decode(x[:2], *weight, act_scale, mode_split=mode_split)
        assert not calls


@pytest.mark.parametrize(
    "native,hip,runs", [(False, None, True), (True, None, False), (False, "6.2", False)]
)
def test_auto_decode_w4a8_mode_only_on_nvidia_without_native_fp4(
    native, hip, runs, monkeypatch
):
    # Mode 2 (W4A8, NVIDIA PTX) is timed only without native FP4 and off ROCm.
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch
    from executorch.backends.cuda.triton.kernels import nvfp4_w4a8_gemv

    calls = []

    def launch(x, qdata, scale, pts, max_rows, *, split_k):
        calls.append(split_k)
        return x.new_empty((x.shape[0], qdata.shape[0]))

    monkeypatch.setattr(nvfp4_w4a8_gemv, "launch_w4a8_gemv", launch)
    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: native)
    monkeypatch.setattr(torch.version, "hip", hip)
    w = shards(512)[0]
    x = torch.randn(2, 512, dtype=torch.bfloat16)
    mode_split = 2 * nvfp4_dispatch._MODE_STRIDE + 4
    args = (x, w.qdata, w.scale, w.per_tensor_scale, None)
    if runs:
        nvfp4_dispatch._decode(*args, mode_split=mode_split)
        assert calls == [4]
    else:
        with pytest.raises(InvalidLaunchParam, match="W4A8"):
            nvfp4_dispatch._decode(*args, mode_split=mode_split)
        assert not calls


@CUDA
def test_auto_prefill_tunes_with_a_static_scale():
    # Timing runs W4A4 candidates on random stand-ins, including the scale;
    # without native FP4 it skips them.
    from executorch.backends.cuda.autotune.launch_params import (
        autotune_launch_params,
        clear_launch_param_cache,
    )
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch

    w = shards(512, device="cuda")[0]
    x = torch.randn(37, 512, device="cuda", dtype=torch.bfloat16)
    act = torch.full((1, 1), 0.002, device="cuda")
    clear_launch_param_cache()
    with autotune_launch_params() as stats:
        out = nvfp4_dispatch.nvfp4_auto_prefill(
            x, w.qdata, w.scale, w.per_tensor_scale, act
        )
    (pick,) = [p for p in stats.picks if p.param == "mode_split"]
    fp4 = {c // nvfp4_dispatch._MODE_STRIDE for c in pick.times_us}
    assert fp4 == ({0, 1} if _has_native_fp4() else {0})
    assert out.shape == (37, w.shape[0]) and bool(out.isfinite().all())


@pytest.mark.parametrize(
    "mode,rows,op",
    [
        ("w4a4", 37, "nvfp4_w4a4"),
        ("w4a4", 1, "nvfp4_w4a4"),
        ("auto", 37, "nvfp4_auto_prefill"),
        ("auto", 1, "nvfp4_auto_decode"),
    ],
)
def test_calibrated_activation_scale_reaches_w4a4(mode, rows, op, monkeypatch):
    # A ModelOpt/GGUF input_scale must reach every op that can run W4A4.
    from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch
    from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm

    calls = []

    def recorder(name):
        def fake_op(x, qdata, scale, per_tensor_scale, act_scale, *args, **kwargs):
            calls.append((name, act_scale))
            dtype = kwargs.get("output_dtype") or x.dtype
            return x.new_empty((x.shape[0], qdata.shape[0]), dtype=dtype)

        return fake_op

    monkeypatch.setattr(nvfp4_quantized_gemm, "nvfp4_w4a4", recorder("nvfp4_w4a4"))
    for name in ("nvfp4_auto_prefill", "nvfp4_auto_decode"):
        monkeypatch.setattr(nvfp4_dispatch, name, recorder(name))
    monkeypatch.setattr(nvfp4_dispatch, "_has_native_fp4", lambda: True)
    monkeypatch.setenv("ET_CUDA_NVFP4_ACT", mode)
    w = shards(192)[0]
    w = CudaNVFP4Tensor(
        w.qdata, w.scale, w.per_tensor_scale, act_per_tensor_scale=0.002
    )
    fake = FakeTensorMode()
    x = fake.from_tensor(torch.randn(rows, 192, dtype=torch.bfloat16))
    weight = fake.from_tensor(w)
    with fake:
        F.linear(x, weight)
    assert [name for name, _ in calls] == [op]
    assert calls[0][1] is weight.act_per_tensor_scale


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
    out = nvfp4_dispatch.nvfp4_auto_prefill(
        x, *weight, None, output_dtype=torch.float32
    )
    assert out.dtype == torch.float32
    nvfp4_dispatch.nvfp4_auto_decode(x[:2], *weight, None, False)
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
