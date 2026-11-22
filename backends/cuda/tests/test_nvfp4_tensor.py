# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-only NVFP4 packing, fusion and serialized metadata contracts."""

import io
import math

import pytest
import torch
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.tests.nvfp4_test_utils import (
    portable_weight as weight,
    shards,
)
from executorch.extension.llm.export.nvfp4 import (
    ExportableNVFP4Config,
    ExportableNVFP4Tensor,
)
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torchao.quantization import quantize_


@pytest.mark.parametrize("use_per_tensor_scale", [False, True])
def test_from_exportable_matches_portable_dequantize(use_per_tensor_scale):
    torch.manual_seed(0)
    linear = torch.nn.Linear(96, 48, bias=False, dtype=torch.bfloat16)
    quantize_(linear, ExportableNVFP4Config(use_per_tensor_scale=use_per_tensor_scale))
    w = linear.weight
    packed = CudaNVFP4Tensor.from_exportable(w)
    # A scalar stays a 2-D (1, 1) weight constant (see the class docstring).
    assert w.per_tensor_scale.ndim == 0 and packed.per_tensor_scale.shape == (1, 1)
    assert packed.unit_scale is not use_per_tensor_scale
    assert torch.equal(packed.dequantize(), w.dequantize())
    assert torch.equal(
        packed.dequantize(torch.float32), w.dequantize(output_dtype=torch.float32)
    )


@pytest.mark.parametrize("row_scales", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fusion_preserves_original_rows(dtype, row_scales):
    # Differing scalar global scales (0.25, 0.37, 0.083) also become per-row.
    pieces = shards(192, dtype=dtype)
    if row_scales:
        pieces[1] = CudaNVFP4Tensor(
            pieces[1].qdata,
            pieces[1].scale,
            torch.linspace(0.01, 0.035, 19),
            orig_dtype=dtype,
        )
    fused = CudaNVFP4Tensor.fuse_along_output(pieces)
    assert fused.per_tensor_scale.shape == (97,)
    assert torch.equal(fused.qdata, torch.cat([w.qdata for w in pieces]))
    assert torch.equal(fused.scale, torch.cat([w.scale for w in pieces]))
    assert torch.equal(fused.dequantize(), torch.cat([w.dequantize() for w in pieces]))
    moved = fused.to(dtype=torch.float16)
    assert torch.equal(moved.qdata, fused.qdata)
    assert torch.equal(moved.scale, fused.scale)
    assert torch.equal(moved.per_tensor_scale, fused.per_tensor_scale)
    assert moved.dtype == torch.float16


def test_fusion_preserves_scalar_scale_and_rejects_mismatched_activation_scale():
    pieces = shards(192)
    pts = pieces[0].per_tensor_scale
    pieces = [CudaNVFP4Tensor(w.qdata, w.scale, pts) for w in pieces]
    fused = CudaNVFP4Tensor.fuse_along_output(pieces)
    assert fused.per_tensor_scale.numel() == 1
    assert torch.equal(fused.dequantize(), torch.cat([w.dequantize() for w in pieces]))
    pieces[1] = CudaNVFP4Tensor(
        pieces[1].qdata, pieces[1].scale, pts, act_per_tensor_scale=0.001
    )
    with pytest.raises(ValueError, match="activation scale"):
        CudaNVFP4Tensor.fuse_along_output(pieces)


def test_fusion_rejects_incompatible_weights():
    a, b, _ = shards(192)
    k_mismatch = shards(384)[1]
    float_b = CudaNVFP4Tensor(
        b.qdata, b.scale, b.per_tensor_scale, orig_dtype=torch.float32
    )
    for weights in ([], [a, k_mismatch], [a, float_b]):
        with pytest.raises(ValueError, match="CUDA NVFP4 fusion"):
            CudaNVFP4Tensor.fuse_along_output(weights)
    with pytest.raises(TypeError, match="CUDA NVFP4 fusion"):
        CudaNVFP4Tensor.fuse_along_output([a, b.dequantize()])


def test_to_meta_keeps_layout():
    w = CudaNVFP4Tensor.from_exportable(weight())
    meta = w.to("meta")
    assert meta.device.type == "meta"
    assert meta.qdata.device.type == "meta" and meta.scale.device.type == "meta"
    assert meta.shape == w.shape and meta.unit_scale


def test_meta_and_fake_weights_check_a_number_scale_but_not_tensor_values():
    w = shards(192)[0]
    cpu = (w.qdata, w.scale, w.per_tensor_scale)
    moved = CudaNVFP4Tensor(*cpu, act_per_tensor_scale=0.5).to("meta")
    assert moved.act_per_tensor_scale.device.type == "meta"
    meta = tuple(t.to("meta") for t in cpu)
    # A meta tensor carries no value to read, so even zero is accepted.
    CudaNVFP4Tensor(*meta, act_per_tensor_scale=torch.zeros(1, device="meta"))
    # A Python number is checked on the host, for meta and fake weights too.
    with pytest.raises(ValueError, match="finite and positive"):
        CudaNVFP4Tensor(*meta, act_per_tensor_scale=0.0)
    mode = FakeTensorMode()
    fake = tuple(mode.from_tensor(t) for t in cpu)
    with mode, pytest.raises(ValueError, match="finite and positive"):
        CudaNVFP4Tensor(*fake, act_per_tensor_scale=-1.0)
    portable = weight()
    meta_portable = ExportableNVFP4Tensor(
        *(t.to("meta") for t in (portable.qdata, portable.scale)),
        portable.per_tensor_scale.to("meta"),
        16,
        torch.bfloat16,
        act_per_tensor_scale=0.0,
    )
    with pytest.raises(ValueError, match="finite and positive"):
        CudaNVFP4Tensor.from_exportable(meta_portable)


def test_activation_scale_is_a_scalar_tensor():
    w = shards(192)[0]
    w = CudaNVFP4Tensor(
        w.qdata, w.scale, w.per_tensor_scale, act_per_tensor_scale=torch.tensor([0.5])
    )
    for t in (
        w,
        CudaNVFP4Tensor.fuse_along_output([w, w]),
        w.to(dtype=torch.float16),
    ):
        act = t.act_per_tensor_scale
        assert act.shape == (1, 1) and act.dtype == torch.float32
        assert act.item() == 0.5
    other = CudaNVFP4Tensor(
        w.qdata, w.scale, w.per_tensor_scale, act_per_tensor_scale=0.25
    )
    with pytest.raises(ValueError, match="activation scale"):
        CudaNVFP4Tensor.fuse_along_output([w, other])
    stream = io.BytesIO()
    torch.save(w, stream)
    stream.seek(0)
    assert torch.load(stream, weights_only=True).act_per_tensor_scale.item() == 0.5
    with FakeTensorMode() as mode:
        assert is_fake(mode.from_tensor(w).act_per_tensor_scale)


@pytest.mark.parametrize(
    "act",
    [
        0.0,
        -1.0,
        math.inf,
        math.nan,
        1e39,
        torch.tensor(0.0),
        torch.tensor(-1.0),
        torch.tensor(math.inf),
        torch.tensor(math.nan),
        torch.ones(2),
        torch.ones(1, dtype=torch.bfloat16),
        torch.ones(1, device="meta"),
    ],
)
def test_invalid_activation_scale_rejected(act):
    w = shards(192)[0]
    with pytest.raises(ValueError, match="act_per_tensor_scale"):
        CudaNVFP4Tensor(w.qdata, w.scale, w.per_tensor_scale, act_per_tensor_scale=act)


@pytest.mark.parametrize("byte", [0x7F, 0xFF])
def test_from_exportable_rejects_nan_scales(byte):
    # Kernels assume finite E4M3 scales.
    w = weight()
    w.scale[3, 5] = byte
    with pytest.raises(ValueError, match="finite"):
        CudaNVFP4Tensor.from_exportable(w)


def test_from_exportable_carries_activation_scale():
    portable = weight()
    fields = (portable.qdata, portable.scale, portable.per_tensor_scale, 16)
    packed = CudaNVFP4Tensor.from_exportable(
        ExportableNVFP4Tensor(*fields, torch.bfloat16, act_per_tensor_scale=0.02)
    )
    act = packed.act_per_tensor_scale
    assert act.shape == (1, 1) and act.dtype == torch.float32
    assert act.item() == torch.tensor(0.02).item()
    # The portable class accepts a zero scale; packing rejects it.
    zero = ExportableNVFP4Tensor(*fields, torch.bfloat16, act_per_tensor_scale=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        CudaNVFP4Tensor.from_exportable(zero)


@pytest.mark.parametrize(
    "pts,scale_byte,expected",
    [
        (1.0, 0, True),
        (1.0, 126, True),
        (0.5, 40, False),
        (1.0, 128, False),
        (1.0, 0xA0, False),
    ],
)
def test_unit_scale_inferred_only_for_exact_domain(pts, scale_byte, expected):
    actual = CudaNVFP4Tensor.from_exportable(weight(pts, scale_byte))
    assert actual.unit_scale is expected


def test_unit_scale_requires_every_scale_byte():
    w = weight()
    scale = w.scale.clone()
    scale[3, 5] = 0xA0
    mixed = ExportableNVFP4Tensor(
        w.qdata, scale, w.per_tensor_scale, w.block_size, w.orig_dtype
    )
    assert not CudaNVFP4Tensor.from_exportable(mixed).unit_scale


def test_unit_scale_metadata_survives_fake_to_fusion_and_serialization():
    unit = CudaNVFP4Tensor.from_exportable(weight())
    nonunit = CudaNVFP4Tensor.from_exportable(weight(0.5))
    assert unit.to(dtype=torch.float16).unit_scale
    assert CudaNVFP4Tensor.fuse_along_output([unit, unit]).unit_scale
    assert not CudaNVFP4Tensor.fuse_along_output([unit, nonunit]).unit_scale
    stream = io.BytesIO()
    torch.save(unit, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    assert type(restored) is CudaNVFP4Tensor
    assert restored.unit_scale and restored.act_per_tensor_scale is None
    assert (restored.shape, restored.dtype) == (unit.shape, unit.dtype)
    assert torch.equal(restored.qdata, unit.qdata)
    assert torch.equal(restored.scale, unit.scale)
    assert torch.equal(restored.per_tensor_scale, unit.per_tensor_scale)
    with FakeTensorMode() as mode:
        fake = mode.from_tensor(unit)
        assert is_fake(fake.qdata) and fake.unit_scale
        # A new fake weight cannot prove scale values. Reconstruction of an
        # already materialized weight instead preserves the serialized flag.
        unknown = CudaNVFP4Tensor.from_exportable(weight())
        assert not unknown.unit_scale


@pytest.mark.parametrize("row_scales", [False, True])
def test_dequantize_low_nibble_first(row_scales):
    codes = torch.arange(16, dtype=torch.uint8).repeat(2, 1)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    scales = torch.full((2, 1), 56, dtype=torch.uint8)
    pts = torch.tensor([0.25, 0.5] if row_scales else [0.25])
    tensor = CudaNVFP4Tensor(packed, scales, pts, orig_dtype=torch.float32)
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
    )
    torch.testing.assert_close(
        tensor.dequantize(), lut.repeat(2, 1) * pts[:, None], rtol=0, atol=0
    )
    assert tensor.qdata.is_contiguous() and tensor.scale.is_contiguous()
    assert tensor.per_tensor_scale.shape == ((2,) if row_scales else (1, 1))


@pytest.mark.parametrize(
    "invalid",
    [
        "qdata_dtype",
        "qdata_shape",
        "block_size",
        "scale_shape",
        "scale_dtype",
        "global_scale",
        "global_scale_dtype",
        "device",
        "k_multiple",
        "orig_dtype",
    ],
)
def test_invalid_layout_rejected(invalid):
    q = torch.zeros((2, 8), dtype=torch.uint8)
    s = torch.full((2, 1), 56, dtype=torch.uint8)
    pts = torch.ones(1)
    block = 16
    orig_dtype = torch.bfloat16
    if invalid == "qdata_dtype":
        q = q.float()
    elif invalid == "qdata_shape":
        q = q.flatten()
    elif invalid == "block_size":
        block = 32
    elif invalid == "scale_shape":
        s = torch.zeros((2, 2), dtype=torch.uint8)
    elif invalid == "scale_dtype":
        s = s.float()
    elif invalid == "global_scale_dtype":
        pts = torch.ones(1, dtype=torch.float64)
    elif invalid == "device":
        pts = torch.ones(1, device="meta")
    elif invalid == "k_multiple":
        q = torch.zeros((2, 12), dtype=torch.uint8)
    elif invalid == "orig_dtype":
        orig_dtype = torch.int8
    else:
        pts = torch.ones(3)
    with pytest.raises(ValueError, match="NVFP4"):
        CudaNVFP4Tensor(q, s, pts, block_size=block, orig_dtype=orig_dtype)
