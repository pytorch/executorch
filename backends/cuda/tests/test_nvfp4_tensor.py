# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-only NVFP4 packing, fusion and serialized metadata contracts."""

import io

import pytest
import torch
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.tests.nvfp4_test_utils import portable_weight as weight, shards
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fusion_preserves_original_rows(dtype):
    pieces = shards(192, dtype=dtype)
    pieces[1].per_tensor_scale = torch.linspace(0.01, 0.035, 19)
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
    for w in pieces:
        w.per_tensor_scale = pieces[0].per_tensor_scale
    fused = CudaNVFP4Tensor.fuse_along_output(pieces)
    assert fused.per_tensor_scale.numel() == 1
    assert torch.equal(fused.dequantize(), torch.cat([w.dequantize() for w in pieces]))
    pieces[1].act_per_tensor_scale = 0.001
    with pytest.raises(ValueError, match="activation scale"):
        CudaNVFP4Tensor.fuse_along_output(pieces)


@pytest.mark.parametrize(
    "pts,scale_byte,expected",
    [
        (1.0, 0, True),
        (1.0, 126, True),
        (0.5, 40, False),
        (1.0, 127, False),
        (1.0, 128, False),
    ],
)
def test_unit_scale_inferred_only_for_exact_domain(pts, scale_byte, expected):
    actual = CudaNVFP4Tensor.from_exportable(weight(pts, scale_byte))
    assert actual.unit_scale is expected


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
    assert restored.unit_scale and torch.equal(restored.scale, unit.scale)
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
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    torch.testing.assert_close(tensor.dequantize(), lut.repeat(2, 1) * pts[:, None], rtol=0, atol=0)
    assert tensor.qdata.is_contiguous() and tensor.scale.is_contiguous()
    assert tensor.per_tensor_scale.ndim == 1


@pytest.mark.parametrize("invalid", ["qdata_dtype", "qdata_shape", "block_size", "scale_shape", "scale_dtype", "global_scale"])
def test_invalid_layout_rejected(invalid):
    q = torch.zeros((2, 8), dtype=torch.uint8)
    s = torch.full((2, 1), 56, dtype=torch.uint8)
    pts = torch.ones(1)
    block = 16
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
    else:
        pts = torch.ones(3)
    with pytest.raises(ValueError, match="CUDA NVFP4"):
        CudaNVFP4Tensor(q, s, pts, block_size=block)
