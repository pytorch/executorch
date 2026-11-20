# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import gguf
import numpy as np
import pytest
import torch
from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from torch._subclasses.fake_tensor import FakeTensorMode


requires_gguf_nvfp4 = pytest.mark.skipif(
    not hasattr(gguf.GGMLQuantizationType, "NVFP4"),
    reason="gguf package without the NVFP4 reference dequantizer",
)


def assert_bits_equal(a, b):
    assert torch.equal(
        a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    )


@requires_gguf_nvfp4
def test_all_scale_bytes_and_codes():
    # Every scale bit pattern and E2M1 code, including negative zero.
    raw = torch.zeros(255, 36, dtype=torch.uint8)
    raw[:, :4] = torch.arange(255, dtype=torch.int64).to(torch.uint8)[:, None]
    raw[:, 4:] = torch.tensor(
        [i | ((i + 8) << 4) for i in range(8)], dtype=torch.uint8
    ).repeat(4)
    w = ExportableGGUFTensor(raw, "nvfp4", torch.float32)
    expected = torch.from_numpy(
        gguf.dequantize(raw.numpy(), gguf.GGMLQuantizationType.NVFP4)
    )
    assert w.shape == (255, 64)
    assert_bits_equal(w.dequantize(), expected)
    assert_bits_equal(w.to_exportable_nvfp4_tensor().dequantize(), expected)


@requires_gguf_nvfp4
@pytest.mark.parametrize("n,k", [(1, 64), (7, 192), (32, 512)])
def test_random_repacking(n, k):
    raw = torch.randint(0, 256, (n, k // 64 * 36), dtype=torch.uint8)
    blocks = raw.reshape(n, -1, 36)
    blocks[..., :4].clamp_(max=254)
    w = ExportableGGUFTensor(raw, "nvfp4", torch.float32)
    expected = torch.from_numpy(
        np.ascontiguousarray(
            gguf.dequantize(raw.numpy(), gguf.GGMLQuantizationType.NVFP4)
        )
    )
    converted = w.to_exportable_nvfp4_tensor()
    assert_bits_equal(converted.dequantize(), expected)
    assert_bits_equal(w.dequantize(), expected)
    assert_bits_equal(
        w.to_exportable_nvfp4_tensor(torch.tensor(0.25)).dequantize(), expected * 0.25
    )
    moved = converted.to(dtype=torch.bfloat16)
    assert moved.qdata.dtype == torch.uint32
    assert moved.scale.dtype == torch.uint8
    assert torch.equal(moved.dequantize(), expected.to(torch.bfloat16))


def test_fake_shape():
    with FakeTensorMode():
        w = ExportableGGUFTensor(
            torch.empty(7, 108, dtype=torch.uint8), "nvfp4", torch.bfloat16
        )
        assert w.shape == (7, 192)
        assert w.dequantize().shape == (7, 192)
        assert w.to_exportable_nvfp4_tensor().dequantize().shape == (7, 192)


def test_invalid_block():
    with pytest.raises(ValueError, match="not a multiple"):
        ExportableGGUFTensor(
            torch.empty(2, 35, dtype=torch.uint8), "nvfp4", torch.float32
        )


@requires_gguf_nvfp4
def test_unrepresentable_scale_is_rejected():
    raw = torch.zeros(1, 36, dtype=torch.uint8)
    raw[:, 0] = 255
    raw[:, 4:] = 0x11
    w = ExportableGGUFTensor(raw, "nvfp4", torch.float32)
    expected = torch.from_numpy(
        gguf.dequantize(raw.numpy(), gguf.GGMLQuantizationType.NVFP4)
    )
    assert_bits_equal(w.dequantize(), expected)
    with pytest.raises(ValueError, match="480"):
        w.to_exportable_nvfp4_tensor()
