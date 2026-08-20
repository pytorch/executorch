# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests: torchao subclasses survive safetensors roundtrip."""

import os
import tempfile
import unittest

import torch

from safetensors import safe_open
from safetensors.torch import save_file
from torchao.prototype.safetensors.safetensors_support import (
    flatten_tensor_state_dict,
    unflatten_tensor_state_dict,
)


def save(state_dict, path):
    tensors_data, metadata = flatten_tensor_state_dict(state_dict)
    save_file(tensors_data, path, metadata=metadata)


def load(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    result, _ = unflatten_tensor_state_dict(tensors, metadata)
    return result


from torchao.quantization import IntxUnpackedToInt8Tensor
from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor


def _make_int4(shape, group_size=32):
    """Build a random Int4Tensor."""
    N, K = shape
    packed = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8)
    scale = torch.randn(K // group_size, N, dtype=torch.bfloat16)
    zp = torch.zeros(K // group_size, N, dtype=torch.bfloat16)
    return Int4Tensor(
        qdata=packed,
        scale=scale,
        zero_point=zp,
        block_size=[1, group_size],
        shape=torch.Size([N, K]),
    )


def _make_int8(shape, group_size=32):
    """Build a random IntxUnpackedToInt8Tensor."""
    N, K = shape
    return IntxUnpackedToInt8Tensor(
        qdata=torch.randint(-128, 127, (N, K), dtype=torch.int8),
        scale=torch.randn(N, K // group_size, dtype=torch.bfloat16),
        zero_point=torch.zeros(N, K // group_size, dtype=torch.int8),
        target_dtype=torch.int8,
        block_size=(1, group_size),
        dtype=torch.bfloat16,
        activation_quantization=None,
    )


class TestSaveLoad(unittest.TestCase):
    def test_int4_roundtrip(self):
        """Int4Tensor survives save/load."""
        t = _make_int4((64, 128))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"layer.weight": t}, path)
            loaded = load(path)

        self.assertIn("layer.weight", loaded)
        self.assertIsInstance(loaded["layer.weight"], Int4Tensor)
        self.assertTrue(torch.equal(t.qdata, loaded["layer.weight"].qdata))
        self.assertTrue(torch.equal(t.scale, loaded["layer.weight"].scale))

    def test_int8_roundtrip(self):
        """IntxUnpackedToInt8Tensor survives save/load."""
        t = _make_int8((64, 128))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"layer.weight": t}, path)
            loaded = load(path)

        self.assertIn("layer.weight", loaded)
        self.assertIsInstance(loaded["layer.weight"], IntxUnpackedToInt8Tensor)
        self.assertTrue(torch.equal(t.qdata, loaded["layer.weight"].qdata))

    def test_mixed_state_dict(self):
        """Mixed Int4 + Int8 + plain tensor roundtrip."""
        state = {
            "linear.weight": _make_int4((64, 128)),
            "embed.weight": _make_int8((100, 64)),
            "norm.weight": torch.randn(64, dtype=torch.bfloat16),
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save(state, path)
            loaded = load(path)

        self.assertEqual(set(state.keys()), set(loaded.keys()))
        self.assertIsInstance(loaded["linear.weight"], Int4Tensor)
        self.assertIsInstance(loaded["embed.weight"], IntxUnpackedToInt8Tensor)
        self.assertIsInstance(loaded["norm.weight"], torch.Tensor)
        self.assertTrue(torch.equal(state["norm.weight"], loaded["norm.weight"]))

    def test_plain_tensor_only(self):
        """State dict with only plain tensors roundtrips."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"model.norm.weight": torch.randn(64, dtype=torch.bfloat16)}, path)
            loaded = load(path)
        self.assertIn("model.norm.weight", loaded)

    def test_3d_int4(self):
        """3D Int4Tensor (MoE expert weights) roundtrips."""
        # 3D: (num_experts, N, K//2) packed
        N, K, gs = 32, 64, 32
        packed = torch.randint(0, 255, (4, N, K // 2), dtype=torch.uint8)
        scale = torch.randn(4, K // gs, N, dtype=torch.bfloat16)
        zp = torch.zeros(4, K // gs, N, dtype=torch.bfloat16)
        t = Int4Tensor(
            qdata=packed,
            scale=scale,
            zero_point=zp,
            block_size=[1, 1, gs],
            shape=torch.Size([4, N, K]),
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"experts.w1": t}, path)
            loaded = load(path)
        self.assertTrue(torch.equal(t.qdata, loaded["experts.w1"].qdata))


if __name__ == "__main__":
    unittest.main()
