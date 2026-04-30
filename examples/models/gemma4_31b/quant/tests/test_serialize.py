# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for quant/serialize.py — data format and I/O only.

Tests nibble pack/unpack and save/load. Does NOT test
quantize_weight (that lives in test_quantize.py). Save/load tests use
hand-built CanonicalQuantizedWeight fixtures to avoid coupling to the
quantizer.
"""

import json
import os
import tempfile
import unittest

import torch

from executorch.examples.models.gemma4_31b.quant.recipe import QuantConfig

from executorch.examples.models.gemma4_31b.quant.serialize import (
    _nibble_pack,
    _nibble_unpack,
    CanonicalQuantizedWeight,
    deserialize,
    load,
    save,
    serialize,
)
from safetensors import safe_open


def _make_cqw(
    shape: tuple[int, ...],
    config: QuantConfig,
) -> CanonicalQuantizedWeight:
    """Build a CanonicalQuantizedWeight with random data (no actual quantization)."""
    K = shape[-1]
    n_groups = K // config.group_size
    scale_shape = (*shape[:-1], n_groups)

    if config.bits == 4:
        qdata = torch.randint(0, 16, shape, dtype=torch.int8)
    else:
        qdata = torch.randint(-128, 128, shape, dtype=torch.int8)

    return CanonicalQuantizedWeight(
        qdata=qdata,
        scale=torch.randn(scale_shape, dtype=torch.bfloat16),
        zero=(
            torch.randn(scale_shape, dtype=torch.bfloat16)
            if not config.symmetric
            else None
        ),
        config=config,
    )


# ---------------------------------------------------------------------------
# Nibble pack / unpack


class TestNibblePack(unittest.TestCase):
    def test_roundtrip(self):
        qdata = torch.randint(0, 16, (8, 64), dtype=torch.int8)
        packed = _nibble_pack(qdata)
        self.assertEqual(packed.shape, (8, 32))
        self.assertTrue(torch.equal(qdata, _nibble_unpack(packed, 64)))

    def test_rejects_odd_last_dim(self):
        with self.assertRaises(AssertionError):
            _nibble_pack(torch.zeros(4, 33, dtype=torch.int8))

    def test_3d(self):
        """Nibble pack works for 3D tensors (MoE expert weights)."""
        qdata = torch.randint(0, 16, (4, 32, 64), dtype=torch.int8)
        packed = _nibble_pack(qdata)
        self.assertEqual(packed.shape, (4, 32, 32))
        self.assertTrue(torch.equal(qdata, _nibble_unpack(packed, 64)))


# ---------------------------------------------------------------------------
# save / load


class TestSerializeDeserialize(unittest.TestCase):
    """Pure logic layer — no disk I/O."""

    def test_roundtrip(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = _make_cqw((64, 128), config)
        unq = {"embed": torch.randn(8, 8, dtype=torch.bfloat16)}

        tensors, header = serialize({"w": cw}, unq)
        q, u = deserialize(tensors, header)

        self.assertTrue(torch.equal(cw.qdata, q["w"].qdata))
        self.assertTrue(torch.equal(cw.scale, q["w"].scale))
        self.assertTrue(torch.equal(cw.zero, q["w"].zero))
        self.assertEqual(cw.config, q["w"].config)
        self.assertTrue(torch.equal(unq["embed"], u["embed"]))

    def test_rejects_unsupported_version(self):
        tensors, header = serialize({}, {"w": torch.randn(4, 4)})
        header["format_version"] = "99"
        with self.assertRaises(ValueError) as ctx:
            deserialize(tensors, header)
        self.assertIn("99", str(ctx.exception))


class TestSaveLoad(unittest.TestCase):
    """I/O layer — roundtrip through safetensors on disk."""

    def test_roundtrip_asymmetric(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = _make_cqw((64, 128), config)
        unq = {"embed.weight": torch.randn(32, 64, dtype=torch.bfloat16)}

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"w": cw}, unq, path)
            q, u = load(path)

        self.assertTrue(torch.equal(cw.qdata, q["w"].qdata))
        self.assertTrue(torch.equal(cw.scale, q["w"].scale))
        self.assertTrue(torch.equal(cw.zero, q["w"].zero))
        self.assertEqual(cw.config, q["w"].config)
        self.assertTrue(torch.equal(unq["embed.weight"], u["embed.weight"]))

    def test_roundtrip_symmetric(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        cw = _make_cqw((32, 64), config)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"w": cw}, {}, path)
            q, _ = load(path)

        self.assertIsNone(q["w"].zero)
        self.assertTrue(torch.equal(cw.qdata, q["w"].qdata))

    def test_roundtrip_3d(self):
        """3D quantized weights (MoE experts) roundtrip correctly."""
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = _make_cqw((8, 64, 128), config)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"experts.w1": cw}, {}, path)
            q, _ = load(path)

        self.assertTrue(torch.equal(cw.qdata, q["experts.w1"].qdata))
        self.assertEqual(q["experts.w1"].scale.shape, (8, 64, 4))

    def test_4bit_nibble_packed_on_disk(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = _make_cqw((64, 128), config)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"w": cw}, {}, path)
            with safe_open(path, framework="pt", device="cpu") as f:
                on_disk = f.get_tensor("w.qdata")
        self.assertEqual(on_disk.shape, (64, 64))

    def test_8bit_not_nibble_packed(self):
        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        cw = _make_cqw((32, 64), config)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"w": cw}, {}, path)
            with safe_open(path, framework="pt", device="cpu") as f:
                on_disk = f.get_tensor("w.qdata")
        self.assertEqual(on_disk.shape, (32, 64))  # no packing for 8-bit

    def test_header_metadata(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = _make_cqw((32, 64), config)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"foo.weight": cw}, {}, path)
            with safe_open(path, framework="pt", device="cpu") as f:
                meta = json.loads(f.metadata()["quant"])

        self.assertIn("foo.weight", meta)
        self.assertEqual(meta["foo.weight"]["bits"], 4)
        self.assertEqual(meta["foo.weight"]["group_size"], 32)
        self.assertFalse(meta["foo.weight"]["symmetric"])
        self.assertEqual(meta["foo.weight"]["method"], "min_max")

    def test_empty_quantized(self):
        unq = {"w": torch.randn(8, 8, dtype=torch.bfloat16)}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({}, unq, path)
            q, u = load(path)
        self.assertEqual(len(q), 0)
        self.assertTrue(torch.equal(unq["w"], u["w"]))


if __name__ == "__main__":
    unittest.main()
