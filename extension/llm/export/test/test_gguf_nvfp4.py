# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import tempfile
import unittest

import numpy as np
import torch
from executorch.extension.llm.export.gguf import ExportableGGUFTensor, iter_gguf
from executorch.extension.llm.export.quant.convert import fuse_along_output
from torch._subclasses.fake_tensor import FakeTensorMode

try:
    import gguf

    _HAS_GGUF_NVFP4 = hasattr(gguf.GGMLQuantizationType, "NVFP4")
except ImportError:
    _HAS_GGUF_NVFP4 = False

ONE = torch.tensor(1.0)

E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


# gguf added NVFP4 in 0.19; without it the reference checks skip and the
# gguf-independent tests still run.
requires_gguf_nvfp4 = unittest.skipUnless(
    _HAS_GGUF_NVFP4, "needs gguf >= 0.19 (GGMLQuantizationType.NVFP4)"
)


def blocks(w, dtype=torch.float32):
    """The blocks decoded without their per-tensor scale."""
    return torch.ops.torchao.dequantize_gguf(w.raw, "nvfp4", dtype)


def random_raw(n, k, generator):
    raw = torch.randint(
        0, 256, (n, k // 64 * 36), dtype=torch.uint8, generator=generator
    )
    raw.reshape(n, -1, 36)[..., :4].clamp_(max=254)
    return raw


def reference(raw):
    return torch.from_numpy(
        np.ascontiguousarray(
            gguf.dequantize(raw.numpy(), gguf.GGMLQuantizationType.NVFP4)
        )
    )


class TestGGUFNVFP4(unittest.TestCase):
    def assert_bits_equal(self, a, b):
        self.assertTrue(
            torch.equal(
                a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
            )
        )

    def test_hand_built_block(self):
        # Scales 1.0, 2.0, 1.0 with the sign bit GGML ignores, and 0x7f (zero).
        # Each K16 group holds codes 0..15, packed as eight lows then eight highs.
        raw = torch.zeros(1, 36, dtype=torch.uint8)
        raw[0, :4] = torch.tensor([0x38, 0x40, 0xB8, 0x7F], dtype=torch.uint8)
        raw[0, 4:] = torch.tensor(
            [i | ((i + 8) << 4) for i in range(8)], dtype=torch.uint8
        ).repeat(4)
        expected = torch.cat((E2M1, E2M1 * 2, E2M1, E2M1 * 0))[None]

        w = ExportableGGUFTensor(raw, "nvfp4", torch.float32)
        converted = w.to_exportable_nvfp4_tensor(per_tensor_scale=ONE)
        self.assertTrue(torch.equal(blocks(w), expected))
        self.assertTrue(torch.equal(converted.dequantize(), expected))
        # torchao keeps the even element in the low nibble; code 8 becomes +0.
        q = converted.qdata.view(torch.uint8)[0, :8]
        self.assertEqual(q.tolist(), [0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE])
        self.assertEqual(converted.scale[0].tolist(), [0x38, 0x40, 0x38, 0x00])

    @requires_gguf_nvfp4
    def test_all_scale_bytes_and_codes(self):
        # All 256 scale bytes and every E2M1 code, including negative zero.
        raw = torch.zeros(256, 36, dtype=torch.uint8)
        raw[:, :4] = torch.arange(256, dtype=torch.int64).to(torch.uint8)[:, None]
        raw[:, 4:] = torch.tensor(
            [i | ((i + 8) << 4) for i in range(8)], dtype=torch.uint8
        ).repeat(4)
        w = ExportableGGUFTensor(raw, "nvfp4", torch.float32)
        expected = reference(raw)
        self.assertEqual(w.shape, (256, 64))
        self.assert_bits_equal(blocks(w), expected)
        # 0xff (480) has no E4M3 encoding, so only the other bytes convert.
        converted = ExportableGGUFTensor(raw[:255], "nvfp4", torch.float32)
        self.assert_bits_equal(
            converted.to_exportable_nvfp4_tensor(per_tensor_scale=ONE).dequantize(),
            expected[:255],
        )

    @requires_gguf_nvfp4
    def test_random_blocks_match_reference(self):
        generator = torch.Generator().manual_seed(0)
        for n, k in [(1, 64), (7, 192), (32, 512)]:
            with self.subTest(n=n, k=k):
                raw = random_raw(n, k, generator)
                w = ExportableGGUFTensor(raw, "nvfp4", torch.float32)
                expected = reference(raw)
                converted = w.to_exportable_nvfp4_tensor(per_tensor_scale=ONE)
                self.assert_bits_equal(converted.dequantize(), expected)
                self.assert_bits_equal(blocks(w), expected)

    def test_per_tensor_scale_and_dtype(self):
        generator = torch.Generator().manual_seed(0)
        w = ExportableGGUFTensor(random_raw(7, 192, generator), "nvfp4", torch.float32)
        expected = blocks(w)
        self.assert_bits_equal(
            w.to_exportable_nvfp4_tensor(
                per_tensor_scale=torch.tensor(0.25)
            ).dequantize(),
            expected * 0.25,
        )
        moved = w.to_exportable_nvfp4_tensor(per_tensor_scale=ONE).to(
            dtype=torch.bfloat16
        )
        self.assertEqual(moved.qdata.dtype, torch.uint32)
        self.assertEqual(moved.scale.dtype, torch.uint8)
        self.assertTrue(torch.equal(moved.dequantize(), expected.to(torch.bfloat16)))
        for scales in (
            {"per_tensor_scale": torch.ones(2)},
            {"per_tensor_scale": ONE, "act_per_tensor_scale": torch.ones(2)},
        ):
            with self.assertRaisesRegex(ValueError, "scalars"):
                w.to_exportable_nvfp4_tensor(**scales)
        # A loader's dtype cast would round the F32 scale before pairing.
        for scales in (
            {"per_tensor_scale": torch.tensor(0.25, dtype=torch.bfloat16)},
            {"per_tensor_scale": ONE, "act_per_tensor_scale": ONE.bfloat16()},
        ):
            with self.assertRaisesRegex(ValueError, "float32"):
                w.to_exportable_nvfp4_tensor(**scales)
        self.assert_bits_equal(
            w.to_exportable_nvfp4_tensor(per_tensor_scale=0.25).dequantize(),
            expected * 0.25,
        )

    def test_act_per_tensor_scale_survives_to(self):
        w = ExportableGGUFTensor(
            torch.zeros(1, 36, dtype=torch.uint8), "nvfp4", torch.float32
        )
        converted = w.to_exportable_nvfp4_tensor(
            per_tensor_scale=ONE, act_per_tensor_scale=torch.tensor([0.001])
        )
        moved = converted.to(torch.bfloat16)
        self.assertIs(type(moved.act_per_tensor_scale), float)
        self.assertAlmostEqual(moved.act_per_tensor_scale, 0.001)
        self.assertIsNone(
            w.to_exportable_nvfp4_tensor(per_tensor_scale=ONE)
            .to(torch.bfloat16)
            .act_per_tensor_scale
        )

    def test_conversion_rejects_other_quant_types(self):
        w = ExportableGGUFTensor(
            torch.zeros(1, 210, dtype=torch.uint8), "q6_k", torch.float32
        )
        with self.assertRaises(NotImplementedError):
            w.to_exportable_nvfp4_tensor(per_tensor_scale=ONE)

    def test_export(self):
        generator = torch.Generator().manual_seed(0)
        w = ExportableGGUFTensor(random_raw(4, 128, generator), "nvfp4", torch.bfloat16)
        weight = w.to_exportable_nvfp4_tensor(per_tensor_scale=torch.tensor(0.5))
        linear = torch.nn.Linear(128, 4, bias=False)
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        x = torch.randn(2, 128, dtype=torch.bfloat16)
        ep = torch.export.export(linear, (x,)).run_decompositions({})
        targets = {str(n.target) for n in ep.graph.nodes if n.op == "call_function"}
        self.assertIn("torchao.dequantize_nvfp4.default", targets)
        self.assertTrue(torch.equal(ep.module()(x), linear(x)))
        self.assertTrue(torch.equal(linear(x), x @ weight.dequantize().t()))

    def test_blocks_without_scale_are_not_a_weight(self):
        w = ExportableGGUFTensor(
            torch.zeros(4, 36, dtype=torch.uint8), "nvfp4", torch.float32
        )
        uses = {
            "dequantize": w.dequantize,
            "linear": lambda: torch.nn.functional.linear(torch.randn(2, 64), w),
            "embedding": lambda: torch.nn.functional.embedding(torch.tensor([1]), w),
            "t": w.t,
            "_to_copy": lambda: torch.ops.aten._to_copy.default(w),
        }
        for name, use in uses.items():
            with self.subTest(use=name):
                with self.assertRaisesRegex(ValueError, "to_exportable_nvfp4_tensor"):
                    use()
        with self.assertRaisesRegex(TypeError, "to_exportable_nvfp4_tensor"):
            fuse_along_output([w, w])

    @requires_gguf_nvfp4
    def test_iter_gguf_pairs_weight_and_scale(self):
        raw = random_raw(3, 128, torch.Generator().manual_seed(0))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nvfp4.gguf")
            writer = gguf.GGUFWriter(path, "test")
            writer.add_tensor(
                "blk.0.weight", raw.numpy(), raw_dtype=gguf.GGMLQuantizationType.NVFP4
            )
            writer.add_tensor("blk.0.scale", np.array([0.5], dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            state = dict(iter_gguf(path))
        w = state["blk.0.weight"]
        self.assertIsInstance(w, ExportableGGUFTensor)
        self.assertEqual((w.ggml_type, w.shape), ("nvfp4", (3, 128)))
        self.assert_bits_equal(blocks(w), reference(raw))
        paired = w.to_exportable_nvfp4_tensor(per_tensor_scale=state["blk.0.scale"])
        self.assert_bits_equal(paired.dequantize(torch.float32), reference(raw) * 0.5)

    def test_fake_shape(self):
        with FakeTensorMode():
            w = ExportableGGUFTensor(
                torch.empty(7, 108, dtype=torch.uint8), "nvfp4", torch.bfloat16
            )
            self.assertEqual(w.shape, (7, 192))
            self.assertEqual(blocks(w, torch.bfloat16).shape, (7, 192))
            self.assertEqual(
                w.to_exportable_nvfp4_tensor(per_tensor_scale=torch.tensor(1.0))
                .dequantize()
                .shape,
                (7, 192),
            )

    def test_invalid_block(self):
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            ExportableGGUFTensor(
                torch.empty(2, 35, dtype=torch.uint8), "nvfp4", torch.float32
            )

    def test_unrepresentable_scale_is_rejected(self):
        raw = torch.zeros(1, 36, dtype=torch.uint8)
        raw[:, 0] = 255
        raw[:, 4:] = 0x11
        w = ExportableGGUFTensor(raw, "nvfp4", torch.float32)
        # GGML decodes scale 0xff as 480; the other three groups have scale 0.
        expected = torch.cat((torch.full((16,), 240.0), torch.zeros(48)))[None]
        self.assertTrue(torch.equal(blocks(w), expected))
        with self.assertRaisesRegex(ValueError, "480"):
            w.to_exportable_nvfp4_tensor(per_tensor_scale=ONE)


if __name__ == "__main__":
    unittest.main()
