# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for quant/pack_cuda.py. Requires CUDA."""

import os
import tempfile
import unittest

import torch
import torch.nn as nn

from executorch.examples.models.gemma4_31b.quant.pack import pack_one
from executorch.examples.models.gemma4_31b.quant.pack_cuda import (
    DEFAULT_CUDA_PACKERS,
    load_and_pack_for_cuda,
    pack_embedding_for_cuda,
    pack_int4_for_cuda,
    pack_int8_for_cuda,
    pack_linear_for_cuda,
    pack_model,
)
from executorch.examples.models.gemma4_31b.quant.quantize import quantize_weight
from executorch.examples.models.gemma4_31b.quant.recipe import QuantConfig

from executorch.examples.models.gemma4_31b.quant.serialize import save


class TestPackInt4ForCuda(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_symmetric_works(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        cw = quantize_weight(torch.randn(128, 256, dtype=torch.bfloat16), config)
        self.assertEqual(pack_int4_for_cuda(cw).shape, torch.Size([128, 256]))

    def test_rejects_1d(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        cw = quantize_weight(torch.randn(1, 128, dtype=torch.bfloat16), config)
        cw.qdata = cw.qdata.squeeze(0)
        with self.assertRaises(AssertionError):
            pack_int4_for_cuda(cw)

    def test_rejects_8bit(self):
        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        cw = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        with self.assertRaises(AssertionError):
            pack_int4_for_cuda(cw)

    def test_different_group_sizes(self):
        for gs in (32, 64, 128):
            with self.subTest(group_size=gs):
                config = QuantConfig(
                    bits=4, group_size=gs, symmetric=False, method="min_max"
                )
                cw = quantize_weight(
                    torch.randn(128, 256, dtype=torch.bfloat16), config
                )
                self.assertEqual(pack_int4_for_cuda(cw).shape, torch.Size([128, 256]))

    def test_matmul_approximates_original(self):
        """Packed weight produces matmul output close to the original."""
        torch.manual_seed(0)
        # Use dimensions already aligned to tinygemm requirements
        # (K multiple of 1024, N multiple of 8) to avoid padding effects.
        weight = torch.randn(256, 1024, dtype=torch.bfloat16)
        x = torch.randn(1, 1024, dtype=torch.bfloat16)

        original_out = torch.nn.functional.linear(x.cuda(), weight.cuda())

        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = quantize_weight(weight, config)
        packed = pack_int4_for_cuda(cw)

        packed_out = torch.nn.functional.linear(x.cuda(), packed.cuda())

        rel_error = (
            packed_out.float() - original_out.float()
        ).abs().mean() / original_out.float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)

    def test_symmetric_matmul_approximates_original(self):
        """Symmetric 4-bit (e.g. HQQ) packs correctly for tinygemm."""
        torch.manual_seed(0)
        weight = torch.randn(256, 1024, dtype=torch.bfloat16)
        x = torch.randn(1, 1024, dtype=torch.bfloat16)

        original_out = torch.nn.functional.linear(x.cuda(), weight.cuda())

        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        cw = quantize_weight(weight, config)
        packed = pack_int4_for_cuda(cw)

        packed_out = torch.nn.functional.linear(x.cuda(), packed.cuda())

        rel_error = (
            packed_out.float() - original_out.float()
        ).abs().mean() / original_out.float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)


class TestPackInt8ForCuda(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_rejects_4bit(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        with self.assertRaises(AssertionError):
            pack_int8_for_cuda(cw)

    def test_matmul_approximates_original(self):
        torch.manual_seed(0)
        weight = torch.randn(256, 128, dtype=torch.bfloat16)
        x = torch.randn(1, 128, dtype=torch.bfloat16)

        original_out = torch.nn.functional.linear(x.cuda(), weight.cuda())

        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        cw = quantize_weight(weight, config)
        packed = pack_int8_for_cuda(cw)

        packed_out = torch.nn.functional.linear(x.cuda(), packed.cuda())

        rel_error = (
            packed_out.float() - original_out.float()
        ).abs().mean() / original_out.float().abs().mean()
        self.assertLess(rel_error.item(), 0.02)

    def test_per_axis_gather_approximates_original(self):
        """Per-axis INT8 (group_size == K) works for embedding gather."""
        torch.manual_seed(0)
        weight = torch.randn(1000, 64, dtype=torch.bfloat16)
        ids = torch.tensor([0, 1, 42, 500, 999])

        original = weight[ids]

        config = QuantConfig(bits=8, group_size=64, symmetric=True, method="min_max")
        cw = quantize_weight(weight, config)
        packed = pack_int8_for_cuda(cw)

        emb = nn.Embedding(1000, 64)
        emb.weight = nn.Parameter(packed, requires_grad=False)
        emb.to("cuda")
        packed_out = emb(ids.cuda())

        rel_error = (
            packed_out.cpu().float() - original.float()
        ).abs().mean() / original.float().abs().mean()
        self.assertLess(rel_error.item(), 0.02)


class TestPackLinearForCuda(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_4bit_modifies_module_in_place(self):
        module = nn.Linear(128, 256, bias=False)
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = quantize_weight(torch.randn(256, 128, dtype=torch.bfloat16), config)
        pack_linear_for_cuda(module, {"weight": cw})
        self.assertEqual(module.weight.device.type, "cpu")
        self.assertEqual(module.weight.shape, torch.Size([256, 128]))

    def test_8bit_modifies_module_in_place(self):
        module = nn.Linear(128, 64, bias=False)
        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        cw = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        pack_linear_for_cuda(module, {"weight": cw})
        self.assertEqual(module.weight.shape, torch.Size([64, 128]))


class TestPackEmbeddingForCuda(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_gather_approximates_original(self):
        """INT8 quantized embedding gather matches bf16 gather."""
        torch.manual_seed(0)
        weight = torch.randn(1000, 64, dtype=torch.bfloat16)
        ids = torch.tensor([0, 1, 42, 500, 999])

        original = weight[ids]

        config = QuantConfig(bits=8, group_size=64, symmetric=True, method="min_max")
        cw = quantize_weight(weight, config)

        module = nn.Embedding(1000, 64)
        pack_embedding_for_cuda(module, {"weight": cw})
        module.to("cuda")
        packed_out = module(ids.cuda())

        rel_error = (
            packed_out.cpu().float() - original.float()
        ).abs().mean() / original.float().abs().mean()
        self.assertLess(rel_error.item(), 0.02)

    def test_rejects_4bit(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        cw = quantize_weight(torch.randn(100, 64, dtype=torch.bfloat16), config)
        module = nn.Embedding(100, 64)
        with self.assertRaises(ValueError):
            pack_embedding_for_cuda(module, {"weight": cw})


class TestLoadAndPackForCuda(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_pack_model_in_memory(self):
        """pack_model works with in-memory dicts (no file I/O)."""
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        unq = {"norm.weight": torch.randn(64, dtype=torch.bfloat16)}

        with torch.device("meta"):
            model = nn.ModuleDict(
                {
                    "proj": nn.Linear(128, 64, bias=False),
                    "norm": nn.LayerNorm(64, bias=False),
                }
            )
        pack_model(model, {"proj.weight": cw}, unq, DEFAULT_CUDA_PACKERS)

        self.assertEqual(model.proj.weight.shape, torch.Size([64, 128]))
        self.assertEqual(model.norm.weight.shape, torch.Size([64]))

    def test_pack_model_mixed_precision(self):
        """pack_model handles 4-bit and 8-bit weights in the same model."""
        q4_config = QuantConfig(
            bits=4, group_size=32, symmetric=False, method="min_max"
        )
        q8_config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        cw4 = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), q4_config)
        cw8 = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), q8_config)

        with torch.device("meta"):
            model = nn.ModuleDict(
                {
                    "q_proj": nn.Linear(128, 64, bias=False),
                    "v_proj": nn.Linear(128, 64, bias=False),
                }
            )
        pack_model(
            model,
            {"q_proj.weight": cw4, "v_proj.weight": cw8},
            {},
            DEFAULT_CUDA_PACKERS,
        )

        self.assertEqual(model.q_proj.weight.shape, torch.Size([64, 128]))
        self.assertEqual(model.v_proj.weight.shape, torch.Size([64, 128]))
        # Verify different subclass types
        self.assertNotEqual(
            type(model.q_proj.weight.data).__name__,
            type(model.v_proj.weight.data).__name__,
        )

    def test_dispatches_by_module_type(self):
        """load_and_pack_for_cuda reads from disk and dispatches."""
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"proj.weight": cw}, {}, path)

            with torch.device("meta"):
                model2 = nn.ModuleDict({"proj": nn.Linear(128, 64, bias=False)})
            load_and_pack_for_cuda(path, model2)

        self.assertEqual(model2.proj.weight.shape, torch.Size([64, 128]))
        self.assertEqual(model2.proj.weight.device.type, "cpu")

    def test_unknown_module_type_raises(self):
        """Unregistered module types get a clear error."""

        class CustomModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(32, 64))

        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = quantize_weight(torch.randn(32, 64, dtype=torch.bfloat16), config)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"custom.weight": cw}, {}, path)

            with torch.device("meta"):
                model2 = nn.ModuleDict({"custom": CustomModule()})
            with self.assertRaises(ValueError) as ctx:
                load_and_pack_for_cuda(path, model2)
            self.assertIn("CustomModule", str(ctx.exception))

    def test_missing_weight_raises(self):
        """A meta-device parameter after loading means the checkpoint was incomplete."""
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = quantize_weight(torch.randn(32, 64, dtype=torch.bfloat16), config)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            # Only save weight for 'a', not 'b'
            save({"a.weight": cw}, {}, path)

            with torch.device("meta"):
                model2 = nn.ModuleDict(
                    {
                        "a": nn.Linear(64, 32, bias=False),
                        "b": nn.Linear(64, 32, bias=False),
                    }
                )
            with self.assertRaises(RuntimeError) as ctx:
                load_and_pack_for_cuda(path, model2)
            self.assertIn("b.weight", str(ctx.exception))

    def test_custom_packer_via_dict(self):
        """Models can extend the packer dict with custom module types."""
        call_log = []

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(32, 64))

        def my_packer(module, weights):
            call_log.append(("my_packer", list(weights.keys())))
            cw = weights["weight"]
            module.weight = nn.Parameter(
                cw.qdata.to(torch.bfloat16), requires_grad=False
            )

        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = quantize_weight(torch.randn(32, 64, dtype=torch.bfloat16), config)

        custom_packers = {**DEFAULT_CUDA_PACKERS, MyModule: my_packer}

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save({"m.weight": cw}, {}, path)

            with torch.device("meta"):
                model2 = nn.ModuleDict({"m": MyModule()})
            load_and_pack_for_cuda(path, model2, packers=custom_packers)

        self.assertEqual(len(call_log), 1)
        self.assertEqual(call_log[0], ("my_packer", ["weight"]))
        self.assertEqual(model2.m.weight.device.type, "cpu")

    def test_multi_weight_module_grouped(self):
        """pack_model groups multiple weights per module (MoE-style)."""
        call_log = []

        class FusedExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Parameter(torch.randn(32, 64))
                self.w2 = nn.Parameter(torch.randn(32, 64))

        def moe_packer(module, weights):
            call_log.append(sorted(weights.keys()))
            for attr, cw in weights.items():
                setattr(
                    module,
                    attr,
                    nn.Parameter(cw.qdata.to(torch.bfloat16), requires_grad=False),
                )

        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw1 = quantize_weight(torch.randn(32, 64, dtype=torch.bfloat16), config)
        cw2 = quantize_weight(torch.randn(32, 64, dtype=torch.bfloat16), config)

        with torch.device("meta"):
            model = nn.ModuleDict({"experts": FusedExperts()})

        packers = {**DEFAULT_CUDA_PACKERS, FusedExperts: moe_packer}
        pack_model(
            model,
            {"experts.w1": cw1, "experts.w2": cw2},
            {},
            packers,
        )

        # Packer should be called ONCE with both weights
        self.assertEqual(len(call_log), 1)
        self.assertEqual(call_log[0], ["w1", "w2"])


class TestPackOne(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_quantized_weight(self):
        """pack_one dispatches CQW to the module packer."""
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)

        with torch.device("meta"):
            model = nn.ModuleDict({"proj": nn.Linear(128, 64, bias=False)})
        pack_one(model, "proj.weight", cw, DEFAULT_CUDA_PACKERS)

        self.assertNotEqual(model.proj.weight.device.type, "meta")
        self.assertEqual(model.proj.weight.shape, torch.Size([64, 128]))

    def test_plain_tensor(self):
        """pack_one assigns a plain tensor as a parameter or buffer."""
        with torch.device("meta"):
            model = nn.ModuleDict({"norm": nn.LayerNorm(64, bias=False)})
        pack_one(
            model,
            "norm.weight",
            torch.randn(64, dtype=torch.bfloat16),
            DEFAULT_CUDA_PACKERS,
        )

        self.assertEqual(model.norm.weight.shape, torch.Size([64]))
        self.assertEqual(model.norm.weight.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
