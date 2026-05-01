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
    pack_linear_for_cuda,
    pack_model,
)
from executorch.examples.models.gemma4_31b.quant.quantize import quantize_weight
from executorch.examples.models.gemma4_31b.quant.recipe import QuantConfig
from safetensors.torch import save_file
from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict


class TestPackInt4ForCuda(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_basic(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        q = quantize_weight(torch.randn(128, 256, dtype=torch.bfloat16), config)
        self.assertEqual(pack_int4_for_cuda(q).shape, torch.Size([128, 256]))

    def test_different_group_sizes(self):
        for gs in (32, 64, 128):
            with self.subTest(group_size=gs):
                config = QuantConfig(
                    bits=4, group_size=gs, symmetric=False, method="min_max"
                )
                q = quantize_weight(torch.randn(128, 256, dtype=torch.bfloat16), config)
                self.assertEqual(pack_int4_for_cuda(q).shape, torch.Size([128, 256]))

    def test_matmul_approximates_original(self):
        torch.manual_seed(0)
        weight = torch.randn(256, 1024, dtype=torch.bfloat16)
        x = torch.randn(1, 1024, dtype=torch.bfloat16)
        original_out = torch.nn.functional.linear(x.cuda(), weight.cuda())

        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        q = quantize_weight(weight, config)
        packed = pack_int4_for_cuda(q)
        packed_out = torch.nn.functional.linear(x.cuda(), packed.cuda())

        rel_error = (
            packed_out.float() - original_out.float()
        ).abs().mean() / original_out.float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)

    def test_symmetric_matmul(self):
        torch.manual_seed(0)
        weight = torch.randn(256, 1024, dtype=torch.bfloat16)
        x = torch.randn(1, 1024, dtype=torch.bfloat16)
        original_out = torch.nn.functional.linear(x.cuda(), weight.cuda())

        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        q = quantize_weight(weight, config)
        packed = pack_int4_for_cuda(q)
        packed_out = torch.nn.functional.linear(x.cuda(), packed.cuda())

        rel_error = (
            packed_out.float() - original_out.float()
        ).abs().mean() / original_out.float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)


class TestPackInt8OnCuda(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_matmul_approximates_original(self):
        torch.manual_seed(0)
        weight = torch.randn(256, 128, dtype=torch.bfloat16)
        x = torch.randn(1, 128, dtype=torch.bfloat16)
        original_out = torch.nn.functional.linear(x.cuda(), weight.cuda())

        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        q = quantize_weight(weight, config)
        # IntxUnpackedToInt8Tensor is already the CUDA format
        emb = nn.Linear(128, 256, bias=False)
        emb.weight = nn.Parameter(q, requires_grad=False)
        emb.to("cuda")
        packed_out = emb(x.cuda())

        rel_error = (
            packed_out.float() - original_out.float()
        ).abs().mean() / original_out.float().abs().mean()
        self.assertLess(rel_error.item(), 0.02)

    def test_per_axis_embedding_gather(self):
        torch.manual_seed(0)
        weight = torch.randn(1000, 64, dtype=torch.bfloat16)
        ids = torch.tensor([0, 1, 42, 500, 999])
        original = weight[ids]

        config = QuantConfig(bits=8, group_size=64, symmetric=True, method="min_max")
        q = quantize_weight(weight, config)
        emb = nn.Embedding(1000, 64)
        emb.weight = nn.Parameter(q, requires_grad=False)
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

    def test_4bit(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        q = quantize_weight(torch.randn(256, 128, dtype=torch.bfloat16), config)
        module = nn.Linear(128, 256, bias=False)
        pack_linear_for_cuda(module, {"weight": q})
        self.assertEqual(module.weight.shape, torch.Size([256, 128]))

    def test_8bit(self):
        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        q = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        module = nn.Linear(128, 64, bias=False)
        pack_linear_for_cuda(module, {"weight": q})
        self.assertEqual(module.weight.shape, torch.Size([64, 128]))


class TestPackEmbeddingForCuda(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_int8_gather(self):
        torch.manual_seed(0)
        weight = torch.randn(1000, 64, dtype=torch.bfloat16)
        ids = torch.tensor([0, 1, 42, 500, 999])
        original = weight[ids]

        config = QuantConfig(bits=8, group_size=64, symmetric=True, method="min_max")
        q = quantize_weight(weight, config)
        module = nn.Embedding(1000, 64)
        pack_embedding_for_cuda(module, {"weight": q})
        module.to("cuda")
        packed_out = module(ids.cuda())

        rel_error = (
            packed_out.cpu().float() - original.float()
        ).abs().mean() / original.float().abs().mean()
        self.assertLess(rel_error.item(), 0.02)

    def test_rejects_4bit(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        q = quantize_weight(torch.randn(100, 64, dtype=torch.bfloat16), config)
        module = nn.Embedding(100, 64)
        with self.assertRaises(ValueError):
            pack_embedding_for_cuda(module, {"weight": q})


class TestPackModel(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_mixed_precision(self):
        """pack_model handles 4-bit and 8-bit weights in the same model."""
        q4_config = QuantConfig(
            bits=4, group_size=32, symmetric=False, method="min_max"
        )
        q8_config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        q4 = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), q4_config)
        q8 = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), q8_config)

        with torch.device("meta"):
            model = nn.ModuleDict(
                {
                    "q_proj": nn.Linear(128, 64, bias=False),
                    "v_proj": nn.Linear(128, 64, bias=False),
                }
            )
        pack_model(
            model, {"q_proj.weight": q4, "v_proj.weight": q8}, DEFAULT_CUDA_PACKERS
        )
        self.assertEqual(model.q_proj.weight.shape, torch.Size([64, 128]))
        self.assertEqual(model.v_proj.weight.shape, torch.Size([64, 128]))

    def test_load_and_pack(self):
        """load_and_pack_for_cuda reads from disk and packs."""
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        q = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            state = {
                "proj.weight": q,
                "norm.weight": torch.randn(64, dtype=torch.bfloat16),
            }
            td, md = flatten_tensor_state_dict(state)
            save_file(td, path, metadata=md)

            with torch.device("meta"):
                model = nn.ModuleDict(
                    {
                        "proj": nn.Linear(128, 64, bias=False),
                        "norm": nn.LayerNorm(64, bias=False),
                    }
                )
            load_and_pack_for_cuda(path, model)
        self.assertEqual(model.proj.weight.shape, torch.Size([64, 128]))
        self.assertEqual(model.norm.weight.shape, torch.Size([64]))


class TestPackOne(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_quantized_weight(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        q = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)

        with torch.device("meta"):
            model = nn.ModuleDict({"proj": nn.Linear(128, 64, bias=False)})
        pack_one(model, "proj.weight", q, DEFAULT_CUDA_PACKERS)
        self.assertNotEqual(model.proj.weight.device.type, "meta")

    def test_plain_tensor(self):
        with torch.device("meta"):
            model = nn.ModuleDict({"norm": nn.LayerNorm(64, bias=False)})
        pack_one(
            model,
            "norm.weight",
            torch.randn(64, dtype=torch.bfloat16),
            DEFAULT_CUDA_PACKERS,
        )
        self.assertEqual(model.norm.weight.dtype, torch.bfloat16)


class TestPackErrorPaths(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_unregistered_module_type(self):
        """pack_model raises for module types not in packers dict."""

        class CustomModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(32, 64))

        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        q = quantize_weight(torch.randn(32, 64, dtype=torch.bfloat16), config)

        with torch.device("meta"):
            model = nn.ModuleDict({"custom": CustomModule()})
        with self.assertRaises(ValueError) as ctx:
            pack_model(model, {"custom.weight": q}, DEFAULT_CUDA_PACKERS)
        self.assertIn("CustomModule", str(ctx.exception))

    def test_missing_weight_detected(self):
        """pack_model raises when a parameter stays on meta after packing."""
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        q = quantize_weight(torch.randn(32, 64, dtype=torch.bfloat16), config)

        with torch.device("meta"):
            model = nn.ModuleDict(
                {
                    "a": nn.Linear(64, 32, bias=False),
                    "b": nn.Linear(64, 32, bias=False),
                }
            )
        with self.assertRaises(RuntimeError) as ctx:
            pack_model(model, {"a.weight": q}, DEFAULT_CUDA_PACKERS)
        self.assertIn("b.weight", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
