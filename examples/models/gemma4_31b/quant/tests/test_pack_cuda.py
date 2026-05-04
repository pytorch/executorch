# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for quant/pack_cuda.py.

Tests the public API contract: after packing, modules produce correct
output via F.linear / nn.Embedding at various batch sizes and configs.
"""

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
    pack_linear_for_cuda,
    pack_model,
)
from executorch.examples.models.gemma4_31b.quant.quantize import quantize_weight
from executorch.examples.models.gemma4_31b.quant.recipe import QuantConfig
from safetensors.torch import save_file
from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict


def _require_cuda(tc: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        tc.skipTest("CUDA required")


class TestPackLinearInt4(unittest.TestCase):
    """pack_linear_for_cuda with INT4 weights produces correct F.linear output."""

    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(0)
        self.weight = torch.randn(256, 1024, dtype=torch.bfloat16)

    def _pack(self, symmetric=False, group_size=32):
        config = QuantConfig(
            bits=4, group_size=group_size, symmetric=symmetric, method="min_max"
        )
        q = quantize_weight(self.weight, config)
        module = nn.Linear(1024, 256, bias=False)
        pack_linear_for_cuda(module, {"weight": q})
        module.cuda()
        return module

    def test_shape_preserved(self):
        module = self._pack()
        self.assertEqual(module.weight.shape, torch.Size([256, 1024]))

    def test_asymmetric_decode(self):
        module = self._pack(symmetric=False)
        x = torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda")
        ref = torch.nn.functional.linear(x, self.weight.cuda())
        out = module(x)
        rel_error = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)

    def test_symmetric_decode(self):
        module = self._pack(symmetric=True)
        x = torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda")
        ref = torch.nn.functional.linear(x, self.weight.cuda())
        out = module(x)
        rel_error = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)

    def test_prefill_batch(self):
        module = self._pack(symmetric=False)
        x = torch.randn(64, 1024, dtype=torch.bfloat16, device="cuda")
        ref = torch.nn.functional.linear(x, self.weight.cuda())
        out = module(x)
        rel_error = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)

    def test_different_group_sizes(self):
        for gs in (32, 64, 128):
            with self.subTest(group_size=gs):
                module = self._pack(group_size=gs)
                x = torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda")
                ref = torch.nn.functional.linear(x, self.weight.cuda())
                out = module(x)
                rel_error = (
                    out.float() - ref.float()
                ).abs().mean() / ref.float().abs().mean()
                self.assertLess(rel_error.item(), 0.15)


class TestPackLinearInt8(unittest.TestCase):
    """pack_linear_for_cuda with INT8 weights produces correct F.linear output."""

    def setUp(self):
        _require_cuda(self)

    def test_matmul_correct(self):
        torch.manual_seed(0)
        weight = torch.randn(256, 128, dtype=torch.bfloat16)
        x = torch.randn(1, 128, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x.cuda(), weight.cuda())

        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        q = quantize_weight(weight, config)
        module = nn.Linear(128, 256, bias=False)
        pack_linear_for_cuda(module, {"weight": q})
        module.cuda()
        out = module(x.cuda())

        rel_error = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_error.item(), 0.02)

    def test_unsupported_type_raises(self):
        module = nn.Linear(64, 32, bias=False)
        with self.assertRaises(ValueError):
            pack_linear_for_cuda(module, {"weight": torch.randn(32, 64)})


class TestPackEmbedding(unittest.TestCase):
    """pack_embedding_for_cuda with INT8 per-axis weights."""

    def setUp(self):
        _require_cuda(self)

    def test_gather_correct(self):
        torch.manual_seed(0)
        weight = torch.randn(1000, 64, dtype=torch.bfloat16)
        ids = torch.tensor([0, 1, 42, 500, 999])
        ref = weight[ids]

        config = QuantConfig(bits=8, group_size=64, symmetric=True, method="min_max")
        q = quantize_weight(weight, config)
        module = nn.Embedding(1000, 64)
        pack_embedding_for_cuda(module, {"weight": q})
        module.cuda()
        out = module(ids.cuda())

        rel_error = (
            out.cpu().float() - ref.float()
        ).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_error.item(), 0.02)

    def test_rejects_4bit(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        q = quantize_weight(torch.randn(100, 64, dtype=torch.bfloat16), config)
        module = nn.Embedding(100, 64)
        with self.assertRaises(ValueError):
            pack_embedding_for_cuda(module, {"weight": q})


class TestPackModel(unittest.TestCase):
    """pack_model handles mixed-precision models and disk loading."""

    def setUp(self):
        _require_cuda(self)

    def test_mixed_precision(self):
        torch.manual_seed(0)
        w4 = torch.randn(64, 128, dtype=torch.bfloat16)
        w8 = torch.randn(64, 128, dtype=torch.bfloat16)
        q4 = quantize_weight(
            w4,
            QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max"),
        )
        q8 = quantize_weight(
            w8,
            QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max"),
        )
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
        model.cuda()
        x = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")

        ref4 = torch.nn.functional.linear(x, w4.cuda())
        out4 = model.q_proj(x)
        self.assertLess(
            (out4.float() - ref4.float()).abs().mean().item()
            / ref4.float().abs().mean().item(),
            0.15,
        )

        ref8 = torch.nn.functional.linear(x, w8.cuda())
        out8 = model.v_proj(x)
        self.assertLess(
            (out8.float() - ref8.float()).abs().mean().item()
            / ref8.float().abs().mean().item(),
            0.02,
        )

    def test_load_and_pack_from_disk(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 128, dtype=torch.bfloat16)
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        q = quantize_weight(weight, config)

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

        model.proj.cuda()
        x = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
        ref = torch.nn.functional.linear(x, weight.cuda())
        out = model.proj(x)
        rel_error = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)

    def test_pack_one_quantized(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        q = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        with torch.device("meta"):
            model = nn.ModuleDict({"proj": nn.Linear(128, 64, bias=False)})
        pack_one(model, "proj.weight", q, DEFAULT_CUDA_PACKERS)
        self.assertNotEqual(model.proj.weight.device.type, "meta")

    def test_pack_one_plain_tensor(self):
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
        _require_cuda(self)

    def test_unregistered_module_type(self):
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
