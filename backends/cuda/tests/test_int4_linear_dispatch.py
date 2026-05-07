# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for use_tinygemm_linears source transform.

Model-agnostic: uses simple nn.Module with INT4, INT8 per-group, and
INT8 per-axis quantized nn.Linear weights.  Requires CUDA.
"""

import unittest

import torch
import torch.nn as nn

from executorch.backends.cuda.transforms.int4_linear_dispatch import (
    use_tinygemm_linears,
)
from executorch.examples.models.gemma4_31b.quant.pack_cuda import int4_tensor_to_intx
from executorch.examples.models.gemma4_31b.quant.quantize import quantize_weight
from executorch.examples.models.gemma4_31b.quant.recipe import QuantConfig


def _require_cuda(tc: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        tc.skipTest("CUDA required")


def _make_model():
    """Build a model with INT4, INT8 per-group, and INT8 per-axis linears."""
    torch.manual_seed(42)

    int4_cfg = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
    int8_per_group_cfg = QuantConfig(
        bits=8, group_size=32, symmetric=True, method="min_max"
    )
    int8_per_axis_cfg = QuantConfig(
        bits=8, group_size=128, symmetric=True, method="min_max"
    )

    model = nn.ModuleDict(
        {
            "proj_int4": nn.Linear(128, 256, bias=False),
            "proj_int8_group": nn.Linear(128, 64, bias=False),
            "proj_int8_axis": nn.Linear(128, 64, bias=False),
        }
    )

    w4 = quantize_weight(torch.randn(256, 128, dtype=torch.bfloat16), int4_cfg)
    w8g = quantize_weight(
        torch.randn(64, 128, dtype=torch.bfloat16), int8_per_group_cfg
    )
    w8a = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), int8_per_axis_cfg)

    model.proj_int4.weight = nn.Parameter(int4_tensor_to_intx(w4), requires_grad=False)
    model.proj_int8_group.weight = nn.Parameter(w8g, requires_grad=False)
    model.proj_int8_axis.weight = nn.Parameter(w8a, requires_grad=False)

    return model


class TestUseTinygemmLinears(unittest.TestCase):
    def setUp(self):
        _require_cuda(self)
        self.model = _make_model()

    def test_converts_int4(self):
        use_tinygemm_linears(self.model)
        self.assertEqual(self.model.proj_int4.weight.shape, torch.Size([256, 128]))

    def test_skips_int8_per_group(self):
        """INT8 per-group weights (K > gs) are not converted."""
        self.model.cuda()
        x = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            out_before = self.model.proj_int8_group(x).clone()
        use_tinygemm_linears(self.model)
        with torch.no_grad():
            out_after = self.model.proj_int8_group(x)
        self.assertTrue(torch.equal(out_before, out_after))

    def test_skips_int8_per_axis(self):
        """INT8 per-axis weights (K == gs) are not converted."""
        self.model.cuda()
        x = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            out_before = self.model.proj_int8_axis(x).clone()
        use_tinygemm_linears(self.model)
        with torch.no_grad():
            out_after = self.model.proj_int8_axis(x)
        self.assertTrue(torch.equal(out_before, out_after))

    def test_idempotent(self):
        use_tinygemm_linears(self.model)
        use_tinygemm_linears(self.model)
        self.assertEqual(self.model.proj_int4.weight.shape, torch.Size([256, 128]))

    def test_matmul_matches_dequant(self):
        """Tinygemm output matches the default dequant+cuBLAS output."""
        self.model.cuda()
        x = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            out_dequant = self.model.proj_int4(x)

        use_tinygemm_linears(self.model)
        with torch.no_grad():
            out_tinygemm = self.model.proj_int4(x)

        rel_error = (
            out_tinygemm.float() - out_dequant.float()
        ).abs().mean() / out_dequant.float().abs().mean()
        self.assertLess(rel_error.item(), 0.05)

    def test_works_on_cuda_model(self):
        """Transform works when model is already on CUDA."""
        self.model.cuda()
        use_tinygemm_linears(self.model)
        x = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            out = self.model.proj_int4(x)
        self.assertEqual(out.shape, torch.Size([1, 256]))
        self.assertFalse(out.isnan().any())


if __name__ == "__main__":
    unittest.main()
