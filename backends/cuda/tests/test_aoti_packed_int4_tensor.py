# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from executorch.backends.cuda.aoti_packed_int4_tensor import (
    AotiPackedInt4Tensor,
    pack_int4_weights_for_aoti,
)
from executorch.extension.llm.export.quantize import quantize_model_


def _make_quantized_linear(device="cpu"):
    module = nn.Linear(64, 32, bias=False, dtype=torch.bfloat16, device=device)
    quantize_model_(
        module,
        qlinear_config="4w",
        qlinear_group_size=32,
    )
    return module


class TestAotiPackedInt4Tensor(unittest.TestCase):
    def test_pack_halves_weight_storage(self):
        module = _make_quantized_linear()
        unpacked = module.weight
        expected = unpacked.dequantize()

        self.assertEqual(pack_int4_weights_for_aoti(module), 1)
        packed = module.weight

        self.assertIsInstance(packed, AotiPackedInt4Tensor)
        self.assertEqual(packed.qdata.shape, (32, 32))
        torch.testing.assert_close(packed.dequantize(), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required")
    def test_eager_linear_uses_packed_weight(self):
        torch.manual_seed(42)
        module = _make_quantized_linear("cuda")
        inputs = torch.randn(7, 64, dtype=torch.bfloat16, device="cuda")
        expected = module(inputs)

        self.assertEqual(pack_int4_weights_for_aoti(module), 1)
        actual = module(inputs)

        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required")
    def test_eager_single_row_uses_packed_weight(self):
        torch.manual_seed(42)
        module = _make_quantized_linear("cuda")
        inputs = torch.randn(1, 64, dtype=torch.bfloat16, device="cuda")
        expected = module(inputs)

        self.assertEqual(pack_int4_weights_for_aoti(module, use_matvec=True), 1)
        actual = module(inputs)

        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required")
    def test_export_contains_triton_int4_matmul(self):
        module = _make_quantized_linear("cuda")
        pack_int4_weights_for_aoti(module)
        inputs = (torch.randn(4, 64, dtype=torch.bfloat16, device="cuda"),)

        exported = torch.export.export(module, inputs, strict=True).run_decompositions(
            {}
        )

        self.assertIn("triton.int4_matmul.default", exported.graph_module.code)
        self.assertNotIn("triton.int4_matvec_bf16.default", exported.graph_module.code)
        self.assertNotIn("dequantize_affine", exported.graph_module.code)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required")
    def test_single_row_export_defaults_to_triton_int4_matmul(self):
        module = _make_quantized_linear("cuda")
        pack_int4_weights_for_aoti(module)
        inputs = (torch.randn(1, 64, dtype=torch.bfloat16, device="cuda"),)

        exported = torch.export.export(module, inputs, strict=True).run_decompositions(
            {}
        )

        self.assertIn("triton.int4_matmul.default", exported.graph_module.code)
        self.assertNotIn("triton.int4_matvec_bf16.default", exported.graph_module.code)
        self.assertNotIn("dequantize_affine", exported.graph_module.code)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required")
    def test_single_row_export_can_select_triton_int4_matvec(self):
        module = _make_quantized_linear("cuda")
        pack_int4_weights_for_aoti(module, use_matvec=True)
        inputs = (torch.randn(1, 64, dtype=torch.bfloat16, device="cuda"),)

        exported = torch.export.export(module, inputs, strict=True).run_decompositions(
            {}
        )

        self.assertIn("triton.int4_matvec_bf16.default", exported.graph_module.code)
        self.assertNotIn("triton.int4_matmul.default", exported.graph_module.code)
        self.assertNotIn("dequantize_affine", exported.graph_module.code)


if __name__ == "__main__":
    unittest.main()
