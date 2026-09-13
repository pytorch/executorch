# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import Counter

import torch
import torch.nn as nn

from executorch.backends.apple.coreai.partition.partitioner import CoreAIPartitioner
from executorch.backends.apple.coreai.quantizer.quantizer import CoreAIQuantizer
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import (
    executorch_call_delegate,
    get_lowered_backend_modules,
)


def _coreai_quant_op_names(graph_module):
    """Names of call_function targets in a converted graph (test helper)."""
    return [
        getattr(node.target, "__name__", str(node.target))
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    ]


def _model():
    return nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)).eval()


class _ConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def _full_finalize(model, example_inputs):
    """coreai_opt standalone finalize(CoreAI) for equivalence checks."""
    from coreai_opt.common import ExportBackend
    from coreai_opt.quantization import Quantizer, QuantizerConfig

    q = Quantizer(model, QuantizerConfig())
    prepared = q.prepare(example_inputs)
    with q.calibration_mode():
        prepared(*example_inputs)
    return q.finalize(backend=ExportBackend.CoreAI)


class CoreAIQuantizerTest(unittest.TestCase):
    def setUp(self):
        self.example_inputs = (torch.randn(2, 32),)

    def _convert(self, model):
        q = CoreAIQuantizer(model)
        prepared = q.prepare(self.example_inputs)
        with q.calibration_mode():
            prepared(*self.example_inputs)
        return q.convert()

    def test_convert_produces_coreai_quant_ops(self):
        names = set(_coreai_quant_op_names(self._convert(_model())))
        self.assertIn("quantize", names)
        self.assertIn("dequantize", names)
        self.assertIn("constexpr_blockwise_shift_scale", names)

    def test_convert_matches_full_finalize(self):
        # convert() IS the full coreai_opt finalize(CoreAI).
        converted = self._convert(_model())
        full = _full_finalize(_model(), self.example_inputs)
        self.assertEqual(
            Counter(_coreai_quant_op_names(converted)),
            Counter(_coreai_quant_op_names(full)),
        )

    def test_convert_output_is_exportable(self):
        # The key property: no fake-quant remains, so strict torch.export works.
        converted = self._convert(_model())
        ep = torch.export.export(converted, self.example_inputs)
        namespaces = {
            getattr(n.target, "namespace", None)
            for n in ep.graph.nodes
            if n.op == "call_function"
        }
        # Quant ops are lowered into the coreai namespace after export.
        self.assertIn("coreai", namespaces)

    def test_prepare_is_required_before_convert(self):
        q = CoreAIQuantizer(_model())
        with self.assertRaises(RuntimeError):
            q.convert()

    def test_prepare_is_required_before_the_mode_context_managers(self):
        """Otherwise coreai_opt's own error surfaces instead of ours."""
        for method in ("calibration_mode", "training_mode"):
            with self.subTest(method):
                q = CoreAIQuantizer(_model())
                with self.assertRaisesRegex(RuntimeError, f"before {method}"):
                    getattr(q, method)()


class CoreAIQuantizerConvBNTest(unittest.TestCase):
    def setUp(self):
        self.example_inputs = (torch.randn(1, 3, 16, 16),)

    def _convert(self, model):
        q = CoreAIQuantizer(model)
        prepared = q.prepare(self.example_inputs)
        with q.calibration_mode():
            prepared(*self.example_inputs)
        return q.convert()

    def test_convert_folds_conv_bn(self):
        # convert() folds conv+bn (pre-export) so no batch_norm survives.
        names = _coreai_quant_op_names(self._convert(_ConvBN().eval()))
        self.assertTrue(any("conv" in n for n in names), names)
        self.assertFalse(
            any("batch_norm" in n for n in names),
            f"batch_norm was not folded: {names}",
        )

    def test_convert_matches_full_finalize_conv_bn(self):
        converted = self._convert(_ConvBN().eval())
        full = _full_finalize(_ConvBN().eval(), self.example_inputs)
        self.assertEqual(
            Counter(_coreai_quant_op_names(converted)),
            Counter(_coreai_quant_op_names(full)),
        )


class CoreAIQuantizedLoweringTest(unittest.TestCase):
    """Full quantized lowering: quantize -> export -> to_edge_transform_and_lower."""

    def setUp(self):
        self.example_inputs = (torch.randn(2, 32),)

    def _lower(self):
        q = CoreAIQuantizer(_model())
        prepared = q.prepare(self.example_inputs)
        with q.calibration_mode():
            prepared(*self.example_inputs)
        ep = torch.export.export(q.convert(), self.example_inputs)
        return to_edge_transform_and_lower(ep, partitioner=[CoreAIPartitioner()])

    def test_quantized_linear_lowers_to_coreai(self):
        lowered = self._lower()
        gm = lowered.exported_program().graph_module

        delegates = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is executorch_call_delegate
        ]
        leftover = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is not executorch_call_delegate
            and "getitem" not in str(n.target)
        ]
        # The quantized subgraph (incl. coreai:: quant ops) is delegated and
        # converted by coreai-torch, leaving nothing outside the delegate.
        self.assertGreaterEqual(len(delegates), 1)
        self.assertEqual(
            leftover,
            [],
            f"unexpected ops left outside the delegate: "
            f"{[str(n.target) for n in leftover]}",
        )

    def test_quantized_lowers_through_to_executorch(self):
        lowered = self._lower()
        # A single CoreAI delegate, with its asset embedded (NDS present).
        lbms = get_lowered_backend_modules(lowered.exported_program().graph_module)
        self.assertEqual([lbm.backend_id for lbm in lbms], ["CoreAIBackend"])
        self.assertIsNotNone(lbms[0].named_data_store_output)
        # It lowers all the way to a non-empty .pte.
        buffer = bytes(lowered.to_executorch().buffer)
        self.assertGreater(len(buffer), 0)


if __name__ == "__main__":
    unittest.main()
